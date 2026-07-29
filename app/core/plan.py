"""타임라인 — 칩만으로 하루의 뼈대를 만든다 (LLM·네트워크 없음).

식사 시각이 칩으로 확정되므로(아침 10 · 점심 13 · 저녁 19시) 장소를 고르기 **전에**
하루가 어떻게 나뉘는지 알 수 있다. 이 골격을 select 프롬프트에 실어 보내면 LLM 이

  - 각 구간에 몇 곳이 들어가는지 알고 장소를 고르고
  - "점심 13시 직후 첫 장소" 라는 맥락을 알고 activities 를 쓰고
  - 구간 길이에 맞춰 체류시간(dwell_min)을 제안한다

예) 창 09~21시, 끼니 [점심, 저녁]

  09:00 ─────── 13:00 ─── 14:00 ─────── 19:00 ─── 20:00 ─── 21:00
  │  세그먼트 A  │  점심   │  세그먼트 B  │  저녁   │ 세그먼트 C │
  │   (4시간)    │  고정   │   (5시간)    │  고정   │  (1시간)  │

끼니를 하나도 안 고르면 세그먼트는 창 전체 하나가 된다 — 그래프를 분기하지 않고
이 함수 안에서만 갈린다 (하위 노드는 "세그먼트 리스트" 하나로 동일하게 동작).
"""
from __future__ import annotations

from app.features.course.schema import CourseChips

# 세그먼트가 이보다 짧으면 장소를 못 넣는다 (이동 + 최소 체류)
MIN_SEGMENT_MIN = 40
# 수용 한계 계산용 — 장소 하나의 최소 체류시간과 구간 내 이동 추정치
MIN_DWELL_MIN = 30
TRAVEL_EST_MIN = 15


def hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def build_skeleton(chips: CourseChips) -> dict:
    """칩 → 하루 골격.

    반환:
      {
        "window": (start_min, end_min) | None,
        "meals":    [{"label","start","end"}],        # 고정 시각 식사 슬롯
        "segments": [{"index","start","end","minutes","after_meal","before_meal"}],
        "stops_per_day": int,
      }
    시간 창이 없으면 빈 골격 — 시간표 없는 코스로 흐른다.
    """
    w = chips.resolved_window()
    per_day = chips.stops_per_day()
    if not w:
        return {"window": None, "meals": [], "segments": [], "stops_per_day": per_day}

    w_start, w_end = w.start * 60, w.end * 60
    meals = [{"label": lab, "start": s, "end": e} for lab, s, e in chips.meal_slots()]

    # 식사 슬롯이 창을 잘라 세그먼트를 만든다 (끼니 0개면 창 전체가 세그먼트 1개)
    segments: list[dict] = []
    cursor = w_start
    prev_meal: str | None = None
    for m in meals:
        if m["start"] - cursor >= MIN_SEGMENT_MIN:
            segments.append(_seg(len(segments), cursor, m["start"], prev_meal, m["label"]))
        cursor = max(cursor, m["end"])
        prev_meal = m["label"]
    if w_end - cursor >= MIN_SEGMENT_MIN:
        segments.append(_seg(len(segments), cursor, w_end, prev_meal, None))

    return {"window": (w_start, w_end), "meals": meals, "segments": segments,
            "stops_per_day": per_day}


def _seg(i: int, start: int, end: int, after: str | None, before: str | None) -> dict:
    return {"index": i, "start": start, "end": end, "minutes": end - start,
            "after_meal": after, "before_meal": before}


def capacity(minutes: int) -> int:
    """이 구간에 물리적으로 넣을 수 있는 최대 장소 수.

    첫 장소는 이동이 없고, 이후는 매번 이동이 붙는다:
        minutes >= MIN_DWELL + (n-1) * (MIN_DWELL + TRAVEL_EST)
    """
    if minutes < MIN_DWELL_MIN:
        return 0
    return 1 + max(0, (minutes - MIN_DWELL_MIN) // (MIN_DWELL_MIN + TRAVEL_EST_MIN))


def allocate_stops(skeleton: dict, n_stops: int) -> tuple[list[int], int]:
    """세그먼트별 장소 배정 → (구간별 배정 수, 못 넣은 수).

    길이에 비례해 나누되 **구간별 수용 한계(capacity)를 넘기지 않는다.**
    60분짜리 구간에 2곳을 넣으면 각 30분에 이동 시간조차 없어 시간표가 깨진다.
    한계 때문에 못 넣은 장소는 flex(시간 없는 자유 방문 제안)로 흘려보낸다.
    """
    segs = skeleton["segments"]
    if not segs or n_stops <= 0:
        return [], max(0, n_stops)

    caps = [capacity(s["minutes"]) for s in segs]

    # 1) 모든 구간에 최소 1곳씩 먼저 깐다 — 순수 비례배분이면 int() 가 0 으로 잘려
    #    짧은 구간이 통째로 비고(예: 12~13시), 긴 구간이 장소를 다 가져가 체류가 짧아진다.
    #    장소가 구간 수보다 적으면 긴 구간부터 하나씩 준다.
    by_len = sorted(range(len(segs)), key=lambda i: -segs[i]["minutes"])
    quota = [0] * len(segs)
    for i in by_len:
        if sum(quota) >= n_stops:
            break
        if caps[i] > 0:
            quota[i] = 1

    # 2) 남은 몫은 길이에 비례해 나눈다
    rest = n_stops - sum(quota)
    total = sum(s["minutes"] for s in segs) or 1
    if rest > 0:
        for i, s in enumerate(segs):
            add = min(caps[i] - quota[i], int(rest * s["minutes"] / total))
            quota[i] += max(0, add)

    # 남은 몫을 긴 구간부터 채우되 한계 안에서만
    order = by_len
    changed = True
    while sum(quota) < n_stops and changed:
        changed = False
        for i in order:
            if sum(quota) >= n_stops:
                break
            if quota[i] < caps[i]:
                quota[i] += 1
                changed = True
    # 비율 계산이 과하게 잡혔으면 짧은 구간부터 덜어낸다
    while sum(quota) > n_stops:
        for i in reversed(order):
            if sum(quota) <= n_stops:
                break
            if quota[i] > 0:
                quota[i] -= 1
    return quota, n_stops - sum(quota)


def describe(skeleton: dict, quota: list[int] | None = None) -> str:
    """LLM 프롬프트용 한 덩어리 — 하루가 어떻게 나뉘는지 보여준다."""
    if not skeleton["segments"] and not skeleton["meals"]:
        return ""
    lines: list[str] = []
    items: list[tuple[int, str]] = []
    for i, s in enumerate(skeleton["segments"]):
        n = f" · 장소 {quota[i]}곳" if quota and i < len(quota) else ""
        items.append((s["start"],
                      f"  {hhmm(s['start'])}~{hhmm(s['end'])}  "
                      f"[구간 {i + 1}] {s['minutes']}분{n}"))
    for m in skeleton["meals"]:
        items.append((m["start"],
                      f"  {hhmm(m['start'])}~{hhmm(m['end'])}  "
                      f"[{m['label']} 식사] 장소는 넣지 말 것 (식당은 서버가 붙인다)"))
    lines += [t for _, t in sorted(items)]
    return "\n".join(lines)
