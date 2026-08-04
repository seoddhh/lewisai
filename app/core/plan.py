"""칩만 보고 하루의 시간 뼈대를 만든다. 계산만 하고 AI 나 네트워크는 안 쓴다.

식사 시각은 칩으로 이미 정해져 있어서(아침 10시·점심 13시·저녁 19시),
장소를 고르기 전에도 하루가 어떻게 나뉘는지 알 수 있다.
이 뼈대를 AI 에게 같이 주면 구간에 맞는 장소와 할 일, 체류시간을 제안할 수 있다.

예) 9시~21시에 점심과 저녁을 골랐다면:

  09:00 ─────── 13:00 ─── 14:00 ─────── 19:00 ─── 20:00 ─── 21:00
  │    구간 A    │  점심   │    구간 B    │  저녁   │   구간 C  │
  │   (4시간)    │  고정   │   (5시간)    │  고정   │  (1시간)  │

끼니를 하나도 안 골랐으면 구간은 전체 하나가 된다. 뒤에 오는 노드들은
어느 쪽이든 "구간 목록"만 보면 되므로 따로 갈라질 필요가 없다.
"""
from __future__ import annotations

from app.core.scheduler import hhmm
from app.features.course.schema import CourseChips

# 구간이 이보다 짧으면 장소를 넣지 않는다. 이동하고 잠깐 있기에도 모자란 시간이다
MIN_SEGMENT_MIN = 40
# 구간에 장소가 몇 곳까지 들어가는지 셀 때 쓰는 값
MIN_DWELL_MIN = 30      # 한 곳에 최소 이만큼은 머문다
TRAVEL_EST_MIN = 15     # 장소 사이 이동에 이만큼 걸린다고 본다


def build_skeleton(chips: CourseChips) -> dict:
    """칩을 보고 하루의 시간 뼈대를 만든다.

    식사 시각(meals)과, 장소가 들어갈 구간(segments)이 들어 있다.
    시간 범위가 없는 요청이면 빈 뼈대를 주고, 그런 코스는 시간표 없이 만들어진다.
    """
    w = chips.resolved_window()
    per_day = chips.stops_per_day()
    if not w:
        return {"window": None, "meals": [], "segments": [], "stops_per_day": per_day}

    w_start, w_end = w.start * 60, w.end * 60
    meals = [{"label": lab, "start": s, "end": e} for lab, s, e in chips.meal_slots()]

    # 식사 시각이 하루를 잘라 구간을 만든다. 끼니를 안 골랐으면 전체가 구간 하나가 된다
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
    """이 구간에 최대 몇 곳까지 넣을 수 있는지 센다.

    첫 장소는 바로 시작하지만, 두 번째부터는 이동 시간이 앞에 붙는다.
    """
    if minutes < MIN_DWELL_MIN:
        return 0
    return 1 + max(0, (minutes - MIN_DWELL_MIN) // (MIN_DWELL_MIN + TRAVEL_EST_MIN))


def allocate_stops(skeleton: dict, n_stops: int) -> tuple[list[int], int]:
    """장소 수를 구간마다 나눠 준다. (구간별 개수, 못 넣은 개수)를 돌려준다.

    구간 길이에 비례해 나누되, 구간이 감당할 수 있는 수를 넘기지 않는다.
    60분 구간에 2곳을 넣으면 한 곳당 30분에 이동 시간도 없어 시간표가 깨진다.
    자리가 없어 못 넣은 장소는 "자유 방문"으로 빠진다.
    """
    segs = skeleton["segments"]
    if not segs or n_stops <= 0:
        return [], max(0, n_stops)

    caps = [capacity(s["minutes"]) for s in segs]

    # 1) 먼저 구간마다 1곳씩 깔아 둔다. 그냥 비율로만 나누면 짧은 구간 몫이 0으로 잘려
    #    통째로 비어 버리고, 긴 구간이 장소를 다 가져간다.
    #    장소가 구간 수보다 적으면 긴 구간부터 하나씩 준다.
    by_len = sorted(range(len(segs)), key=lambda i: -segs[i]["minutes"])
    quota = [0] * len(segs)
    for i in by_len:
        if sum(quota) >= n_stops:
            break
        if caps[i] > 0:
            quota[i] = 1

    # 2) 남은 장소는 구간 길이에 비례해 나눈다
    rest = n_stops - sum(quota)
    total = sum(s["minutes"] for s in segs) or 1
    if rest > 0:
        for i, s in enumerate(segs):
            add = min(caps[i] - quota[i], int(rest * s["minutes"] / total))
            quota[i] += max(0, add)

    # 3) 그래도 남으면 긴 구간부터 하나씩 더 넣는다. 감당할 수 있는 만큼만
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
    # 4) 반대로 많이 잡혔으면 짧은 구간부터 덜어낸다
    while sum(quota) > n_stops:
        for i in reversed(order):
            if sum(quota) <= n_stops:
                break
            if quota[i] > 0:
                quota[i] -= 1
    return quota, n_stops - sum(quota)


def describe(skeleton: dict, quota: list[int] | None = None) -> str:
    """하루가 어떻게 나뉘는지를 AI 에게 보여줄 텍스트로 만든다."""
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
