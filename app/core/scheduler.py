"""시간표를 짠다. plan 이 만든 시간 뼈대 위에 확정된 장소를 앉히는 일이다.

계산만 하고 AI 나 네트워크는 쓰지 않는다. 방문 순서와 시각, 이동 시간을 코드로 정한다.
AI 는 시간 계산을 자주 틀려서, 다 짜인 시간표 위에 글만 쓰게 한다.

## 시간 나누기

구간이 먼저 있고, 그 시간 안에 장소를 늘리거나 줄여 넣는다.

    구간 300분에 장소 2곳
      → 이동에 40분쯤 걸릴 테니 260분이 머무는 데 쓸 시간
      → 원하는 체류시간이 [90, 120]이면 그 비율대로 260분에 맞춰 [111, 149]
      → 30분 아래로 눌리는 장소는 시간표에서 빼고 "자유 방문"으로 돌린다

원하는 체류시간은 AI 가 제안한 값을 쓰고, 없으면 장소 종류별 기본값을 쓴다.

## 순서 정하기

장소가 3~5곳이라 가능한 순서를 전부(많아야 120가지) 만들어 보고,
총 이동거리가 가장 짧은 것을 고른다.

거리는 직선거리로 어림한다. 실제 도로 경로는 프론트가 그린다.
"""
from __future__ import annotations

from itertools import permutations

from app.core.geo import haversine_km

# 장소 종류별 기본 체류시간(분). AI 가 체류시간을 안 줬을 때만 쓴다.
# 여기 없는 종류는 아래 기본값으로 떨어진다.
DWELL_MIN: dict[str, int] = {
    "고궁·역사": 90, "관광·역사": 90, "미술관·박물관": 90,
    "상권·역세권": 90, "복합공간·쇼핑": 90,
    "공원·자연": 60, "공원·정원": 60, "공원/광장": 60,
    "한강": 60, "한강·다리": 60, "도심·거리": 60,
    "전망대·산": 45, "야경·전망": 45, "K-콘텐츠 촬영지": 45,
}
_DWELL_DEFAULT = 60
DWELL_FLOOR = 30             # 시간이 모자라도 여기까지만 줄인다. 더 눌리면 시간표에서 뺀다
DWELL_CEIL = 180             # AI 가 너무 길게 부르는 것을 막는 상한

# ── 이동 시간 어림잡기 ─────────────────────────────────────────────────────
_WALK_MIN_PER_KM = 13        # 1km 걷는 데 걸리는 시간 (약 4.6km/h)
_ROUTE_FACTOR = 1.25         # 직선거리는 실제 길보다 짧으니 이만큼 늘려 잡는다
_MOVE_BUFFER_MIN = 5         # 길 찾고 신호 기다리는 여유
# 이 거리까지는 걷는 것으로 본다. 2km 면 26분쯤이라 걸을 만하다.
# 더 짧게 잡으면 20분이면 걸어갈 길에 지하철을 태우게 된다.
_TRANSIT_THRESHOLD_KM = 2.0
_TRANSIT_MIN_PER_KM = 3      # 지하철·버스로 1km 가는 데 걸리는 시간
_TRANSIT_OVERHEAD_MIN = 12   # 역까지 걸어가고 기다리는 시간

_MAX_PERM_STOPS = 6          # 장소가 이보다 많으면 순서를 다 따져보지 않는다


def hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def duration_label(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h}시간 {m}분"
    return f"{h}시간" if h else f"{m}분"


def wanted_dwell(stop: dict) -> int:
    """이 장소에 얼마나 머물고 싶은지. AI 가 낸 값을 먼저 쓰고, 없으면 종류별 기본값을 쓴다."""
    raw = stop.get("dwell_min")
    if isinstance(raw, (int, float)) and raw > 0:
        return max(DWELL_FLOOR // 2, min(DWELL_CEIL, int(raw)))
    return DWELL_MIN.get(stop.get("category", ""), _DWELL_DEFAULT)


def travel(a: dict, b: dict) -> tuple[int, str]:
    """두 장소 사이 이동 시간과 수단을 어림한다. 2km 까지는 걷고, 그보다 멀면 대중교통으로 본다."""
    km = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]) * _ROUTE_FACTOR
    if km <= _TRANSIT_THRESHOLD_KM:
        return round(km * _WALK_MIN_PER_KM) + _MOVE_BUFFER_MIN, "walk"
    return round(km * _TRANSIT_MIN_PER_KM) + _TRANSIT_OVERHEAD_MIN + _MOVE_BUFFER_MIN, "transit"


def order_by_distance(stops: list[dict], anchor: tuple[float, float] | None) -> list[dict]:
    """총 이동거리가 가장 짧은 방문 순서를 찾는다.

    6곳까지는 가능한 순서를 다 만들어 보고, 그보다 많으면 가까운 곳부터 차례로 잇는다.
    """
    usable = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    if len(usable) <= 1:
        return usable
    if len(usable) <= _MAX_PERM_STOPS:
        def cost(order: tuple[dict, ...]) -> float:
            d = sum(haversine_km(order[i]["lat"], order[i]["lng"],
                                 order[i + 1]["lat"], order[i + 1]["lng"])
                    for i in range(len(order) - 1))
            if anchor:
                d += haversine_km(anchor[0], anchor[1], order[0]["lat"], order[0]["lng"])
            return d
        return list(min(permutations(usable), key=cost))
    # 장소가 많으면 시작점에서 가장 가까운 곳부터 하나씩 이어 붙인다
    remaining = list(usable)
    cur = anchor or (remaining[0]["lat"], remaining[0]["lng"])
    out: list[dict] = []
    while remaining:
        nxt = min(remaining, key=lambda s: haversine_km(cur[0], cur[1], s["lat"], s["lng"]))
        remaining.remove(nxt)
        out.append(nxt)
        cur = (nxt["lat"], nxt["lng"])
    return out


def fit_segment(
    stops: list[dict], seg_start: int, seg_end: int,
    anchor: tuple[float, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """구간 하나에 장소들을 시간 맞춰 앉힌다. (앉힌 슬롯, 못 앉힌 장소)를 돌려준다.

    1) 이동거리가 가장 짧은 순서를 정하고
    2) 거기서 이동 시간을 빼 머물 수 있는 시간을 구한 다음
    3) 원하는 체류시간 비율대로 나눈다
    4) 30분도 안 되게 눌리는 장소가 있으면 뒤에서부터 덜어낸다
    """
    ordered = order_by_distance(stops, anchor)
    dropped = [s for s in stops if s not in ordered]
    if not ordered:
        return [], dropped

    budget_total = seg_end - seg_start
    while ordered:
        moves = [0] + [travel(ordered[i], ordered[i + 1])[0] for i in range(len(ordered) - 1)]
        modes = [""] + [travel(ordered[i], ordered[i + 1])[1] for i in range(len(ordered) - 1)]
        dwell_budget = budget_total - sum(moves)
        wants = [wanted_dwell(s) for s in ordered]
        if dwell_budget >= DWELL_FLOOR * len(ordered):
            break
        dropped.append(ordered.pop())        # 뒤에서부터 뺀다. 앞쪽이 동선상 더 중요하다
    if not ordered:
        return [], dropped

    # 원하는 시간 비율대로 나눈 뒤, 너무 짧거나 길지 않게 자른다
    total_want = sum(wants) or 1
    dwells = [max(DWELL_FLOOR, min(DWELL_CEIL, round(dwell_budget * w / total_want)))
              for w in wants]
    # 자르고 나서 합이 시간을 넘으면 긴 것부터 깎는다
    over = sum(dwells) - dwell_budget
    while over > 0:
        i = max(range(len(dwells)), key=lambda k: dwells[k])
        if dwells[i] <= DWELL_FLOOR:
            break
        cut = min(over, dwells[i] - DWELL_FLOOR)
        dwells[i] -= cut
        over -= cut

    slots: list[dict] = []
    t = seg_start
    for i, s in enumerate(ordered):
        t += moves[i]
        slots.append({"stop": s, "start": t, "end": t + dwells[i],
                      "dwell_min": dwells[i],
                      "travel_min": moves[i], "travel_mode": modes[i]})
        t += dwells[i]
    return slots, dropped


def build_day(
    stops_by_segment: list[list[dict]], segments: list[dict],
    anchor: tuple[float, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """하루치 시간표를 만든다. 구간마다 fit_segment 를 돌려 이어 붙인다.

    돌려주는 두 번째 목록은 시간이 모자라 못 넣은 장소들이다("자유 방문"으로 나간다).
    """
    slots: list[dict] = []
    flex: list[dict] = []
    prev_last: dict | None = None
    for seg, group in zip(segments, stops_by_segment):
        a = (prev_last["lat"], prev_last["lng"]) if prev_last else anchor
        s, d = fit_segment(group, seg["start"], seg["end"], anchor=a)
        slots += s
        flex += d
        if s:
            prev_last = s[-1]["stop"]
    return slots, flex
