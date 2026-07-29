"""시간표 계산 — 시간 골격(plan) 위에 확정된 장소를 앉힌다.

순수 계산 모듈 (LLM·네트워크 없음). 방문 순서, 장소별 방문 시각, 이동 시간을
코드로 확정한다 — LLM 은 시간 산술을 자주 틀리므로, 완성된 시간표 위에 서사만
쓰게 한다 (compose).

## 예산 배분 (이 모듈의 핵심)

예전에는 **카테고리 고정 체류시간을 앞에서부터 채우다 넘치면 버렸다.** 그 결과:

    4곳 요청 → 3곳만 앉고 1곳 flex, 창 50분 유휴
    당일치기 3곳 → 14:50 종료, 2시간 40분 공백 뒤 저녁 억지 삽입
    2일 packed → 1일차 17:34 종료, 21시까지 146분 유휴

이제는 **구간(segment)이 먼저 있고, 그 예산 안에 장소를 신축시켜 넣는다.**

    구간 300분, 장소 2곳
      → 이동 추정 40분 제외 = 260분이 체류 예산
      → 희망 체류시간 [90, 120] 비율대로 260분에 맞춤 → [111, 149]
      → 하한(30분) 미만으로 눌리면 그 장소는 flex 로 뺀다

희망 체류시간(dwell_min)은 LLM 이 activities 를 근거로 낸다. 없으면 카테고리
매핑(DWELL_MIN)으로 폴백한다 — 그 매핑은 20%만 적중하므로 주 경로가 아니다.

## 순서 결정

장소가 3~5곳이라 순열 전수 탐색이 가능하다(≤120가지). 각 순서의 총 이동거리를
비교해 최소인 것을 고른다. 운영시간이 안 맞는 장소는 애초에 후보에서 빠지므로
(retrieve 단계 하드 제외) 여기서는 거리만 본다.

거리·이동시간은 직선거리 기반 추정이다 — 실경로/폴리라인은 strangemap 프론트
(courseRouting.ts) 담당.
"""
from __future__ import annotations

from itertools import permutations

from app.core.geo import haversine_km

# 카테고리별 기본 체류시간(분) — **폴백 전용.** 주 경로는 LLM 의 dwell_min 이다.
# 이 매핑은 seoul_places 계열 어휘만 커버해서 전체의 20%만 적중한다
# (쇼핑 225·역사관광 77·박물관 51·미술관 50 등이 전부 기본값으로 떨어진다).
DWELL_MIN: dict[str, int] = {
    "고궁·역사": 90, "관광·역사": 90, "미술관·박물관": 90,
    "상권·역세권": 90, "복합공간·쇼핑": 90,
    "공원·자연": 60, "공원·정원": 60, "공원/광장": 60,
    "한강": 60, "한강·다리": 60, "도심·거리": 60,
    "전망대·산": 45, "야경·전망": 45, "K-콘텐츠 촬영지": 45,
}
_DWELL_DEFAULT = 60
DWELL_FLOOR = 30             # 예산이 빠듯해도 여기까지만 줄인다. 더 눌리면 flex 로 뺀다
DWELL_CEIL = 180             # LLM 이 과하게 부르는 것을 막는 상한

# ── 이동 추정 ──────────────────────────────────────────────────────────────
# 이전 값(15분/km + 10분 버퍼)은 과대 추정이었다 — 직선 1.02km 가 25분으로 잡혀
# 5곳 코스면 이동만 100분+ 이 창을 갉아먹었다. 실제 도보는 1km 에 13~15분이고
# 직선거리는 실제 경로보다 짧으므로 보정 계수(1.25)로 흡수한다.
_WALK_MIN_PER_KM = 13        # 도보 속도 (약 4.6km/h)
_ROUTE_FACTOR = 1.25         # 직선거리 → 실제 도로거리 보정
_MOVE_BUFFER_MIN = 5         # 길찾기·신호 여유 (이전 10분에서 축소)
# 보정 후 거리 기준. 1.5km 였을 땐 직선 1.2km(보정 1.5km)도 대중교통으로 잡혀
# "20분 걸으면 되는 길"에 지하철을 태웠다. 2km(도보 26분)까지는 걷는 것으로 본다.
_TRANSIT_THRESHOLD_KM = 2.0
_TRANSIT_MIN_PER_KM = 3      # 지하철·버스 평균 (환승 포함 추정)
_TRANSIT_OVERHEAD_MIN = 12   # 역·정류장 접근과 대기

_MAX_PERM_STOPS = 6          # 이보다 많으면 순열 탐색 생략


def hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def duration_label(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h}시간 {m}분"
    return f"{h}시간" if h else f"{m}분"


def wanted_dwell(stop: dict) -> int:
    """희망 체류시간 — LLM 의 dwell_min 우선, 없으면 카테고리 매핑, 그것도 없으면 60분."""
    raw = stop.get("dwell_min")
    if isinstance(raw, (int, float)) and raw > 0:
        return max(DWELL_FLOOR // 2, min(DWELL_CEIL, int(raw)))
    return DWELL_MIN.get(stop.get("category", ""), _DWELL_DEFAULT)


def travel(a: dict, b: dict) -> tuple[int, str]:
    """구간 이동 (분, 수단). 1.5km 까지는 도보, 그 이상은 대중교통 추정."""
    km = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]) * _ROUTE_FACTOR
    if km <= _TRANSIT_THRESHOLD_KM:
        return round(km * _WALK_MIN_PER_KM) + _MOVE_BUFFER_MIN, "walk"
    return round(km * _TRANSIT_MIN_PER_KM) + _TRANSIT_OVERHEAD_MIN + _MOVE_BUFFER_MIN, "transit"


def order_by_distance(stops: list[dict], anchor: tuple[float, float] | None) -> list[dict]:
    """이동 거리가 최소인 방문 순서. 6곳 이하면 순열 전수, 넘으면 최근접 이웃."""
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
    # 최근접 이웃 (앵커에서 출발)
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
    """한 구간에 장소들을 예산에 맞춰 앉힌다 → (슬롯[], 못 앉힌 장소[]).

    슬롯: {"stop", "start", "end", "dwell_min", "travel_min", "travel_mode"}

    1) 이동 거리 최소 순서를 정한다
    2) 이동 시간 합을 빼서 체류 예산을 구한다
    3) 희망 체류시간의 비율대로 예산을 나눈다
    4) 하한(30분) 미만으로 눌리는 장소가 있으면 뒤에서부터 덜어내 flex 로 보낸다
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
        dropped.append(ordered.pop())        # 뒤에서부터 덜어낸다 (앞이 동선상 우선)
    if not ordered:
        return [], dropped

    # 희망 비율대로 예산 분배 → 하한/상한 클램프
    total_want = sum(wants) or 1
    dwells = [max(DWELL_FLOOR, min(DWELL_CEIL, round(dwell_budget * w / total_want)))
              for w in wants]
    # 클램프로 합이 예산을 넘으면 긴 것부터 깎는다
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
    """하루치 — 구간별로 fit_segment 를 돌려 합친다 → (슬롯[], flex 장소[])."""
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
