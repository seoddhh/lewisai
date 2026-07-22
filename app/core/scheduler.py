"""시간표 계산 — 시간 범위가 주어진 코스를 "여행 플래너" 수준의 스케줄로.

순수 계산 모듈 (LLM·네트워크 없음). 방문 순서, 장소별 방문 시각, 도보 이동 시간,
식사 슬롯 삽입까지 코드로 확정한다 — LLM 은 시간 산술을 자주 틀리므로,
완성된 시간표 위에 서사만 쓰게 한다 (compose).

순서 결정: 장소가 3~5곳이라 순열 전수 탐색이 가능하다(≤120가지).
각 순서를 시뮬레이션해 "운영시간 안에 앉힌 장소 수 최대 → 이동 거리 최소"인
순서를 고른다. 운영시간이 좁은 장소는 못 앉히는 순서가 탈락하므로 자연히 우선된다.

여기서의 거리·이동시간은 직선거리 기반 추정이다 — 실경로/폴리라인은 여전히
strangemap 프론트(courseRouting.ts) 담당.
"""
from __future__ import annotations

from itertools import permutations

from app.core.geo import haversine_km

# 카테고리별 기본 체류시간(분)
DWELL_MIN: dict[str, int] = {
    "고궁·역사": 90, "관광·역사": 90, "미술관·박물관": 90,
    "상권·역세권": 90, "복합공간·쇼핑": 90,
    "공원·자연": 60, "공원·정원": 60, "공원/광장": 60,
    "한강": 60, "한강·다리": 60, "도심·거리": 60,
    "전망대·산": 45, "야경·전망": 45, "K-콘텐츠 촬영지": 45,
}
_DWELL_DEFAULT = 60
_DWELL_FLOOR = 30            # 남은 시간이 빠듯하면 여기까지 줄이고, 그래도 안 되면 flex 제안
_WALK_MIN_PER_KM = 15        # 직선거리 → 도보 시간 환산
_MOVE_BUFFER_MIN = 10        # 이동마다 붙는 여유 (길찾기·신호 등)
# 도보권을 넘는 구간은 대중교통으로 추정한다 — 8km(종로↔강남)를 걷는 시간표는 비현실적
_TRANSIT_THRESHOLD_KM = 1.5  # 이보다 멀면 도보 대신 대중교통
_TRANSIT_MIN_PER_KM = 3      # 지하철·버스 평균 (환승 포함 추정)
_TRANSIT_OVERHEAD_MIN = 15   # 역·정류장 접근과 대기
_MEAL_MIN = 60
# 끼니 창 — 시간 범위에 13:00/19:00 이 들어오면 점심/저녁 슬롯이 생긴다.
# 창 폭(≥150분)을 최대 체류시간(90분)보다 넓게 둬서, 한 장소 체류가 창 전체를
# 삼켜 끼니가 건너뛰어지는 일이 없게 한다.
#
# 아침은 슬롯을 만들지 않는다. 이유 두 가지:
#  (1) Visit Seoul 음식 데이터에 아침 영업 식당이 사실상 없다 — 한식당 18곳 실측에서
#      09시 이전 개점 0곳, 가장 이른 곳이 10:30, 대다수 11:00~11:30.
#  (2) 오전 시작 코스라도 아침 식사부터 하는 동선은 실제 이용 패턴과 안 맞는다.
# 따라서 09:00 시작을 골라도 첫 끼니는 점심이다.
_MEAL_WINDOWS = (
    (11 * 60 + 30, 14 * 60, "점심"),
    (17 * 60 + 30, 20 * 60, "저녁"),
)
_MAX_PERM_STOPS = 6          # 이보다 많으면 순열 탐색 생략 (현재 place_count 는 3~5)


def hhmm(minutes: int) -> str:
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def duration_label(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    if h and m:
        return f"{h}시간 {m}분"
    return f"{h}시간" if h else f"{m}분"


def _dwell(stop: dict) -> int:
    return DWELL_MIN.get(stop.get("category", ""), _DWELL_DEFAULT)


def _op_minutes(stop: dict) -> tuple[int, int]:
    s = int(stop.get("op_start", 0)) * 60
    e = int(stop.get("op_end", 24)) * 60
    if e <= s:
        e += 24 * 60  # 심야 영업 (예: 5시~익일 1시)
    return s, e


def _travel(a: dict, b: dict) -> tuple[int, str]:
    """구간 이동 (분, 수단). 1.5km 까지는 도보, 그 이상은 대중교통 추정."""
    km = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
    if km <= _TRANSIT_THRESHOLD_KM:
        return round(km * _WALK_MIN_PER_KM) + _MOVE_BUFFER_MIN, "walk"
    return round(km * _TRANSIT_MIN_PER_KM) + _TRANSIT_OVERHEAD_MIN + _MOVE_BUFFER_MIN, "transit"


def _simulate(
    order: list[dict], w_start: int, w_end: int
) -> tuple[list[dict], list[dict], float]:
    """한 방문 순서의 시간표 시뮬레이션 → (슬롯[], 못 앉힌 장소[], 총 이동 km).

    슬롯: {"slot_type":"place","stop":…,"start","end"} 또는
          {"slot_type":"meal","label":"점심|저녁","anchor":장소명,"start","end"}
    시각은 w_start 기준 분(자정 넘김은 1440+). 식사는 시각이 식사 시간대에 걸릴 때 삽입
    — 단일 장소 체류가 식사 시간대 전체를 삼키는 드문 경우엔 건너뛴다.
    """
    t = w_start
    prev: dict | None = None
    slots: list[dict] = []
    flex: list[dict] = []
    travel_km = 0.0
    meals_done: set[str] = set()

    def _meal_if_due() -> None:
        nonlocal t
        for m_start, m_end, label in _MEAL_WINDOWS:
            if label in meals_done or not (m_start <= t < m_end) or t + _MEAL_MIN > w_end:
                continue
            anchor = (prev or (order[0] if order else {})).get("name", "")
            slots.append({"slot_type": "meal", "label": label, "anchor": anchor,
                          "start": t, "end": t + _MEAL_MIN})
            meals_done.add(label)
            t += _MEAL_MIN

    for stop in order:
        _meal_if_due()
        travel_min, travel_mode = (0, "") if prev is None else _travel(prev, stop)
        arrive = t + travel_min
        op_s, op_e = _op_minutes(stop)

        placed: tuple[int, int] | None = None
        for shift in (0, 1440):  # 코스가 자정을 넘으면 운영시간을 다음 날로도 밀어 본다
            start = max(arrive, op_s + shift)
            latest_end = min(op_e + shift, w_end)
            for dwell in (_dwell(stop), _DWELL_FLOOR):
                if start + dwell <= latest_end:
                    placed = (start, start + dwell)
                    break
            if placed:
                break

        if placed is None:
            flex.append(stop)  # 이 순서에선 시간표에 못 앉힘 — 자유 방문 제안으로
            continue
        if prev is not None:
            travel_km += haversine_km(prev["lat"], prev["lng"], stop["lat"], stop["lng"])
        slots.append({"slot_type": "place", "stop": stop, "start": placed[0], "end": placed[1],
                      "travel_min": travel_min, "travel_mode": travel_mode})
        t = placed[1]
        prev = stop

    _meal_if_due()
    # 아직 안 온 끼니 창이 시간 범위 안에 남아 있으면 마무리로 붙인다 — 장소 일정이
    # 일찍 끝나도(예: 16시) 창이 19시를 포함하면 저녁 추천은 나와야 한다.
    for m_start, m_end, label in _MEAL_WINDOWS:
        if label in meals_done or m_start < t or m_start + _MEAL_MIN > w_end:
            continue
        anchor = (prev or (order[0] if order else {})).get("name", "")
        slots.append({"slot_type": "meal", "label": label, "anchor": anchor,
                      "start": m_start, "end": m_start + _MEAL_MIN})
        meals_done.add(label)
        t = m_start + _MEAL_MIN
    return slots, flex, travel_km


def build_schedule(
    stops: list[dict], w_start_h: int, w_end_h: int,
    anchor: tuple[float, float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """확정 장소들 → (시간 순 슬롯 목록, 시간표에 못 앉힌 장소 목록).

    stops 는 name/lat/lng/category/op_start/op_end 를 가진 dict.
    anchor(위치 칩 좌표)가 있으면 출발점에서 가까운 곳부터 시작하는 순서가 유리해진다.
    """
    w_start, w_end = w_start_h * 60, w_end_h * 60
    usable = [s for s in stops if s.get("lat") is not None and s.get("lng") is not None]
    if not usable:
        return [], list(stops)

    orders = (
        [list(p) for p in permutations(usable)]
        if len(usable) <= _MAX_PERM_STOPS else [usable]
    )
    best: tuple[tuple, list[dict], list[dict]] | None = None
    for order in orders:
        slots, flex, km = _simulate(order, w_start, w_end)
        if anchor and order:
            km += haversine_km(anchor[0], anchor[1], order[0]["lat"], order[0]["lng"])
        n_placed = sum(1 for s in slots if s["slot_type"] == "place")
        last_end = slots[-1]["end"] if slots else w_start
        key = (-n_placed, round(km, 3), last_end)
        if best is None or key < best[0]:
            best = (key, slots, flex)

    dropped = [s for s in stops if s.get("lat") is None or s.get("lng") is None]
    return best[1], best[2] + dropped
