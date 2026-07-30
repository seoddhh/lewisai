"""시간 골격(plan) + 예산 배분 시간표(scheduler) 단위 테스트.

순수 계산이라 LLM/네트워크 없이 검증한다:
 - 칩 → 골격: 식사 시각이 창을 잘라 구간을 만들고, 끼니가 없으면 구간 1개
 - 구간별 장소 배정: 길이 비례 + 수용 한계, 짧은 구간도 굶기지 않음
 - 예산 배분: 구간을 꽉 채우되 하한(30분) 아래로는 안 눌리고 넘치면 flex
 - 이동 추정: 가까운 구간은 도보, 먼 구간은 대중교통
"""
from __future__ import annotations

from app.core.plan import allocate_stops, build_skeleton, capacity, describe
from app.core.scheduler import (
    DWELL_FLOOR,
    build_day,
    duration_label,
    fit_segment,
    hhmm,
    travel,
    wanted_dwell,
)
from app.features.course.schema import CourseChips, TimeWindow


def _stop(name, lat, lng, category="공원·자연", dwell=None):
    s = {"name": name, "lat": lat, "lng": lng, "category": category}
    if dwell is not None:
        s["dwell_min"] = dwell
    return s


# 종로 일대 — 서로 1km 내외
_PALACE = _stop("경복궁", 37.5796, 126.9770, "고궁·역사")
_VILLAGE = _stop("북촌한옥마을", 37.5826, 126.9838)
_MARKET = _stop("광장시장", 37.5701, 126.9996, "상권·역세권")
_PARK = _stop("남산공원", 37.5512, 126.9882)


# ── TimeWindow.overlaps ──────────────────────────────────────────────────────

def test_overlaps_basic():
    assert TimeWindow(start=12, end=18).overlaps(9, 18)
    assert not TimeWindow(start=18, end=23).overlaps(9, 18)


def test_overlaps_always_open():
    assert TimeWindow(start=18, end=23).overlaps(0, 24)


def test_overlaps_overnight_operation():
    assert TimeWindow(start=18, end=23).overlaps(5, 1)
    assert not TimeWindow(start=2, end=4).overlaps(5, 1)


# ── 끼니 칩 → 시간 창 ─────────────────────────────────────────────────────────

def test_meals_extend_window():
    """오후(12~18) 창에 저녁(19시)을 고르면 창이 20시까지 늘어난다.

    "저녁 먹고 싶다" = "저녁까지 있겠다" 이므로, 고른 끼니가 조용히 사라지면 안 된다.
    """
    c = CourseChips(audience="local", time="오후", meals=["저녁"])
    w = c.resolved_window()
    assert (w.start, w.end) == (12, 20)
    assert [m[0] for m in c.meal_slots()] == ["저녁"]


def test_meals_within_window_unchanged():
    c = CourseChips(audience="local", time="오후", meals=["점심"])
    assert (c.resolved_window().start, c.resolved_window().end) == (12, 18)


def test_no_window_means_no_meal_slots():
    """시간 창이 없으면(현지인 시간 미선택) 끼니만으로 창을 만들지 않는다."""
    c = CourseChips(audience="local", meals=["저녁"])
    assert c.resolved_window() is None
    assert c.meal_slots() == []


def test_retired_purpose_is_dropped():
    """'맛집 탐방'은 끼니 칩으로 분리됐다 — 값이 와도 칩 파싱을 깨지 않고 탈락한다."""
    c = CourseChips(purposes=["맛집 탐방", "힐링", "쇼핑"])
    assert c.purposes == ["자연·힐링", "쇼핑"]


def test_meals_are_deduped_and_sorted():
    assert CourseChips(meals=["저녁", "아침", "저녁", "브런치"]).meals == ["아침", "저녁"]


# ── 시간 골격 ────────────────────────────────────────────────────────────────

def test_meals_split_window_into_segments():
    sk = build_skeleton(CourseChips(audience="tourist", meals=["점심", "저녁"]))
    assert [s["minutes"] for s in sk["segments"]] == [240, 300, 60]
    assert [m["label"] for m in sk["meals"]] == ["점심", "저녁"]
    # 식사 직후 구간은 after_meal 로 맥락이 붙는다 (select 프롬프트가 쓴다)
    assert sk["segments"][1]["after_meal"] == "점심"


def test_no_meals_means_single_segment():
    """끼니 미선택도 그래프를 분기하지 않는다 — 구간이 1개가 될 뿐."""
    sk = build_skeleton(CourseChips(audience="tourist"))
    assert len(sk["segments"]) == 1
    assert sk["segments"][0]["minutes"] == 720
    assert sk["meals"] == []


def test_no_window_means_empty_skeleton():
    sk = build_skeleton(CourseChips(audience="local"))
    assert sk["segments"] == [] and sk["window"] is None


# ── 구간별 배정 ──────────────────────────────────────────────────────────────

def test_capacity_limits_short_segments():
    assert capacity(60) == 1        # 60분에 두 곳은 이동 시간조차 없다
    assert capacity(240) == 5


def test_short_segment_is_not_starved():
    """순수 비례배분이면 int(4*60/300)=0 이라 짧은 구간이 통째로 빈다."""
    sk = build_skeleton(CourseChips(audience="local", time="오후", meals=["점심"]))
    quota, flex = allocate_stops(sk, 4)
    assert quota == [1, 3] and flex == 0


def test_allocation_respects_capacity_and_overflow_goes_flex():
    sk = build_skeleton(CourseChips(audience="local", time="오전", meals=["아침"],
                                    pace="relaxed"))
    quota, flex = allocate_stops(sk, 3)
    assert quota == [1, 1], "60분 구간 둘은 각 1곳이 한계"
    assert flex == 1, "한계를 넘은 장소는 버리지 않고 flex 로"


def test_describe_mentions_meal_slots():
    sk = build_skeleton(CourseChips(audience="tourist", meals=["점심"]))
    text = describe(sk, allocate_stops(sk, 4)[0])
    assert "점심 식사" in text and "구간 1" in text


# ── 예산 배분 ────────────────────────────────────────────────────────────────

def test_segment_budget_is_filled_without_idle_tail():
    """예전 그리디 앞채움은 4곳 중 3곳만 앉히고 창 50분을 남겼다."""
    slots, dropped = fit_segment([_PALACE, _VILLAGE, _MARKET, _PARK], 14 * 60, 18 * 60)
    assert len(slots) == 4 and not dropped
    assert slots[-1]["end"] == 18 * 60, "구간 끝까지 채운다"
    for a, b in zip(slots, slots[1:]):
        assert a["end"] <= b["start"], "슬롯이 겹치면 안 된다"


def test_llm_dwell_ratio_is_preserved():
    """LLM 희망 체류시간의 '상대적 길이'가 예산 안에서 유지된다."""
    stops = [_stop("A", 37.5796, 126.9770, dwell=120),
             _stop("B", 37.5800, 126.9775, dwell=40)]
    slots, _ = fit_segment(stops, 14 * 60, 17 * 60)
    a, b = slots[0]["dwell_min"], slots[1]["dwell_min"]
    assert a > b, "희망이 긴 쪽이 더 오래 머문다"


def test_dwell_never_below_floor_overflow_goes_flex():
    slots, dropped = fit_segment([_PALACE, _VILLAGE, _MARKET, _PARK],
                                 14 * 60, 14 * 60 + 90)
    assert all(s["dwell_min"] >= DWELL_FLOOR for s in slots)
    assert dropped, "예산이 모자라면 하한 아래로 누르지 않고 flex 로 뺀다"
    assert len(slots) + len(dropped) == 4, "장소를 잃어버리지 않는다"


def test_dwell_falls_back_to_category_when_llm_omits():
    assert wanted_dwell({"category": "고궁·역사"}) == 90
    assert wanted_dwell({"category": "없는분류"}) == 60
    assert wanted_dwell({"category": "고궁·역사", "dwell_min": 45}) == 45


def test_missing_coords_are_dropped_not_crashed():
    broken = {"name": "좌표없음", "lat": None, "lng": None, "category": ""}
    slots, dropped = fit_segment([_VILLAGE, broken], 14 * 60, 16 * 60)
    assert "좌표없음" in [d["name"] for d in dropped]
    assert len(slots) == 1


# ── 이동 추정 ────────────────────────────────────────────────────────────────

def test_near_leg_walks_far_leg_takes_transit():
    near_min, near_mode = travel(_PALACE, _VILLAGE)
    far_min, far_mode = travel(_PALACE, _stop("코엑스", 37.5115, 127.0595))
    assert near_mode == "walk" and near_min < 20
    assert far_mode == "transit" and far_min < 90, "9km 를 걷는 시간표는 비현실적"


def test_walk_estimate_is_not_inflated():
    """이전 추정(15분/km + 10분 버퍼)은 1km 를 25분으로 잡아 창을 갉아먹었다."""
    minutes, mode = travel(_PALACE, _stop("X", 37.5886, 126.9770))   # ~1.0km
    assert mode == "walk" and minutes <= 22


# ── 하루 조립 ────────────────────────────────────────────────────────────────

def test_build_day_places_stops_across_segments():
    sk = build_skeleton(CourseChips(audience="tourist", meals=["점심"], pace="relaxed"))
    quota, _ = allocate_stops(sk, 3)
    groups, i = [], 0
    all_stops = [_PALACE, _VILLAGE, _MARKET]
    for n in quota:
        groups.append(all_stops[i:i + n])
        i += n
    slots, flex = build_day(groups, sk["segments"])
    assert len(slots) + len(flex) == 3
    for s in slots:                       # 모든 슬롯이 자기 구간 안에 있다
        assert any(seg["start"] <= s["start"] and s["end"] <= seg["end"]
                   for seg in sk["segments"])


# ── 라벨 헬퍼 ────────────────────────────────────────────────────────────────

def test_hhmm_and_duration_label():
    assert hhmm(9 * 60 + 5) == "09:05"
    assert hhmm(25 * 60) == "01:00", "자정 넘김은 다음날 시각으로"
    assert duration_label(90) == "1시간 30분"
    assert duration_label(60) == "1시간"
    assert duration_label(45) == "45분"
