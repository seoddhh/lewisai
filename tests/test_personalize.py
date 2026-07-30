"""개인화 — 목적은 검색을, 동반자는 행동(activities)·서사를 움직인다.

장소는 지역구 단위라 태깅 대신 동반×목적 렌즈로 생성을 개인화한다:
 - purpose_rule(): 검색어(query) + 선정 조건(rule)  — vibes 는 데이터가 없어 제거됨
 - companion_rule(): 동반자별 행동 렌즈 (검색엔 미관여)
자연어 합성 칩(_synth_chips)·후보 재랭킹(_geo_rerank)도 여기서 함께 검증한다.
"""
from __future__ import annotations

from app.features.course.schema import CourseChips
from app.graph.nodes.common import _synth_chips
from app.graph.nodes.course import _geo_rerank

_ANCHOR = (37.5720, 126.9860)  # 종로·중구 칩 좌표


def _cand(name: str, lat: float, lng: float) -> dict:
    return {
        "name": name, "lat": lat, "lng": lng, "category": "명소", "area_name": name,
        "op_start": 0, "op_end": 24, "text": name,
    }


# 거리는 사실상 같고(모두 도보권) 의미유사도만 다른 4곳 — 블렌드 순위를 본다.
_POOL = [
    (_cand("고궁정원", 37.5721, 126.9861), 0.10),
    (_cand("트렌디카페거리", 37.5722, 126.9862), 0.11),
    (_cand("전통찻집골목", 37.5721, 126.9862), 0.13),
    (_cand("포토스팟거리", 37.5722, 126.9861), 0.16),
]


def test_geo_rerank_priority():
    ranked = _geo_rerank(_POOL, _ANCHOR, want=4)
    # 유사도+거리 블렌드 — 의미거리 1위(고궁정원)가 상위권 유지, 개수 보존
    assert ranked[0]["name"] == "고궁정원"
    assert len(ranked) == 4


# ── 목적·동반자 렌즈 ─────────────────────────────────────────────────────────

def test_purpose_rule_query_and_rule_only():
    """목적 규칙은 검색어(query)와 선정 조건(rule)만 — vibes 키는 제거되었다."""
    r = CourseChips(purposes=["힐링", "맛집 탐방"]).purpose_rule()
    assert "vibes" not in r
    assert r["query"] and r["rule"]
    assert CourseChips().purpose_rule() == {}


def test_companion_rule_lens():
    """동반자 렌즈 — 동반자별로 다른 행동 렌즈 문장을 주고, 여러 명이면 합친다."""
    solo = CourseChips(companions=["혼자"]).companion_rule()
    friends = CourseChips(companions=["친구와"]).companion_rule()
    assert solo and friends and solo != friends, "동반자마다 렌즈가 달라야 한다"
    both = CourseChips(companions=["혼자", "친구와"]).companion_rule()
    assert solo in both and friends in both, "여러 동반자 선택 시 각 렌즈가 합쳐진다"
    assert CourseChips().companion_rule() == "", "미선택이면 빈 문자열(검색·생성 무영향)"


# ── 자연어 → 합성 칩 ─────────────────────────────────────────────────────────

def test_synth_chips_time_range():
    chips = _synth_chips({"time_start": 14, "time_end": 20})
    assert chips["time_window"] == {"start": 14, "end": 20}


def test_synth_chips_time_word_fallback():
    chips = _synth_chips({"time": "저녁", "time_start": None, "time_end": None})
    assert chips["time_window"] == {"start": 18, "end": 22}


def test_synth_chips_rejects_bad_values():
    chips = _synth_chips({"time_start": 20, "time_end": 3,   # 역순(자정 넘김은 26으로 와야 함)
                          "companion": "반려견"})              # 허용값 밖
    assert chips == {}


def test_synth_chips_companion_purpose():
    """목적은 현행 7칩 어휘만 통과한다 (구 '힐링'은 '자연·힐링'으로 통합됐다)."""
    chips = _synth_chips({"companion": "부모님과", "purpose": "자연·힐링"})
    assert chips == {"companions": ["부모님과"], "purposes": ["자연·힐링"]}
    # 폐기된 목적은 조용히 탈락 — 칩 전체 파싱 실패로 번지지 않는다
    assert _synth_chips({"purpose": "맛집 탐방"}) == {}


def test_synth_chips_meals():
    """자연어에 식사 의도가 있으면 끼니 칩으로 뽑는다 (칩 경로에선 사용자가 직접 고른다)."""
    assert _synth_chips({"meals": ["저녁"]}) == {"meals": ["저녁"]}
    assert _synth_chips({"meals": ["점심", "저녁"]}) == {"meals": ["점심", "저녁"]}
    assert _synth_chips({"meals": ["브런치"]}) == {}, "허용값 밖은 버린다"
    assert _synth_chips({"meals": []}) == {}


def test_synth_chips_audience_days_pace():
    chips = _synth_chips({"audience": "tourist", "days": 3, "pace": "relaxed"})
    assert chips == {"audience": "tourist", "days": 3, "pace": "relaxed"}
    assert _synth_chips({"audience": "화성인", "days": 9, "pace": "빨리"}) == {}


def test_day_areas_rotation():
    """여행자 멀티데이 + 위치 상관없음 → 목적에 맞는 권역으로 시작해 날마다 분산."""
    from app.features.course.schema import CourseChips
    from app.graph.nodes.course import _day_areas

    areas = _day_areas(CourseChips(audience="tourist", days=3, purposes=["핫플레이스"],
                                   locations=["상관없음"]))
    assert areas[1] == "성수·건대", "1일차는 목적에 맞는 권역"
    assert len(set(areas.values())) == 3, "날마다 다른 권역"
    # 동네를 하나만 골랐거나, 하루짜리거나, 로컬이면 분산하지 않는다
    assert _day_areas(CourseChips(audience="tourist", days=3, locations=["홍대·마포"])) is None
    assert _day_areas(CourseChips(audience="tourist", days=1, locations=["상관없음"])) is None
    assert _day_areas(CourseChips(audience="local", days=2, locations=["상관없음"])) is None


def test_day_areas_multi_location_by_day():
    """동네를 여행 일수만큼 여러 개 고르면 그 순서대로 하루씩 배정한다."""
    from app.features.course.schema import CourseChips
    from app.graph.nodes.course import _day_areas

    areas = _day_areas(CourseChips(audience="tourist", days=3,
                                   locations=["홍대·마포", "강남·서초", "성수·건대"]))
    assert areas == {1: "홍대·마포", 2: "강남·서초", 3: "성수·건대"}
