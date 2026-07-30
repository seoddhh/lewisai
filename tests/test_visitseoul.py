"""Visit Seoul 어댑터 (Http + Mock) 단위 테스트 — httpx.MockTransport 사용, 네트워크 없음."""
from __future__ import annotations

import json

import httpx
import pytest

from app.tools.visitseoul import (
    CAT_EVENTS,
    KIND_ATTRACTION,
    KIND_BAR,
    KIND_CAFE,
    KIND_EVENT,
    KIND_RESTAURANT,
    HttpVisitSeoulClient,
    MockVisitSeoulClient,
    VisitSeoulError,
    classify,
    get_visitseoul_client,
    search_nearby,
)

_LIST_BODY = {
    "data": [
        {"cid": "EV001", "post_sj": "경복궁 별빛야행", "sumry": "야간 개장", "main_img": "img1", "com_ctgry_sn": CAT_EVENTS},
        {"cid": "EV002", "post_sj": "어린이축제", "sumry": "가족 축제", "main_img": "img2", "com_ctgry_sn": CAT_EVENTS},
    ],
    "paging": {"page_no": 1, "page_size": 20, "total_count": 2},
    "result_code": 200,
    "result_message": "OK",
}

_DETAIL_BODY = {
    "cid": "EV001",
    "post_sj": "경복궁 별빛야행",
    "sumry": "야간 개장",
    "post_desc": "<p>경복궁   <b>야간</b> 관람</p>",
    "main_img": "img1",
    "traffic": {
        "adres": "서울 종로구 사직로 161",
        "new_adres": "서울특별시 종로구 사직로 161",
        "map_position_x": "126.977",
        "map_position_y": "37.5796",
        "subway_info": "3호선 경복궁역",
    },
    "schdul_info_bgnde": "2026-05-01",
    "schdul_info_endde": "2026-10-31",
    "extra": {"cmmn_use_time": "19:00~21:30", "closed_days": "", "cmmn_telno": "02-1522-2295"},
    "tag": ["전통", "야간"],
    "result_code": 200,
    "result_message": "OK",
}


def _client(handler) -> HttpVisitSeoulClient:
    return HttpVisitSeoulClient(
        "test-key", "https://api-call.visitseoul.net",
        transport=httpx.MockTransport(handler),
    )


async def test_list_contents_sends_key_header_and_normalizes():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("VISITSEOUL-API-KEY")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_LIST_BODY)

    items = await _client(handler).list_contents(category=CAT_EVENTS, keyword="종로")
    assert seen["path"] == "/api/v1/contents/list"
    assert seen["key"] == "test-key"
    assert seen["body"]["com_ctgry_sn"] == CAT_EVENTS
    assert seen["body"]["keyword"] == "종로"
    assert [i.cid for i in items] == ["EV001", "EV002"]
    assert items[0].title == "경복궁 별빛야행"


async def test_get_content_normalizes_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/contents/info"
        assert request.method == "POST"  # GET 은 405
        assert json.loads(request.content)["cid"] == "EV001"
        return httpx.Response(200, json={"data": _DETAIL_BODY})

    d = await _client(handler).get_content("EV001")
    assert d is not None
    assert d.lat == 37.5796 and d.lng == 126.977  # map_position_y → lat
    assert d.address == "서울 종로구 사직로 161"
    assert d.begin_date == "2026-05-01" and d.end_date == "2026-10-31"
    assert "야간" in d.description and "<" not in d.description  # HTML 제거
    assert d.tel == "02-1522-2295"


async def test_list_retries_429_then_raises():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"result_message": "Too Many Requests"})

    with pytest.raises(VisitSeoulError):
        await _client(handler).list_contents(category=CAT_EVENTS)
    assert calls["n"] == 2  # 1회 재시도


async def test_list_cache_hit_skips_network():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_LIST_BODY)

    c = _client(handler)
    await c.list_contents(category=CAT_EVENTS)
    await c.list_contents(category=CAT_EVENTS)
    assert calls["n"] == 1


def test_missing_key_selects_mock(monkeypatch):
    from app import config

    monkeypatch.setenv("VISITSEOUL_API_KEY", "")
    config.get_settings.cache_clear()
    get_visitseoul_client.cache_clear()
    try:
        assert isinstance(get_visitseoul_client(), MockVisitSeoulClient)
    finally:
        config.get_settings.cache_clear()
        get_visitseoul_client.cache_clear()


async def test_mock_client_filters_category_and_keyword():
    c = MockVisitSeoulClient()
    events = await c.list_contents(category=CAT_EVENTS, keyword="종로")
    assert events and all(e.category == CAT_EVENTS for e in events)
    d = await c.get_content(events[0].cid)
    assert d is not None and d.lat is not None


# 경복궁 (종로) — 주변 검색 앵커
_GBG = (37.5796, 126.977)


def test_classify_uses_cate_depth_not_category_code():
    # 응답의 com_ctgry_sn 은 하위 코드라 신뢰할 수 없다 → cate_depth 로 판정한다
    assert classify(" 축제/공연/행사 > 축제") == KIND_EVENT
    assert classify(" 역사관광 > 역사유적지 > 고궁") == KIND_ATTRACTION
    assert classify(" 쇼핑 > 전문매장/상가") == "", "쇼핑·숙박은 주변 정보 대상이 아니다"


def test_classify_splits_food_subcategories():
    # "음식"을 하나로 뭉치면 카페·주점이 점심 식당으로 추천된다 → 2번째 마디로 가른다
    assert classify(" 음식 > 한식") == KIND_RESTAURANT
    assert classify(" 음식 > 외국식 > 서양식") == KIND_RESTAURANT
    assert classify(" 음식 > 외국식") == KIND_RESTAURANT
    assert classify(" 음식 > 카페/찻집") == KIND_CAFE
    assert classify(" 음식 > 주점") == KIND_BAR
    assert classify(" 음식") == KIND_RESTAURANT, "하위분류 없는 음식은 일반 식당으로 본다"


async def test_search_nearby_filters_by_kind_and_radius():
    c = MockVisitSeoulClient()
    near = await search_nearby(
        lat=_GBG[0], lng=_GBG[1], keywords=("경복궁", "종로"), kinds=(KIND_RESTAURANT,),
        radius_km=1.5, region_terms=("종로",), budget=20, limit=10, client=c,
    )
    assert near, "경복궁 반경 1.5km 안에 식당이 있어야 한다"
    assert all(it.kind == KIND_RESTAURANT for it in near), "요청한 종류만"
    assert all(it.dist_km is None or it.dist_km <= 1.5 for it in near)
    assert near == sorted(near, key=lambda it: it.dist_km or float("inf")), "가까운 순 정렬"

    tight = await search_nearby(
        lat=_GBG[0], lng=_GBG[1], keywords=("경복궁", "종로"), kinds=(KIND_RESTAURANT,),
        radius_km=0.2, region_terms=("종로",), budget=20, limit=10, client=c,
    )
    assert len(tight) <= len(near), "반경을 좁히면 결과가 줄어든다"


async def test_search_nearby_returns_events_and_attractions():
    c = MockVisitSeoulClient()
    items = await search_nearby(
        lat=_GBG[0], lng=_GBG[1], keywords=("경복궁", "종로"),
        kinds=(KIND_EVENT, KIND_ATTRACTION),
        radius_km=5.0, region_terms=("종로",), budget=20, limit=10, client=c,
    )
    assert items
    assert {it.kind for it in items} <= {KIND_EVENT, KIND_ATTRACTION}
    assert any(it.detail.category == CAT_EVENTS for it in items), "축제·공연 포함"
