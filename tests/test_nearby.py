"""nearby_node — 식당은 권역 캐시에서, 행사는 citydata 에서. 캐시가 얇으면 라이브 폴백.

3분할 구조의 핵심 회귀 방지: Visit Seoul 은 식당만, 행사는 citydata(attractions)로 온다.
"""
from __future__ import annotations

from app.graph.nodes import course

# 경복궁 근처 좌표의 단일 스톱 (area_name 은 citydata 행사 조회 키)
_STOP = {"name": "경복궁", "lat": 37.5796, "lng": 126.9770, "area_name": "경복궁"}


def _restaurant(title: str, lat: float, lng: float) -> dict:
    return {"title": title, "summary": "", "address": "", "lat": lat, "lng": lng,
            "use_time": "", "kind": "restaurant"}


async def test_nearby_uses_cache_and_events(monkeypatch):
    cache = [_restaurant(f"식당{i}", 37.5796 + i * 0.0001, 126.9770) for i in range(10)]

    async def fake_events(area):
        assert area == "경복궁"
        return [{"title": "경복궁 별빛야행", "summary": "야간개장", "address": "", "lat": 37.5796,
                 "lng": 126.9770, "period": "2026-07-01~2999-12-31", "use_time": "", "kind": "event"}]

    monkeypatch.setattr(course, "pool_for_stops", lambda stops: cache)
    monkeypatch.setattr(course, "get_events", fake_events)

    out = await course.nearby_node({"selected": [_STOP], "req": {"chips": {}}})

    rests = out["nearby"]["경복궁"]["restaurants"]
    attrs = out["nearby"]["경복궁"]["attractions"]
    assert rests and all(r["kind"] == "restaurant" and "dist_km" in r for r in rests)
    assert [a["title"] for a in attrs] == ["경복궁 별빛야행"]
    assert attrs[0]["kind"] == "event"
    assert out["meal_pool"] == cache, "meal_pool 은 병합된 식당 풀 그대로"


async def test_nearby_falls_back_to_live_when_cache_thin(monkeypatch):
    live = [_restaurant("라이브식당", 37.5796, 126.9770)]

    async def fake_live(selected, chips):
        return live

    async def fake_events(area):
        return []

    monkeypatch.setattr(course, "pool_for_stops", lambda stops: [])   # 캐시 없음
    monkeypatch.setattr(course, "_live_meal_pool", fake_live)
    monkeypatch.setattr(course, "get_events", fake_events)

    out = await course.nearby_node({"selected": [_STOP], "req": {"chips": {}}})
    assert out["meal_pool"] == live, "캐시가 얇으면 라이브 조회로 보충"
    assert [r["title"] for r in out["nearby"]["경복궁"]["restaurants"]] == ["라이브식당"]
