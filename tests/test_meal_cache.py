"""식당 권역 캐시 — 카드 모양, 디스크 로더(좌표 필터), 권역 병합/중복 제거."""
from __future__ import annotations

import json

from app.tools import meal_cache
from app.tools.meal_cache import load_area_pool, meal_card, pool_for_stops
from app.tools.visitseoul import KIND_RESTAURANT, VsDetail


def test_meal_card_shape():
    d = VsDetail(cid="1", title="근처식당", summary="한식", new_address="종로구 1", lat=37.57, lng=126.98,
                 use_time="11:00~22:00")
    card = meal_card(d, KIND_RESTAURANT)
    assert card == {
        "title": "근처식당", "summary": "한식", "address": "종로구 1",
        "lat": 37.57, "lng": 126.98, "use_time": "11:00~22:00", "kind": KIND_RESTAURANT,
    }


def _write(tmp_path, area: str, rows: list[dict]):
    (tmp_path / f"{area}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def test_load_area_pool_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(meal_cache, "_cache_dir", lambda: tmp_path)
    load_area_pool.cache_clear()
    assert load_area_pool("없는권역") == ()


def test_load_area_pool_filters_coordless(monkeypatch, tmp_path):
    _write(tmp_path, "종로·중구", [
        {"title": "좌표있음", "lat": 37.57, "lng": 126.98, "kind": "restaurant"},
        {"title": "좌표없음", "lat": None, "lng": None, "kind": "restaurant"},
    ])
    monkeypatch.setattr(meal_cache, "_cache_dir", lambda: tmp_path)
    load_area_pool.cache_clear()
    pool = load_area_pool("종로·중구")
    assert [c["title"] for c in pool] == ["좌표있음"]


def test_pool_for_stops_merges_and_dedups(monkeypatch, tmp_path):
    # 종로 좌표 스톱 + 강남 좌표 스톱 → 두 권역 캐시를 합치되 제목 중복은 한 번만
    _write(tmp_path, "종로·중구", [
        {"title": "공용식당", "lat": 37.5720, "lng": 126.9860, "kind": "restaurant"},
        {"title": "종로집", "lat": 37.5721, "lng": 126.9861, "kind": "restaurant"},
    ])
    _write(tmp_path, "강남·서초", [
        {"title": "공용식당", "lat": 37.4979, "lng": 127.0276, "kind": "restaurant"},
        {"title": "강남집", "lat": 37.4980, "lng": 127.0277, "kind": "restaurant"},
    ])
    monkeypatch.setattr(meal_cache, "_cache_dir", lambda: tmp_path)
    load_area_pool.cache_clear()

    stops = [
        {"name": "경복궁", "lat": 37.5796, "lng": 126.9770},   # → 종로·중구
        {"name": "코엑스", "lat": 37.5126, "lng": 127.0590},   # → 강남·서초
    ]
    titles = sorted(c["title"] for c in pool_for_stops(stops))
    assert titles == ["강남집", "공용식당", "종로집"], "두 권역 병합 + 제목 중복 제거"
