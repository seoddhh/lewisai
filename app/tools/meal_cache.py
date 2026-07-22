"""식당 권역 캐시 — Visit Seoul 식당 api를 권역별로 미리 받아 필요한 스키마만 저장해놓았
이유: Visit Seoul 상세 API 로 조회하면 rate limit(~1.4 req/s) 때문에 최대 수십 초가 걸리기 때문에
레이턴시가 20초~40초 가량 늘어난다
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.core.geo import nearest_chip
from app.tools.visitseoul import VsDetail

logger = logging.getLogger("lewisai.meal_cache")


def meal_card(detail: VsDetail, kind: str) -> dict:
    """식당 스키마"""
    return {
        "title": detail.title,
        "summary": detail.summary or detail.description[:100],
        "address": detail.new_address or detail.address,
        "lat": detail.lat,
        "lng": detail.lng,
        "use_time": detail.use_time,
        "kind": kind,
    }


def _cache_dir() -> Path:
    return Path(get_settings().meal_cache_dir)


@lru_cache(maxsize=None)
def load_area_pool(area: str) -> tuple[dict, ...]:
    """권역 이름 → 캐시된 식당 카드들. 파일이 없으면 빈 튜플.

    lru_cache 로 프로세스당 파일을 한 번만 읽는다. 반환은 읽기 전용으로 다룰 것.
    """
    path = _cache_dir() / f"{area}.json"
    if not path.exists():
        return ()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        logger.warning("식당 캐시 로드 실패 area=%s: %s", area, err)
        return ()
    return tuple(r for r in rows if r.get("lat") is not None and r.get("lng") is not None)


def pool_for_stops(stops: list[dict]) -> list[dict]:
    """스톱들의 최근접 권역 캐시를 합친 식당 풀 (제목 기준 중복 제거).

    각 스톱이 속한 권역만 읽으므로, 코스가 여러 권역에 걸쳐도 필요한 만큼만 로드한다.
    """
    areas = {
        nearest_chip(s["lat"], s["lng"])
        for s in stops
        if s.get("lat") is not None and s.get("lng") is not None
    }
    pool: list[dict] = []
    seen: set[str] = set()
    for area in areas:
        for card in load_area_pool(area):
            if card["title"] not in seen:
                seen.add(card["title"])
                pool.append(card)
    return pool
