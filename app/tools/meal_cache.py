"""동네별로 미리 받아둔 식당 목록을 읽는다.

코스를 만들 때마다 Visit Seoul 을 부르면, 호출 제한 때문에 20~40초가 더 걸린다.
그래서 미리 받아 파일로 구워두고 실행 중에는 그 파일만 읽는다.
캐시를 굽는 건 scripts/build_meal_cache.py 가 한다.
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
    """Visit Seoul 응답에서 필요한 값만 뽑아 식당 카드 하나로 만든다."""
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
    """그 동네의 캐시 파일을 읽는다. 파일이 없으면 빈 값을 준다.

    한 번 읽으면 계속 재사용하므로, 돌려받은 값을 고치면 안 된다.
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
    """장소들이 속한 동네의 캐시를 합쳐 식당 목록을 만든다. 이름이 겹치면 하나만 남긴다.

    코스가 여러 동네에 걸쳐 있어도 실제로 쓰이는 동네만 읽는다.
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
