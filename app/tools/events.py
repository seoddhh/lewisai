"""서울 실시간 문화행사 (culturalEventInfo). RAG 아님 — 직접 fetch 후 주입.

strangemap fetchRealEvents 의 파이썬 이식(간이판). lat/lng 반경 필터.
"""
from __future__ import annotations

import math

import httpx

from app.config import get_settings

_RADIUS_KM = 3.0


def _dist_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def get_events(lat: float | None = None, lng: float | None = None, limit: int = 3) -> list[dict]:
    s = get_settings()
    if not s.seoul_api_key:
        return []

    url = f"http://openapi.seoul.go.kr:8088/{s.seoul_api_key}/json/culturalEventInfo/1/500/"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(url)
        if res.status_code != 200:
            return []
        rows = res.json().get("culturalEventInfo", {}).get("row", [])
    except (httpx.HTTPError, ValueError):
        return []

    out = []
    for r in rows:
        try:
            r_lat, r_lng = float(r.get("LAT", 0)), float(r.get("LOT", 0))
        except ValueError:
            continue
        if r_lat == 0:
            continue
        dist = _dist_km(lat, lng, r_lat, r_lng) if (lat and lng) else math.inf
        if dist > _RADIUS_KM:
            continue
        out.append(
            {
                "title": r.get("TITLE", ""),
                "desc": r.get("PROGRAM") or r.get("ETC_DESC") or r.get("ORG_NAME") or "",
                "period": r.get("DATE", ""),
                "fee": r.get("USE_FEE") or "무료",
                "dist_km": round(dist, 1) if dist != math.inf else None,
            }
        )
    out.sort(key=lambda e: e["dist_km"] if e["dist_km"] is not None else math.inf)
    return out[:limit]
