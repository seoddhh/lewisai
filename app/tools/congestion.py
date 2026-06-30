"""서울 실시간 혼잡도 (citydata_ppltn). RAG 아님 — 매 호출 직접 fetch 후 프롬프트에 주입.

strangemap fetchCongestion* 의 파이썬 이식. 5분 TTL 캐시.
"""
from __future__ import annotations

import time

import httpx

from app.config import get_settings

_CACHE: dict[str, tuple[str, float]] = {}
_TTL = 5 * 60  # 5분


async def get_congestion(area_name: str) -> str | None:
    """area_name(예: '남산공원') → '여유: ...' 형태 문자열. 실패 시 None."""
    s = get_settings()
    if not s.seoul_api_key or not area_name:
        return None

    cached = _CACHE.get(area_name)
    if cached and time.time() - cached[1] < _TTL:
        return cached[0]

    url = (
        f"http://openapi.seoul.go.kr:8088/{s.seoul_api_key}"
        f"/json/citydata_ppltn/1/5/{area_name}"
    )
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(url)
        if res.status_code != 200:
            return None
        item = res.json().get("SeoulRtd.citydata_ppltn", [{}])[0]
        msg = f"{item.get('AREA_CONGEST_LVL','정보없음')}: {item.get('AREA_CONGEST_MSG','')}"
        _CACHE[area_name] = (msg, time.time())
        return msg
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None
