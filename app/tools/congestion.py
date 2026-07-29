"""서울 혼잡도 — 현재값 + 시간대별 예보 (citydata_ppltn). RAG 아님, 매 호출 직접 fetch.

코스는 "미래에 방문할 곳"이라 현재 혼잡도가 아니라 **방문 시각의 예보**(FCST_PPLTN)를
써야 의미가 맞는다 (예: 오후 2시에 짠 7시 코스 → 7시 예상 혼잡도). 예보는 현재값과 같은
응답에 실려 온다(12시간, 시(時)단위)라 추가 왕복이 없다. 5분 TTL 캐시.
"""
from __future__ import annotations

import time
from datetime import datetime

import httpx

from app.config import get_settings

_CACHE: dict[str, tuple[dict, float]] = {}
_TTL = 5 * 60  # 5분


async def _fetch(area_name: str) -> dict | None:
    """citydata_ppltn 원본 item(현재값 + FCST_PPLTN 예보) — 5분 캐시. 실패/미등록 시 None."""
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
        # 등록 안 된 지점은 레벨이 비어(정보없음) 온다 — 캐시하지 않고 없는 것으로 취급
        if not item.get("AREA_CONGEST_LVL") or item.get("AREA_CONGEST_LVL") == "정보없음":
            return None
        _CACHE[area_name] = (item, time.time())
        return item
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


async def get_forecast_level(area_name: str, at_hhmm: str | None) -> str | None:
    """방문 시각(at_hhmm, 예: '18:00')의 **예상 혼잡도 레벨**. 실패/미등록 시 None.

    예보(FCST_PPLTN)에서 방문 '시(時)'와 같은 미래 버킷을 고른다. 시각이 없거나,
    예보 공백(근시간)·범위(12h) 밖이면 현재 레벨로 폴백한다.
    """
    item = await _fetch(area_name)
    if not item:
        return None
    current = item.get("AREA_CONGEST_LVL") or None
    if not at_hhmm:
        return current
    try:
        target_hour = int(at_hhmm.split(":")[0])
    except (ValueError, AttributeError):
        return current

    now = datetime.now()
    for f in item.get("FCST_PPLTN") or []:
        lvl = f.get("FCST_CONGEST_LVL")
        try:
            dt = datetime.strptime(f.get("FCST_TIME", ""), "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        # 방문 시각과 같은 시(時)의 미래 예보 = 예상 혼잡도 (가장 이른 것부터 매칭)
        if dt >= now and dt.hour == target_hour and lvl:
            return lvl
    return current  # 예보 공백/범위 밖 → 현재값으로 폴백
