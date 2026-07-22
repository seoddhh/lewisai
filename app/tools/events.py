"""서울 실시간 도시데이터(citydata)의 문화행사 — 스톱이 속한 명소의 '지금 하는' 행사.

왜 Visit Seoul 이 아니라 여기인가: Visit Seoul 축제공연행사 목록에는 몇 해 전 행사도
그대로 남아 있어 지난 행사를 걸러내야 한다. citydata 는 서울 주요 명소(~122곳)의
**실시간** 문화행사(EVENT_STTS)를 주므로, 지금 진행 중인 행사만 자연스럽게 나온다.
혼잡도(congestion.py)와 같은 API라 별도 인증이 필요 없다 (seoul_api_key 공유).
"""
from __future__ import annotations

import time
from datetime import date

import httpx

from app.config import get_settings

_CACHE: dict[str, tuple[list[dict], float]] = {}
_TTL = 5 * 60  # 5분


def _to_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if f != 0 else None
    except (TypeError, ValueError):
        return None


def _period_end(period: str) -> str:
    """"2026-07-01~2026-07-31" → "2026-07-31" (끝 날짜만, 하이픈 정규화)."""
    tail = (period or "").split("~")[-1].strip()
    return tail.replace(".", "-").replace("/", "-")[:10]


def _ongoing(period: str, today: str) -> bool:
    """행사 기간의 끝이 오늘 이후면 진행/예정. 파싱 불가면 남긴다(과도한 필터 방지)."""
    end = _period_end(period)
    return not (len(end) == 10 and end < today)


def _parse_event(row: dict) -> dict:
    """citydata EVENT_STTS 항목 → attractions 카드 (nearby 의 관광 카드와 같은 모양)."""
    period = row.get("EVENT_PERIOD", "") or ""
    place = row.get("EVENT_PLACE", "") or ""
    return {
        "title": row.get("EVENT_NM", "") or "",
        "summary": place,
        "address": place,
        "lat": _to_float(row.get("EVENT_Y")),   # EVENT_Y = 위도
        "lng": _to_float(row.get("EVENT_X")),   # EVENT_X = 경도
        "period": period,
        "use_time": "",
        "url": row.get("URL", "") or "",
        "kind": "event",
    }


async def get_events(area_name: str) -> list[dict]:
    """area_name(명소 이름) → 지금 진행 중인 문화행사 카드들. 없거나 실패 시 []."""
    s = get_settings()
    if not s.seoul_api_key or not area_name:
        return []

    cached = _CACHE.get(area_name)
    if cached and time.time() - cached[1] < _TTL:
        return cached[0]

    url = (
        f"http://openapi.seoul.go.kr:8088/{s.seoul_api_key}"
        f"/json/citydata/1/5/{area_name}"
    )
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(url)
        if res.status_code != 200:
            return []
        body = res.json().get("CITYDATA") or {}
        rows = body.get("EVENT_STTS") or []
        if isinstance(rows, dict):  # 단건이면 dict 로 오기도 한다
            rows = [rows]
        today = date.today().isoformat()
        events = [
            card for row in rows
            if (card := _parse_event(row))["title"] and _ongoing(card["period"], today)
        ]
        _CACHE[area_name] = (events, time.time())
        return events
    except (httpx.HTTPError, KeyError, ValueError):
        return []
