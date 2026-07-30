"""citydata 문화행사 — 파싱/기간 필터(순수 함수) + get_events 네트워크 경로(가짜 클라이언트)."""
from __future__ import annotations

from types import SimpleNamespace

from app.tools import events
from app.tools.events import _ongoing, _parse_event, _period_end, get_events


def test_period_end_normalizes():
    assert _period_end("2026-07-01~2026-07-31") == "2026-07-31"
    assert _period_end("2026.08.01 ~ 2026.08.15") == "2026-08-15"


def test_ongoing_filter():
    assert _ongoing("2000-01-01~2000-01-02", "2026-07-21") is False   # 지난 행사
    assert _ongoing("2026-01-01~2999-12-31", "2026-07-21") is True    # 진행 중
    assert _ongoing("", "2026-07-21") is True                         # 파싱 불가 → 남긴다


def test_parse_event_maps_fields():
    card = _parse_event({
        "EVENT_NM": "서울빛초롱축제", "EVENT_PERIOD": "2026-07-01~2026-07-31",
        "EVENT_PLACE": "광화문광장", "EVENT_X": "126.9770", "EVENT_Y": "37.5759",
        "URL": "http://e", "PAY_YN": "무료",
    })
    assert card["title"] == "서울빛초롱축제"
    assert card["lat"] == 37.5759 and card["lng"] == 126.9770   # Y=위도, X=경도
    assert card["kind"] == "event" and card["period"] == "2026-07-01~2026-07-31"


async def test_get_events_no_key_returns_empty(monkeypatch):
    monkeypatch.setattr(events, "get_settings", lambda: SimpleNamespace(seoul_api_key=""))
    assert await get_events("경복궁") == []


class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, payload):
        self._p = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        return _FakeResp(self._p)


async def test_get_events_parses_and_filters(monkeypatch):
    payload = {"CITYDATA": {"EVENT_STTS": [
        {"EVENT_NM": "진행행사", "EVENT_PERIOD": "2026-01-01~2999-12-31",
         "EVENT_PLACE": "경복궁", "EVENT_X": "126.977", "EVENT_Y": "37.5759"},
        {"EVENT_NM": "지난행사", "EVENT_PERIOD": "2000-01-01~2000-02-01", "EVENT_PLACE": "경복궁"},
    ]}}
    monkeypatch.setattr(events, "get_settings", lambda: SimpleNamespace(seoul_api_key="k"))
    monkeypatch.setattr(events.httpx, "AsyncClient", lambda **kw: _FakeClient(payload))
    events._CACHE.clear()

    result = await get_events("경복궁")
    assert [e["title"] for e in result] == ["진행행사"], "지난 행사는 걸러진다"
    assert result[0]["kind"] == "event"
