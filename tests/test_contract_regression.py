"""코스 에이전트 엔드포인트 응답 계약(동결) 회귀 테스트.

LLM 은 실패하는 스텁으로 대체 → 각 체인의 mock 폴백 경로가
프론트 스키마(AGENTS.md 규칙 4)를 그대로 지키는지 확인한다.
retriever 는 픽스처 문서로 대체 (임베딩/네트워크 없음).

대상: /health · /agent/course · /agent/chat(+chips)
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app import config
from app.core import llm as llm_mod
from app.core.llm import get_llm
from app.main import app
from app.rag import retriever


def _boom(*args, **kwargs):
    """항상 실패하는 LLM 호출 — 각 체인의 mock 폴백 경로를 강제한다."""
    raise ValueError("stub llm: no network in tests")


async def _aboom(*args, **kwargs):
    raise ValueError("stub llm: no network in tests")


def _clear_llm_caches() -> None:
    """프로바이더 캐시 전부 비우기 — LLM_PROVIDER 가 바뀌어도 테스트가 안 깨지도록."""
    llm_mod.get_llm.cache_clear()
    llm_mod._solar.cache_clear()
    llm_mod._gemini.cache_clear()


def _doc(name: str, lat: float, lng: float, area: str = "") -> Document:
    return Document(
        page_content=f"[{name}] 서울의 명소 {name} 소개 텍스트",
        metadata={
            "doc_type": "place", "place_id": name, "display_name": name,
            "area_name": area, "category": "명소", "region": "강북",
            "lat": lat, "lng": lng, "op_start": 0, "op_end": 24, "aspect": "summary",
        },
    )


_DOCS = [
    _doc("경복궁", 37.5796, 126.977, "경복궁"),
    _doc("북촌한옥마을", 37.5826, 126.9838, "북촌한옥마을"),
    _doc("광장시장", 37.5701, 126.9996, "광장시장"),
    _doc("남산공원", 37.5512, 126.9882, "남산공원"),
]


@pytest.fixture()
def client(monkeypatch):
    # LLM 강제 실패 → mock 폴백 경로.
    # 챗 모델(ChatUpstage/ChatGoogleGenerativeAI)은 pydantic 모델이라 인스턴스 속성을 못
    # 바꾼다 → 클래스 메서드를 패치한다 (monkeypatch 가 테스트 종료 시 원복).
    # 양쪽 프로바이더 더미 키를 다 넣어, LLM_PROVIDER 가 무엇이든 실제 키 유무와 무관하게
    # 모델이 생성되도록 한다.
    monkeypatch.setenv("UPSTAGE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    config.get_settings.cache_clear()
    _clear_llm_caches()
    llm = get_llm()
    monkeypatch.setattr(type(llm), "invoke", _boom)
    monkeypatch.setattr(type(llm), "ainvoke", _aboom)
    # RAG 픽스처 (임베딩 호출 차단). search_with_score 는 (doc, 의미거리) — 거리 작을수록 유사
    monkeypatch.setattr(
        retriever, "search_with_score",
        lambda q, k=40, filters=None: [(d, 0.1 * i) for i, d in enumerate(_DOCS)],
    )
    # 실시간 API 차단 (키 제거)
    monkeypatch.setenv("SEOUL_API_KEY", "")
    monkeypatch.setenv("VISITSEOUL_API_KEY", "")
    config.get_settings.cache_clear()
    yield TestClient(app)
    config.get_settings.cache_clear()
    _clear_llm_caches()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" if "status" in body else body


def test_course_contract(client):
    res = client.post("/agent/course", json={"note": "역사 산책", "region": "강북", "time": "오후"})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"course", "source"}
    course = body["course"]
    assert set(course) == {"title", "subtitle", "description", "stops", "tags",
                           "scheduled", "days", "day_areas", "day_descriptions"}
    assert course["scheduled"] is False, "시간 범위 없는 요청은 시간표를 만들지 않는다"
    assert len(course["stops"]) >= 2
    stop = course["stops"][0]
    # 기존 프론트 필드는 유지 (동결 계약) + 라우팅/선정근거/스케줄 필드가 추가된다
    assert {"name", "preview", "description", "duration", "tip"} <= set(stop)
    assert {"lat", "lng", "reason", "activities", "nearby"} <= set(stop)
    assert {"start_time", "end_time", "slot_type"} <= set(stop)
    assert stop["lat"] is not None and stop["lng"] is not None, "프론트 폴리라인 계산용 좌표"
    assert set(stop["nearby"]) == {"restaurants", "attractions"}


def test_course_chips_contract(client):
    """칩 진입: 코스 그래프 직행 + 생성 과정 steps 노출."""
    res = client.post(
        "/agent/chat",
        json={
            "message": "조용한 데 위주로",
            "chips": {
                "audience": "local", "companions": ["연인과"], "time": "오후",
                "purposes": ["데이트"], "locations": ["종로·중구"],
                "meals": ["점심"], "congestion": "여유", "pace": "relaxed",
            },
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "course"
    assert body["course"]["stops"], "LLM 실패(mock 폴백)여도 장소는 나와야 한다"
    # 챗봇이 "어떻게 이 장소가 나왔는지" 보여줄 수 있는 트레이스
    tools = [s["tool"] for s in body["steps"]]
    assert tools == ["plan", "retrieve", "select_places", "fit_schedule",
                     "meals", "enrich", "nearby", "compose"]
    assert all("label" in s for s in body["steps"])
    # 시간대 칩(오후) → 시간표 코스: 방문 시각이 계산되어 있어야 한다
    assert body["course"]["scheduled"] is True
    timed = [s for s in body["course"]["stops"] if s["slot_type"] == "place"]
    assert timed and all(s["start_time"] and s["end_time"] for s in timed)
    assert all(isinstance(s["duration_min"], int) for s in timed), \
        "클라이언트가 분 단위를 파싱하지 않도록 정수도 같이 보낸다"


def test_meals_chip_creates_meal_slot(client):
    """고른 끼니만 고정 시각(점심 13시)에 들어간다."""
    res = client.post(
        "/agent/chat",
        json={"message": "", "chips": {
            "audience": "local", "time": "오후", "purposes": ["데이트"],
            "locations": ["종로·중구"], "meals": ["점심"]}},
    )
    meals = [s for s in res.json()["course"]["stops"] if s["slot_type"] == "meal"]
    assert len(meals) == 1
    assert meals[0]["start_time"] == "13:00" and meals[0]["end_time"] == "14:00"


def test_no_meals_chip_means_no_meal_slot(client):
    """끼니 미선택이면 식사 슬롯이 없다 — 예전엔 시간창만 보고 자동 삽입됐다."""
    res = client.post(
        "/agent/chat",
        json={"message": "", "chips": {
            "audience": "local", "time": "오후", "purposes": ["데이트"],
            "locations": ["종로·중구"]}},
    )
    stops = res.json()["course"]["stops"]
    assert stops, "식사가 없어도 장소는 나온다"
    assert not [s for s in stops if s["slot_type"] == "meal"]


def test_course_multiday_tourist_contract(client):
    """여행자 멀티데이: 일자 배정 + 기본 하루 창(9~21) 시간표 + 식사 슬롯 meal_options."""
    res = client.post(
        "/agent/chat",
        json={
            "message": "",
            "chips": {
                "audience": "tourist", "days": 2, "companions": ["친구와"],
                "purposes": ["유명 관광지"], "locations": ["종로·중구"],
                "congestion": "상관없음", "pace": "relaxed",
            },
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["kind"] == "course"
    course = body["course"]
    assert course["days"] == 2
    assert course["scheduled"] is True, "여행자는 시각 선택 없이도 하루 창으로 시간표가 생긴다"
    day_values = {s.get("day") for s in course["stops"]}
    assert day_values <= {1, 2} and 1 in day_values
    # 끼니는 이제 사용자가 고른다 — 안 골랐으면 창이 넓어도 식사 슬롯은 없다
    assert not [s for s in course["stops"] if s["slot_type"] == "meal"]


def test_course_multiday_area_rotation_contract(client):
    """여행자 + 위치 상관없음: 일차별 권역(day_areas)이 응답에 실린다."""
    res = client.post(
        "/agent/chat",
        json={
            "message": "",
            "chips": {
                "audience": "tourist", "days": 2, "purposes": ["핫플레이스"],
                "locations": ["상관없음"], "congestion": "상관없음", "pace": "relaxed",
            },
        },
    )
    assert res.status_code == 200
    course = res.json()["course"]
    assert set(course["day_areas"]) == {"1", "2"}
    assert course["day_areas"]["1"] == "성수·건대", "1일차는 목적(핫플레이스)에 맞는 권역"


def test_course_multi_location_contract(client):
    """여행자 + 동네를 여행일수만큼 여러 개 선택: 그 순서대로 하루씩 배정된다."""
    res = client.post(
        "/agent/chat",
        json={
            "message": "",
            "chips": {
                "audience": "tourist", "days": 3,
                "locations": ["홍대·마포", "강남·서초", "성수·건대"],
                "congestion": "상관없음", "pace": "relaxed",
            },
        },
    )
    assert res.status_code == 200
    course = res.json()["course"]
    assert course["day_areas"] == {"1": "홍대·마포", "2": "강남·서초", "3": "성수·건대"}


def test_chat_nl_returns_course(client):
    """자연어 진입도 코스로 흐른다.

    이 서버는 코스만 만든다 — 의도 분류(router)·잡담(chitchat) 분기는 그래프에서
    제거했고, 그래프 밖 `/agent/chitchat` 라우트도 삭제했다(2026-07-30).
    """
    res = client.post("/agent/chat", json={"message": "연인이랑 종로에서 오후에 갈 만한 곳"})
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"kind", "text", "course", "steps", "source"}
    assert body["kind"] == "course"
    assert body["steps"], "자연어도 생성 과정 트레이스를 낸다"


def test_chat_stream_emits_final(client):
    """스트리밍 진입: SSE 로 최소한 final 이벤트(payload)가 흘러야 한다."""
    with client.stream("POST", "/agent/chat/stream",
                       json={"message": "연인이랑 종로 오후 코스"}) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        text = "".join(res.iter_text())
    assert '"event": "final"' in text and '"kind": "course"' in text


def test_ui_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "서울로" in res.text
