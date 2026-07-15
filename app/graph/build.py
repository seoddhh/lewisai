"""LangGraph StateGraph 조립 + 컴파일(싱글턴) — 서울 코스 생성 에이전트.

흐름:
  START → router → parse_intent → (의도 분기)
    course   → retrieve → select_places → enrich → nearby → compose → END
    chitchat → chitchat → END

에이전트의 임무는 코스 생성이다. 코스와 무관한 입력은 chitchat 로 폴백한다.
방문 순서·지도 폴리라인·거리는 서버가 만들지 않는다 (strangemap courseRouting.ts 담당).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.core.llm import extract_text
from app.graph.nodes.common import dispatch_intent, parse_intent_node, router_node
from app.graph.nodes.course import (
    compose_node,
    enrich_node,
    nearby_node,
    retrieve_node,
    select_places_node,
)
from app.graph.nodes.passthrough import chitchat_node
from app.graph.state import AgentState


@lru_cache
def get_agent_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("parse_intent", parse_intent_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("select_places", select_places_node)
    g.add_node("enrich", enrich_node)
    g.add_node("nearby", nearby_node)
    g.add_node("compose", compose_node)
    g.add_node("chitchat", chitchat_node)

    g.add_edge(START, "router")
    g.add_edge("router", "parse_intent")
    g.add_conditional_edges(
        "parse_intent",
        dispatch_intent,
        {"course": "retrieve", "chitchat": "chitchat"},
    )
    # course 파이프라인
    g.add_edge("retrieve", "select_places")
    g.add_edge("select_places", "enrich")
    g.add_edge("enrich", "nearby")
    g.add_edge("nearby", "compose")
    g.add_edge("compose", END)
    # 코스 외 폴백
    g.add_edge("chitchat", END)

    return g.compile()


async def run_agent(
    *, message: str | None = None, intent: str | None = None, req: dict | None = None
) -> dict[str, Any]:
    """그래프 1회 실행. 자연어(message) 또는 어댑터(intent+req) 진입 모두 지원."""
    state: dict[str, Any] = {}
    if message is not None:
        state["message"] = message
    if intent is not None:
        state["intent"] = intent
    if req is not None:
        state["req"] = req
    return await get_agent_graph().ainvoke(state)


# ── 코스 생성 과정 노출 (칩/자연어 → 실제 장소가 어떻게 나왔는지) ────────────

COURSE_STEP_LABELS: dict[str, str] = {
    "retrieve": "후보 장소 검색 (RAG)",
    "select_places": "AI 장소 선정",
    "enrich": "실시간 혼잡도 확인",
    "nearby": "주변 식당·문화행사 조회 (Visit Seoul)",
    "compose": "코스 서사 작성",
}


def course_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """최종 상태 → 챗봇에 보여줄 생성 단계에서의 장소선택 이유
    """
    selected = state.get("selected", [])
    nearby = state.get("nearby", {})
    n_nearby = sum(len(v.get("restaurants", [])) + len(v.get("attractions", []))
                   for v in nearby.values())
    return [
        {"id": "s1", "tool": "retrieve", "label": COURSE_STEP_LABELS["retrieve"], "ok": True,
         "detail": f"후보 {len(state.get('candidates', []))}곳"},
        {"id": "s2", "tool": "select_places", "label": COURSE_STEP_LABELS["select_places"],
         "ok": bool(selected),
         "detail": ", ".join(s["name"] for s in selected),
         "picks": [{"name": s["name"], "reason": s.get("reason", ""),
                    "activities": s.get("activities", [])} for s in selected]},
        {"id": "s3", "tool": "enrich", "label": COURSE_STEP_LABELS["enrich"], "ok": True,
         "detail": ", ".join(f"{k} {v}" for k, v in state.get("congestion", {}).items())
                   or "실시간 측정 지점 없음"},
        {"id": "s4", "tool": "nearby", "label": COURSE_STEP_LABELS["nearby"], "ok": n_nearby > 0,
         "detail": f"주변 정보 {n_nearby}건"},
        {"id": "s5", "tool": "compose", "label": COURSE_STEP_LABELS["compose"],
         "ok": state.get("source") == "ai", "detail": (state.get("result") or {})
             .get("course", {}).get("title", "")},
    ]


def _payload(state: dict[str, Any]) -> dict[str, Any]:
    """그래프 최종 상태 → chat 응답 payload (course | text).

    course: stops[] (name/lat/lng/reason/activities/nearby) + 생성 단계 트레이스.
    text:   코스 외 요청에 대한 chitchat 답변.
    """
    result = state.get("result") or {}
    if state.get("intent") == "course" or "course" in result:
        return {
            "kind": "course",
            "course": result.get("course", {}),
            "source": result.get("source", state.get("source", "ai")),
            "steps": course_steps(state),
        }
    return {
        "kind": "text",
        "text": result.get("reply", ""),
        "source": state.get("source", "ai"),
        "steps": [],
    }


async def run_course(note: str, chips: dict[str, Any]) -> dict[str, Any]:
    """칩(+자연어) → 코스. 라우터를 건너뛰고 course 파이프라인으로 직행(결정적)."""
    return _payload(await run_agent(intent="course", req={"note": note, "chips": chips}))


async def run_chat(message: str) -> dict[str, Any]:
    """자연어 → 코스 또는 잡담. 라우터가 의도를 판별한다."""
    return _payload(await run_agent(message=message))


async def _stream(initial: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """그래프 진행 상황/토큰/최종 payload 이벤트를 순서대로 yield.

    이벤트 종류(event):
      - progress: {"stage": retrieve|select_places|enrich|nearby|compose} 코스 생성 단계 안내
      - token:    {"text": "..."} chitchat 답변 조각(실시간 타이핑). 코스는 JSON 이라 토큰 미노출.
      - final:    {"payload": {...}} 최종 구조화 응답(코스/steps 렌더링용).
    """
    graph = get_agent_graph()
    state: dict[str, Any] = {}

    async for mode, data in graph.astream(initial, stream_mode=["updates", "messages"]):
        if mode == "messages":
            chunk, meta = data
            # chitchat 답변만 실시간 타이핑 (compose 등은 JSON 이라 흘리지 않는다)
            if meta.get("langgraph_node") == "chitchat":
                text = extract_text(getattr(chunk, "content", ""))
                if text:
                    yield {"event": "token", "text": text}
            continue

        # mode == "updates": {node: patch}
        for node, patch in data.items():
            if not isinstance(patch, dict):
                continue
            state.update(patch)
            if node in COURSE_STEP_LABELS:
                yield {"event": "progress", "stage": node,
                       "message": COURSE_STEP_LABELS[node] + " 완료"}

    yield {"event": "final", "payload": _payload(state)}


async def stream_course(note: str, chips: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """칩 → 코스 스트리밍 (라우터 생략, course 직행)."""
    async for evt in _stream({"intent": "course", "req": {"note": note, "chips": chips}}):
        yield evt


async def stream_chat(message: str) -> AsyncIterator[dict[str, Any]]:
    """자연어 → 코스/잡담 스트리밍."""
    async for evt in _stream({"message": message}):
        yield evt
