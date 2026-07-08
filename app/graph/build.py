"""LangGraph StateGraph 조립 + 컴파일(싱글턴).

흐름:
  START → router → parse_intent → (의도 분기)
    course      → retrieve → select_places → enrich → optimize → compose → END
    recommend   → recommend → END
    place_intro → place_intro → END
    chitchat    → chitchat → END
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.common import dispatch_intent, parse_intent_node, router_node
from app.graph.nodes.course import (
    compose_node,
    enrich_node,
    optimize_node,
    retrieve_node,
    select_places_node,
)
from app.graph.nodes.passthrough import chitchat_node, place_intro_node, recommend_node
from app.graph.state import AgentState


@lru_cache
def get_graph():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("parse_intent", parse_intent_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("select_places", select_places_node)
    g.add_node("enrich", enrich_node)
    g.add_node("optimize", optimize_node)
    g.add_node("compose", compose_node)
    g.add_node("recommend", recommend_node)
    g.add_node("place_intro", place_intro_node)
    g.add_node("chitchat", chitchat_node)

    g.add_edge(START, "router")
    g.add_edge("router", "parse_intent")
    g.add_conditional_edges(
        "parse_intent",
        dispatch_intent,
        {
            "course": "retrieve",
            "recommend": "recommend",
            "place_intro": "place_intro",
            "chitchat": "chitchat",
        },
    )
    # course 파이프라인
    g.add_edge("retrieve", "select_places")
    g.add_edge("select_places", "enrich")
    g.add_edge("enrich", "optimize")
    g.add_edge("optimize", "compose")
    g.add_edge("compose", END)
    # 나머지 기능
    g.add_edge("recommend", END)
    g.add_edge("place_intro", END)
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
    return await get_graph().ainvoke(state)
