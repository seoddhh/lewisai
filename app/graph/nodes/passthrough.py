"""recommend / place_intro / chitchat 노드 — 기존 feature 체인을 그대로 감싼다.

이 3개 기능은 통합 그래프(/agent/chat)에서 재사용되며, 로직 변경 없이 기존 체인을 호출한다.
"""
from __future__ import annotations

from app.graph.state import AgentState


async def recommend_node(state: AgentState) -> dict:
    from app.features.recommend.chain import run
    from app.features.recommend.schema import RecommendRequest

    resp = await run(RecommendRequest(**state.get("req", {})))
    return {"result": resp.model_dump(), "source": resp.source}


async def place_intro_node(state: AgentState) -> dict:
    from app.features.place_intro.chain import run
    from app.features.place_intro.schema import PlaceIntroRequest

    resp = await run(PlaceIntroRequest(**state.get("req", {})))
    return {"result": resp.model_dump(), "source": resp.source}


async def chitchat_node(state: AgentState) -> dict:
    from app.features.chitchat.chain import ChitchatRequest, run

    req = state.get("req", {})
    message = req.get("message") or state.get("message", "")
    resp = await run(ChitchatRequest(message=message))
    return {"result": resp.model_dump(), "source": "ai"}
