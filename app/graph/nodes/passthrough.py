"""chitchat 노드 — 코스 요청이 아닌 일반 대화 폴백. 기존 feature 체인을 그대로 감싼다."""
from __future__ import annotations

from app.graph.state import AgentState


async def chitchat_node(state: AgentState) -> dict:
    from app.features.chitchat.chain import ChitchatRequest, run

    req = state.get("req", {})
    message = req.get("message") or state.get("message", "")
    resp = await run(ChitchatRequest(message=message))
    return {"result": resp.model_dump(), "source": "ai"}
