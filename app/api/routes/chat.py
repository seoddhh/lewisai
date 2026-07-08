"""자연어 통합 진입점 — POST /agent/chat.

자연어 메시지 → LangGraph 에이전트(router → 의도별 파이프라인) → 결과.
course 의도면 최적화된 동선을, 그 외엔 해당 기능 결과를 반환한다.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.build import run_agent

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    intent: str
    result: dict[str, Any]
    source: str = "ai"
    distance_km: float | None = None  # course 의도일 때 최적화 동선 총거리


@router.post("/agent/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    state = await run_agent(message=req.message)
    return ChatResponse(
        intent=state.get("intent", "chitchat"),
        result=state.get("result") or {},
        source=state.get("source", "ai"),
        distance_km=state.get("total_km"),
    )
