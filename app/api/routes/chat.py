"""챗 API. 프론트가 부르는 창구는 여기 둘뿐이다 —
POST /agent/chat(한 번에 응답)과 POST /agent/chat/stream(진행 상황을 실시간으로).

들어오는 길은 두 가지고, 나가는 응답 모양은 같다:
 1. 칩을 골라서 온 경우(chips 있음) → 바로 코스를 만든다.
 2. 그냥 말로 물어본 경우(chips 없음) → 코스 요청인지 잡담인지 먼저 판별한다.

코스 응답에는 steps 를 같이 실어 보낸다. 장소가 어떻게 뽑혔는지(후보 검색 → 선정 →
주변 정보 → 시간표 → 혼잡도) 챗봇이 사용자에게 그대로 보여줄 수 있게 하려는 것이다.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.features.course.schema import CourseChips
from app.graph.build import run_chat, run_course, stream_chat, stream_course

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = ""                     # 사용자가 쓴 문장. 선택사항이라 비워져 있어도 됨
    chips: CourseChips | None = None      # 위저드에서 고른 칩. 있으면 바로 코스를 만든다
    # "다시 만들기" 버튼용. 같은 칩으로 다른 코스를 받고 싶을 때 매번 다른 값을 보낸다.
    # 안 보내면 칩 내용으로 시드가 정해져서, 같은 요청에는 항상 같은 코스가 나온다.
    seed: int | None = None


class ChatResponse(BaseModel):
    kind: str                          # course(코스) | text(잡담 답변)
    text: str | None = None            # kind=text 일 때의 답변
    course: dict | None = None         # kind=course 일 때의 코스. stops[] 안에 장소·좌표·이유가 들어간다
    steps: list[dict] = []             # 코스가 만들어진 과정. 챗봇이 사용자에게 보여준다
    source: str = "ai"                 # ai | mock — AI 응답인지 임시 데이터인지


@router.post("/agent/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """코스나 답변을 한 번에 만들어 돌려준다."""
    payload = (
        await run_course(req.message, req.chips.model_dump(), seed=req.seed)
        if req.chips is not None
        else await run_chat(req.message)
    )
    return ChatResponse(**payload)


async def _sse(req: ChatRequest) -> AsyncIterator[bytes]:
    """SSE 형식으로 진행 상황과 최종 결과를 흘려보낸다."""
    stream = (
        stream_course(req.message, req.chips.model_dump(), seed=req.seed)
        if req.chips is not None
        else stream_chat(req.message)
    )
    try:
        async for evt in stream:
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode("utf-8")
    except Exception as err:  # noqa: BLE001 — 연결을 끊는 대신 오류 이벤트를 보내고 끝낸다
        payload = {"event": "error", "message": str(err)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@router.post("/agent/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """위와 같은 일을 하되, 다 될 때까지 기다리지 않고 중간 진행 상황부터 보내준다."""
    return StreamingResponse(
        _sse(req),
        media_type="text/event-stream",
        # 중간 서버(프록시)가 응답을 모아뒀다 한 번에 보내면 실시간이 아니게 되므로 꺼둔다
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
