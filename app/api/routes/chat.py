"""통합 진입점 — POST /agent/chat (+ /agent/chat/stream).

두 가지 진입을 하나의 응답 계약으로 처리한다:
 1. 칩 진입 (chips 있음): 서울로 경로 생성 칩 → 코스 파이프라인 직행(결정적).
 2. 자연어 진입 (chips 없음): 라우터가 코스/잡담을 판별 → 코스 생성 또는 잡담 폴백.

코스 응답에는 steps 로 "어떻게 이 장소들이 나왔는지" 생성 과정을 함께 실어 보낸다
(후보 검색 → AI 선정(이유) → 혼잡도 → 주변 정보 순서를 챗봇이 그대로 보여줄 수 있게).

방문 순서·지도 폴리라인은 응답에 없다 — strangemap 프론트가 좌표로 계산한다.
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
    message: str = ""                     # 사용자 자연어 (칩만 선택했으면 빈 문자열 가능)
    chips: CourseChips | None = None      # 경로 생성 칩 — 있으면 코스 생성으로 직행


class ChatResponse(BaseModel):
    kind: str                          # course | text
    text: str | None = None            # 잡담 폴백 응답
    course: dict | None = None         # stops[]: name/lat/lng/reason/activities/nearby
    steps: list[dict] = []             # 생성 과정 트레이스 (챗봇 노출용)
    source: str = "ai"


@router.post("/agent/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    payload = (
        await run_course(req.message, req.chips.model_dump())
        if req.chips is not None
        else await run_chat(req.message)
    )
    return ChatResponse(**payload)


async def _sse(req: ChatRequest) -> AsyncIterator[bytes]:
    """Server-Sent Events — 진행 상황/토큰/최종 payload 를 실시간 전송."""
    stream = (
        stream_course(req.message, req.chips.model_dump())
        if req.chips is not None
        else stream_chat(req.message)
    )
    try:
        async for evt in stream:
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n".encode("utf-8")
    except Exception as err:  # noqa: BLE001 — 스트림 중단 대신 오류 이벤트로 마무리
        payload = {"event": "error", "message": str(err)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@router.post("/agent/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _sse(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
