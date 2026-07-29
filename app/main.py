"""FastAPI 엔트리. 실행: uvicorn app.main:app --reload --port 8800"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, chitchat, course, health
from app.config import get_settings

app = FastAPI(title="서울로 AI (lewisai) — LangGraph RAG 에이전트", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 시 서울로 프론트 도메인으로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_internal_token(request: Request, call_next):
    """BFF(strangemap)만 /agent/* 를 호출하도록 공유 시크릿을 검증한다.

    AI_SERVER_TOKEN 이 설정돼 있으면 X-Internal-Token 헤더가 일치할 때만 통과.
    비어 있으면(로컬 개발) 검증을 건너뛴다 — 프로덕션에서만 토큰을 채운다.
    검증 대상은 /agent/* 뿐이라 /health 프로브·정적 UI 는 토큰 없이 열린다.
    """
    token = get_settings().ai_server_token
    if token and request.url.path.startswith("/agent"):
        if request.headers.get("X-Internal-Token") != token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)

app.include_router(health.router)
app.include_router(course.router)
app.include_router(chitchat.router)
app.include_router(chat.router)

# 검증용 챗봇 UI (정적 페이지) — GET /
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
