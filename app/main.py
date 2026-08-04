"""FastAPI 엔트리. 실행: uvicorn app.main:app --reload --port 8800"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, course, health
from app.config import get_settings

app = FastAPI(title="서울로 AI (lewisai) — LangGraph RAG 에이전트", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영에 올릴 때는 프론트 도메인만 남길 것
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_internal_token(request: Request, call_next):
    """아무나 이 서버를 부르지 못하게 막는다.

    AI_SERVER_TOKEN 을 설정해 두면, /agent 로 시작하는 요청은 헤더의 토큰이 맞아야 통과한다.
    토큰이 비어 있으면 그냥 통과시킨다(로컬 개발용).
    /health 와 테스트용 화면은 토큰 없이도 열린다.
    """
    token = get_settings().ai_server_token
    if token and request.url.path.startswith("/agent"):
        if request.headers.get("X-Internal-Token") != token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)

app.include_router(health.router)
app.include_router(course.router)
app.include_router(chat.router)

# 개발 중 직접 굴려보는 테스트용 챗봇 화면. 브라우저에서 / 로 들어가면 나온다
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
