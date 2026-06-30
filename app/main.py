"""FastAPI 엔트리. 실행: uvicorn app.main:app --reload --port 8800"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chitchat, course, health, place_intro, recommend

app = FastAPI(title="서울로 AI (lewisai) — LangChain RAG 에이전트", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 시 서울로 프론트 도메인으로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(place_intro.router)
app.include_router(recommend.router)
app.include_router(course.router)
app.include_router(chitchat.router)
