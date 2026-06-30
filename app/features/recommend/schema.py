"""recommend(상황추천) 입출력. 출력은 서울로 프론트 Suggestion[] 계약과 동일."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    companion: str = Field("", description="누구랑 (예: 연인, 친구)")
    ageGroup: str = ""
    time: str = ""
    purpose: str = ""
    region: str = "상관없음"      # 강북/강남/강서/강동/상관없음
    congestion: str = "상관없음"  # 여유/보통/상관없음


class Suggestion(BaseModel):
    title: str
    place: str
    duration: str
    description: str
    reason: str
    tags: list[str] = []


class RecommendResponse(BaseModel):
    suggestions: list[Suggestion]
    source: str = "ai"
