"""course(테마코스) 입출력. 1차는 기본 RAG 코스 생성. 의미적 동선 고도화는 2차(LangGraph)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CourseRequest(BaseModel):
    note: str = Field("", description="자유서술 (예: 야경 보면서 데이트)")
    region: str = "상관없음"
    time: str = ""


class CourseStop(BaseModel):
    name: str
    preview: str = ""
    description: str = ""
    duration: str = ""
    tip: str | None = None


class Course(BaseModel):
    title: str
    subtitle: str = ""
    description: str = ""
    stops: list[CourseStop]
    tags: list[str] = []


class CourseResponse(BaseModel):
    course: Course
    source: str = "ai"
