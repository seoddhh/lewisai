"""place_intro 입출력 스키마. 출력은 서울로 프론트 AI PlaceInfo 계약과 동일하게 유지."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlaceIntroRequest(BaseModel):
    place: str = Field(..., description="장소명 (예: 남산공원)")
    lat: float | None = None
    lng: float | None = None
    type: str | None = None  # "culture" 면 행사 소개 톤


class AIEvent(BaseModel):
    title: str
    desc: str = ""
    period: str = ""
    fee: str = "무료"
    dist_km: float | None = None


class AIPlaceInfo(BaseModel):
    placeName: str
    summary: str
    highlights: list[str]
    tip: str
    best_time: str
    crowd_tip: str
    right_now: str | None = None
    viewpoint_guide: str | None = None
    nearby: list[str] = []
    vibe: list[str] = []
    tags: list[str] = []
    events: list[AIEvent] | None = None


class PlaceIntroResponse(BaseModel):
    info: AIPlaceInfo
    source: str = "ai"  # ai | mock
