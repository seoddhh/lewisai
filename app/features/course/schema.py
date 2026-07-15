"""course(테마코스) 입출력.

장소 선정과 "왜 이 장소인지"(reason/activities)는 AI 가, 방문 순서·지도 폴리라인은
strangemap 프론트(courseRouting.ts)가 담당한다 — 서버는 좌표만 실어 보낸다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── 서울로 경로 생성 칩 ───────────────────────────────────────────────────────
Companion = Literal["혼자", "친구", "커플", "가족"]
AgeGroup = Literal["10-20대", "20-30대", "30-40대", "40-50대", "60대 이상"]
TimeSlot = Literal["오전", "오후", "밤"]
Purpose = Literal["힐링", "놀거리", "데이트", "관광", "문화생활"]
Location = Literal[
    "종로·중구", "강북·성북", "홍대·마포", "용산·이태원", "여의도·영등포",
    "강남·서초", "성수·건대", "잠실·송파", "관악·사당", "상관없음",
]
CongestionPref = Literal["여유", "보통", "상관없음"]


class CourseChips(BaseModel):
    """프론트 칩 선택값. 모두 선택 사항 — 비면 자연어(note)만으로 생성한다."""

    companion: Companion | None = None
    age: AgeGroup | None = None
    time: TimeSlot | None = None
    purpose: Purpose | None = None
    location: Location | None = None
    congestion: CongestionPref | None = None
    place_count: int = Field(4, ge=3, le=5, description="장소 수 칩 (3~5곳)")

    def summary(self) -> str:
        """프롬프트/트레이스에 넣을 한 줄 요약."""
        parts = [
            f"동반: {self.companion}" if self.companion else "",
            f"나이대: {self.age}" if self.age else "",
            f"시간대: {self.time}" if self.time else "",
            f"목적: {self.purpose}" if self.purpose else "",
            f"위치: {self.location}" if self.location else "",
            f"혼잡도 선호: {self.congestion}" if self.congestion else "",
            f"장소 수: {self.place_count}곳",
        ]
        return ", ".join(p for p in parts if p)


class CourseRequest(BaseModel):
    note: str = Field("", description="자유서술 (예: 야경 보면서 데이트)")
    chips: CourseChips = Field(default_factory=CourseChips)
    # 구 계약 호환 — 칩이 없을 때만 사용 (region 은 강북/강남/강서/강동 권역)
    region: str = "상관없음"
    time: str = ""


class NearbyCard(BaseModel):
    """Visit Seoul 주변 정보 카드 (식당 / 문화행사·관광)."""

    title: str
    summary: str = ""
    address: str = ""
    lat: float | None = None
    lng: float | None = None
    dist_km: float | None = None
    period: str = ""       # 행사 기간 (관광지·식당은 빈 문자열)
    use_time: str = ""
    kind: str = ""         # event | attraction | restaurant


class StopNearby(BaseModel):
    restaurants: list[NearbyCard] = []
    attractions: list[NearbyCard] = []


class CourseStop(BaseModel):
    # ── 기존 계약 (프론트 UI 동결 필드) ──
    name: str
    preview: str = ""
    description: str = ""
    duration: str = ""
    tip: str | None = None
    # ── 추가 계약 (좌표는 프론트 라우팅 입력, reason/activities 는 AI 선정 근거) ──
    lat: float | None = None
    lng: float | None = None
    reason: str = ""                # 왜 이 코스에 이 장소를 골랐는지
    activities: list[str] = []      # 이 장소에서 무엇을 할 수 있는지
    congestion: str | None = None   # 실시간 혼잡도 레벨
    nearby: StopNearby = Field(default_factory=StopNearby)


class Course(BaseModel):
    title: str
    subtitle: str = ""
    description: str = ""
    stops: list[CourseStop]
    tags: list[str] = []


class CourseResponse(BaseModel):
    course: Course
    source: str = "ai"
