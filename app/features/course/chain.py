"""장소 선정·선정 이유·방문 순서는 AI 가 확정한다 — strangemap 프론트는 그 순서 위에서
실제 도로 경로(폴리라인)만 그린다.
"""
from __future__ import annotations

from app.graph.build import run_agent

from .schema import Course, CourseRequest, CourseResponse, CourseStop


async def run(req: CourseRequest) -> CourseResponse:
    state = await run_agent(
        intent="course",
        req={"note": req.note, "chips": req.chips.model_dump()},
    )
    result = state.get("result") or {}
    course_data = result.get("course")
    if course_data and course_data.get("stops"):
        stops = [CourseStop(**s) for s in course_data["stops"]]
        course = Course(
            title=course_data.get("title", "서울 코스"),
            subtitle=course_data.get("subtitle", ""),
            description=course_data.get("description", ""),
            stops=stops,
            tags=course_data.get("tags", []),
            scheduled=course_data.get("scheduled", False),
            days=course_data.get("days", 1),
            day_areas=course_data.get("day_areas", {}),
        )
        return CourseResponse(course=course, source=result.get("source", "ai"))

    # 그래프가 결과를 못 만든 경우 최소 폴백
    return CourseResponse(
        course=Course(title="서울 추천 코스", stops=[], tags=["코스"]),
        source="mock",
    )
