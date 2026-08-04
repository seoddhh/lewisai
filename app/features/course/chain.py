"""코스 그래프를 돌리고, 그 결과를 CourseResponse 모양으로 옮겨 담는다."""
from __future__ import annotations

from app.graph.build import run_agent

from .schema import Course, CourseRequest, CourseResponse, CourseStop


async def run(req: CourseRequest) -> CourseResponse:
    payload: dict = {"note": req.note, "chips": req.chips.model_dump()}
    if req.seed is not None:      # "다시 만들기". 안 주면 요청 내용으로 시드가 정해진다
        payload["seed"] = req.seed
    state = await run_agent(req=payload)
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
            day_descriptions=course_data.get("day_descriptions", {}),
        )
        return CourseResponse(course=course, source=result.get("source", "ai"))

    # 그래프가 코스를 못 만들었을 때 내보낼 빈 응답
    return CourseResponse(
        course=Course(title="서울 추천 코스", stops=[], tags=["코스"]),
        source="mock",
    )
