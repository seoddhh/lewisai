"""course 체인(2차): LangGraph 통합 에이전트에 위임.

동선 순서는 LLM 이 아니라 app/core/routing.py(결정론적 오픈-패스 TSP)가 결정한다.
자연어 진입은 /agent/chat, 이 어댑터는 기존 타입 계약(CourseResponse)을 유지한다.
"""
from __future__ import annotations

from app.graph.build import run_agent

from .schema import Course, CourseRequest, CourseResponse, CourseStop


async def run(req: CourseRequest) -> CourseResponse:
    state = await run_agent(
        intent="course",
        req={"note": req.note, "region": req.region, "time": req.time},
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
        )
        return CourseResponse(course=course, source=result.get("source", "ai"))

    # 그래프가 결과를 못 만든 경우 최소 폴백
    return CourseResponse(
        course=Course(title="서울 추천 코스", stops=[], tags=["코스"]),
        source="mock",
    )
