"""course 체인(1차): RAG 검색 → LLM 동선 생성 → Course.

2차(LangGraph)에서 OSRM 경로·의미적 동선 tool-calling 으로 고도화 예정.
"""
from __future__ import annotations

from app.core.json_parse import parse_json_object
from app.core.llm import extract_text, get_llm
from app.rag import retriever

from .prompt import PROMPT
from .schema import Course, CourseRequest, CourseResponse, CourseStop


async def run(req: CourseRequest) -> CourseResponse:
    filters = {} if req.region == "상관없음" else {"region": req.region}
    query = req.note or "서울 코스"
    docs = retriever.search_diverse(query, k=8, filters=filters or None)
    allowed = {d.metadata.get("display_name", "") for d in docs}
    cand_lines = "\n".join(f"- {d.metadata.get('display_name','')}: {d.page_content[:70]}" for d in docs)

    chain = PROMPT | get_llm()
    try:
        msg = await chain.ainvoke(
            {"note": req.note, "region": req.region, "time": req.time or "상관없음", "candidates": cand_lines}
        )
        data = parse_json_object(extract_text(msg.content))
        stops = [
            CourseStop(**s) for s in data.get("stops", []) if s.get("name") in allowed
        ]
        if not stops:
            raise ValueError("화이트리스트 통과 정류장 없음")
        course = Course(
            title=data.get("title", "서울 코스"),
            subtitle=data.get("subtitle", ""),
            description=data.get("description", ""),
            stops=stops,
            tags=data.get("tags", []),
        )
        return CourseResponse(course=course, source="ai")
    except Exception as err:  # noqa: BLE001 — 폴백 보장
        print(f"[course] 실패, mock 폴백: {err}")
        stops = [
            CourseStop(name=d.metadata.get("display_name", ""), preview=d.page_content[:40], duration="1시간")
            for d in docs[:3]
        ]
        course = Course(title="서울 추천 코스", stops=stops, tags=["코스"])
        return CourseResponse(course=course, source="mock")
