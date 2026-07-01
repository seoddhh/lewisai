"""place_intro 체인: RAG 검색 → 실시간(혼잡도·행사) 주입 → LLM → JSON → AIPlaceInfo.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.json_parse import parse_json_object
from app.core.llm import extract_text, get_llm
from app.rag import retriever
from app.tools.congestion import get_congestion
from app.tools.events import get_events

from .prompt import PROMPT
from .schema import AIEvent, AIPlaceInfo, PlaceIntroRequest, PlaceIntroResponse

_KST = timezone(timedelta(hours=9))


def _now_str() -> str:
    n = datetime.now(_KST)
    periods = ["새벽", "오전", "점심", "오후", "저녁", "밤"]
    p = periods[min(n.hour // 4, 5)]
    wd = ["월", "화", "수", "목", "금", "토", "일"][n.weekday()]
    return f"{n.month}월 {n.day}일 {n.hour}시 ({wd}요일 {p})"


async def run(req: PlaceIntroRequest) -> PlaceIntroResponse:
    docs = retriever.search(req.place, k=4, filters=None)
    context = "\n".join(f"- {d.page_content}" for d in docs) or "(검색 결과 없음)"

    congestion = await get_congestion(req.place) or "정보 없음"
    events = await get_events(req.lat, req.lng) if req.lat and req.lng else []
    events_str = (
        "; ".join(f"{e['title']}({e.get('dist_km','?')}km)" for e in events)
        if events
        else "없음"
    )

    chain = PROMPT | get_llm()
    try:
        msg = await chain.ainvoke(
            {
                "place": req.place,
                "context": context,
                "now": _now_str(),
                "congestion": congestion,
                "events": events_str,
            }
        )
        data = parse_json_object(extract_text(msg.content))
        info = AIPlaceInfo(
            placeName=req.place,
            summary=data["summary"],
            highlights=data.get("highlights", []),
            tip=data.get("tip", ""),
            best_time=data.get("best_time", ""),
            crowd_tip=data.get("crowd_tip", ""),
            right_now=congestion if congestion != "정보 없음" else None,
            nearby=[d.metadata.get("display_name", "") for d in docs[1:3]],
            vibe=data.get("vibe", []),
            tags=data.get("tags", []),
            events=[AIEvent(**e) for e in events] or None,
        )
        return PlaceIntroResponse(info=info, source="ai")
    except Exception as err:  # noqa: BLE001 — 어떤 실패든 mock 폴백 보장
        print(f"[place_intro] LLM 실패, mock 폴백: {err}")
        return PlaceIntroResponse(info=_mock(req.place, congestion), source="mock")


def _mock(place: str, congestion: str) -> AIPlaceInfo:
    return AIPlaceInfo(
        placeName=place,
        summary=f"{place}는 서울의 대표적인 명소입니다.",
        highlights=["서울의 정취를 한눈에", "사진 명소로 인기", "산책 코스로 최적"],
        tip="방문 전 운영 시간을 확인해보세요.",
        best_time="맑은 날 저녁이 가장 좋습니다.",
        crowd_tip="평일 방문이 주말보다 여유롭습니다.",
        right_now=congestion if congestion != "정보 없음" else None,
        tags=["서울", "명소"],
    )
