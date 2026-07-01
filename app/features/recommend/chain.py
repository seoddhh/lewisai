"""recommend 체인: 권역 메타필터 RAG 후보 검색 → 혼잡도 필터 → LLM → 화이트리스트 검증.

화이트리스트: LLM 응답의 place 가 후보 목록에 없으면 폴백(환각 차단).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.json_parse import parse_json_array
from app.core.llm import extract_text, get_llm
from app.rag import retriever
from app.tools.congestion import get_congestion

from .prompt import PROMPT
from .schema import RecommendRequest, RecommendResponse, Suggestion

_KST = timezone(timedelta(hours=9))


def _now_str() -> str:
    n = datetime.now(_KST)
    return f"{n.month}월 {n.day}일 {n.hour}시"


async def _candidates(req: RecommendRequest, k: int = 12) -> list[dict]:
    filters = {} if req.region == "상관없음" else {"region": req.region}
    query = f"{req.purpose} {req.companion} {req.time}".strip() or "서울 가볼 만한 곳"
    docs = retriever.search_diverse(query, k=k, filters=filters or None)

    out = []
    for d in docs:
        m = d.metadata
        cand = {
            "display_name": m.get("display_name", ""),
            "category": m.get("category", ""),
            "area_name": m.get("area_name", ""),
            "text": d.page_content,
        }
        # 혼잡도 선호가 있으면 실시간 조회해 필터
        if req.congestion != "상관없음" and cand["area_name"]:
            lvl = await get_congestion(cand["area_name"])
            level = lvl.split(":")[0].strip() if lvl else "알 수 없음"
            cand["congestion"] = level
            if not _accept(level, req.congestion):
                continue
        out.append(cand)
    return out or [
        {"display_name": d.metadata.get("display_name", ""), "category": d.metadata.get("category", ""), "text": d.page_content}
        for d in docs
    ]


def _accept(level: str, pref: str) -> bool:
    if level == "알 수 없음":
        return True
    if pref == "여유":
        return level == "여유"
    if pref == "보통":
        return level in ("여유", "보통")
    return True


async def run(req: RecommendRequest) -> RecommendResponse:
    cands = await _candidates(req)
    allowed = {c["display_name"] for c in cands}
    cand_lines = "\n".join(
        f"- {c['display_name']} ({c['category']}, {c['text'][:60]})" for c in cands
    )

    chain = PROMPT | get_llm()
    try:
        msg = await chain.ainvoke(
            {
                "now": _now_str(),
                "companion": req.companion or "상관없음",
                "age_group": req.ageGroup or "상관없음",
                "time": req.time or "상관없음",
                "purpose": req.purpose or "상관없음",
                "region": f"서울 {req.region}" if req.region != "상관없음" else "서울 전역",
                "congestion": req.congestion,
                "candidates": cand_lines,
            }
        )
        items = parse_json_array(extract_text(msg.content))
        suggestions = [
            Suggestion(**s) for s in items if s.get("place") in allowed
        ][:3]
        if not suggestions:
            raise ValueError("화이트리스트 통과 결과 없음")
        return RecommendResponse(suggestions=suggestions, source="ai")
    except Exception as err:  # noqa: BLE001 — 폴백 보장
        print(f"[recommend] 실패, mock 폴백: {err}")
        mock = [
            Suggestion(
                title=f"{c['display_name']} 둘러보기",
                place=c["display_name"],
                duration="1~2시간",
                description=c["text"][:80],
                reason="현재 상황에 무난하게 어울리는 코스예요.",
                tags=[c["category"]],
            )
            for c in cands[:3]
        ]
        return RecommendResponse(suggestions=mock, source="mock")
