"""course 파이프라인 노드 — RAG 후보 → 장소 선택 → 혼잡도 → 동선 최적화 → 서사.

역할 분리: select_places(의미) ↔ optimize(기하, 결정론적 TSP).
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.core.json_parse import parse_json_array, parse_json_object
from app.core.llm import extract_text, get_llm
from app.core.routing import plan_course
from app.graph.state import AgentState
from app.rag import retriever
from app.tools.congestion import get_congestion

# 시간대 → 코스 시작 시각 — strangemap aiCourseDraft TIME_TO_START_HOUR
_TIME_TO_START_HOUR: dict[str, float] = {"오전": 10, "오후": 14, "저녁": 19, "밤": 19}

# 코스 정책 (strangemap 동일)
_MAX_STOPS = 5
_MIN_STOPS = 3
_MAX_TOTAL_KM = 10.0
_CROWDED = "붐빔"

_SELECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 서울 코스 큐레이터다. 사용자 요청에 맞는 장소를 후보 목록에서만 골라 "
            'JSON 배열로만 답하라: ["장소명", ...]\n'
            "- 3~5곳. 후보에 없는 이름은 절대 넣지 말 것.\n"
            "- 요청 주제(분위기·목적)에 어울리는 곳 우선.",
        ),
        ("human", '요청: "{note}" (지역:{region}, 시간대:{time})\n\n[후보]\n{candidates}'),
    ]
)

_COMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 서울 테마 코스 큐레이터다. 아래 '확정된 방문 순서'는 이미 동선 최적화로 정해졌으니 "
            "순서를 바꾸지 말고, 각 단계의 정보만 작성해 JSON 객체로만 답하라.\n"
            'form: {{"title":"...","subtitle":"...","description":"...",'
            '"tags":["..."],"stops":[{{"name":"확정 장소명","preview":"한 줄","description":"현장 감성",'
            '"duration":"예: 1시간"}}]}}\n'
            "- stops 의 name 과 순서는 확정 순서와 정확히 일치시킬 것.",
        ),
        ("human", '요청: "{note}"\n\n[확정된 방문 순서]\n{ordered}'),
    ]
)


def _cand(doc) -> dict:
    m = doc.metadata
    return {
        "name": m.get("display_name", ""),
        "lat": m.get("lat"),
        "lng": m.get("lng"),
        "category": m.get("category", ""),
        "area_name": m.get("area_name", ""),
        "operating_hours": {"start": int(m.get("op_start", 0)), "end": int(m.get("op_end", 24))},
        "text": doc.page_content,
    }


async def retrieve_node(state: AgentState) -> dict:
    req = state.get("req", {})
    region = req.get("region", "상관없음")
    filters: dict = {"doc_type": "place"}
    if region and region != "상관없음":
        filters["region"] = region
    query = req.get("note") or "서울 코스"
    docs = retriever.search_diverse(query, k=8, filters=filters)
    cands = [c for c in (_cand(d) for d in docs) if c["lat"] is not None and c["lng"] is not None]
    return {"candidates": cands}


async def select_places_node(state: AgentState) -> dict:
    req = state.get("req", {})
    cands = state.get("candidates", [])
    by_name = {c["name"]: c for c in cands}
    cand_lines = "\n".join(f"- {c['name']} ({c['category']}): {c['text'][:60]}" for c in cands)

    picked: list[str] = []
    try:
        msg = await (_SELECT_PROMPT | get_llm()).ainvoke(
            {
                "note": req.get("note", ""),
                "region": req.get("region", "상관없음"),
                "time": req.get("time", "상관없음"),
                "candidates": cand_lines,
            }
        )
        names = parse_json_array(extract_text(msg.content))
        # 화이트리스트 강제 — 후보에 있는 이름만, 중복 제거
        for n in names:
            if n in by_name and n not in picked:
                picked.append(n)
    except Exception as err:  # noqa: BLE001
        print(f"[course.select] LLM 실패, 후보 상위 사용: {err}")

    selected = [by_name[n] for n in picked[:_MAX_STOPS + 1]]
    if len(selected) < 2:  # 폴백: 후보 상위
        selected = cands[: _MAX_STOPS]
        return {"selected": selected, "source": "mock"}
    return {"selected": selected, "source": "ai"}


def _level(msg: str | None) -> str | None:
    return msg.split(":")[0].strip() if msg else None


async def enrich_node(state: AgentState) -> dict:
    """혼잡도 우선: 선택 장소의 실시간 혼잡도 조회. '붐빔'이면 미사용 후보로 대체 시도."""
    selected = state.get("selected", [])
    chosen = {s["name"] for s in selected}
    leftovers = [c for c in state.get("candidates", []) if c["name"] not in chosen]

    congestion: dict[str, str] = {}
    final: list[dict] = []
    for s in selected:
        lvl = _level(await get_congestion(s["area_name"])) if s.get("area_name") else None
        if lvl:
            congestion[s["name"]] = lvl
        if lvl == _CROWDED and leftovers:
            repl = leftovers.pop(0)
            rlvl = _level(await get_congestion(repl["area_name"])) if repl.get("area_name") else None
            if rlvl:
                congestion[repl["name"]] = rlvl
            final.append(repl)  # 붐비는 곳을 덜 붐비는 후보로 교체
        else:
            final.append(s)
    return {"selected": final, "congestion": congestion}


async def optimize_node(state: AgentState) -> dict:
    """결정론적 오픈-패스 TSP 로 방문 순서 확정 (LLM 아님)."""
    selected = state.get("selected", [])
    if len(selected) < 2:
        return {"ordered": selected, "total_km": 0.0}
    req = state.get("req", {})
    start_hour = _TIME_TO_START_HOUR.get(req.get("time", ""))
    res = plan_course(
        selected,
        max_stops=_MAX_STOPS,
        min_stops=min(_MIN_STOPS, len(selected)),
        max_total_km=_MAX_TOTAL_KM,
        start_hour=start_hour,
    )
    return {"ordered": res.ordered, "total_km": round(res.total_km, 1)}


async def compose_node(state: AgentState) -> dict:
    """확정된 순서에 LLM 서사를 입힌다. 순서·장소는 고정."""
    ordered = state.get("ordered", [])
    req = state.get("req", {})
    congestion = state.get("congestion", {})
    ordered_lines = "\n".join(f"{i+1}. {s['name']} ({s.get('category','')})" for i, s in enumerate(ordered))

    narrative: dict[str, dict] = {}
    title = subtitle = description = ""
    tags: list[str] = []
    try:
        msg = await (_COMPOSE_PROMPT | get_llm()).ainvoke(
            {"note": req.get("note", "서울 코스"), "ordered": ordered_lines}
        )
        data = parse_json_object(extract_text(msg.content))
        title = data.get("title", "")
        subtitle = data.get("subtitle", "")
        description = data.get("description", "")
        tags = data.get("tags", []) or []
        for s in data.get("stops", []):
            if s.get("name"):
                narrative[s["name"]] = s
        source = state.get("source", "ai")
    except Exception as err:  # noqa: BLE001
        print(f"[course.compose] LLM 실패, mock 서사: {err}")
        source = "mock"

    stops = []
    for s in ordered:
        n = narrative.get(s["name"], {})
        tip = n.get("tip")
        lvl = congestion.get(s["name"])
        if lvl and lvl != "여유":
            note = f"현재 혼잡도 {lvl}"
            tip = f"{tip} · {note}" if tip else note
        stops.append(
            {
                "name": s["name"],
                "preview": n.get("preview", s.get("text", "")[:40]),
                "description": n.get("description", ""),
                "duration": n.get("duration", "1시간"),
                "tip": tip,
            }
        )

    course = {
        "title": title or "서울 추천 코스",
        "subtitle": subtitle,
        "description": description,
        "stops": stops,
        "tags": tags or ["코스"],
    }
    return {
        "result": {"course": course, "source": source},
        "source": source,
    }
