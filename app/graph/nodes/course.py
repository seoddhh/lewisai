"""course 파이프라인 — RAG 후보 → AI 장소 선정(+선정 이유) → 혼잡도 → 주변 정보 → 서사.

역할 분리:
 - AI: "어떤 장소를, 왜 골랐는지"와 "거기서 뭘 할 수 있는지(행동추천)".
   지금의 서울(오늘 날짜·시간대·실시간 혼잡도)에 맞춰 쓴다.
 - Visit Seoul: 확정된 장소 **주변**의 식당 / 문화행사·관광정보.
 - strangemap 프론트(courseRouting.ts): 방문 순서·지도 폴리라인·거리.
   → 폴리라인 경로는 프론트에서 작업 Ai 서빙은 해당 장소의 좌표값만 반환
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from app.core.geo import address_terms, chip_of, haversine_km
from app.core.json_parse import parse_json_object
from app.core.llm import extract_text, get_llm
from app.features.course.schema import CourseChips
from app.graph.state import AgentState
from app.rag import retriever
from app.tools.congestion import get_congestion
from app.tools.visitseoul import (
    KIND_ATTRACTION,
    KIND_EVENT,
    KIND_RESTAURANT,
    NearbyItem,
    VisitSeoulError,
    place_keyword,
    search_nearby,
)

logger = logging.getLogger("lewisai.course")

_CROWDED = "붐빔"
_NEARBY_RADIUS_KM = 1.5   # 스톱 하나에 붙일 주변 정보 반경
_NEARBY_PER_STOP = 2      # 스톱당 식당/관광 각 N개
_NEARBY_BUDGET = 14       # 코스 전체 상세 조회 예산 (Visit Seoul rate limit ~1.4 req/s)

# ── 지리 클러스터링 (코스는 지리적 근접성 우선) ──────────────────────────
_FETCH_K = 40                    # 벡터스토어에서 넉넉히 당겨오는 후보 수
_RADIUS_STEPS = (5.0, 7.0, 10.0)  # 앵커 기준 반경 하드필터 — 부족하면 순차 완화
_GEO_ALPHA = 0.3                 # 최종 점수 = α·의미유사도 + (1−α)·근접도 (코스는 근접 우선)

_SELECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 '서울로' 코스 큐레이터다. 후보 목록에서만 장소를 골라 JSON 으로만 답하라.\n"
            'form: {{"stops":[{{"name":"후보에 있는 장소명","reason":"이 사람에게 왜 이 장소인지 한 문장",'
            '"activities":["거기서 할 수 있는 일 2~3개"]}}]}}\n'
            "- 정확히 {count}곳. 후보에 없는 이름은 절대 넣지 말 것.\n"
            "- reason 은 선택 조건(동반·나이대·목적·시간대)과 오늘의 서울(날짜/계절/혼잡도)에 근거해 쓴다.\n"
            "- activities 는 그 장소에서 실제로 할 수 있는 구체적 행동 (예: 야경 사진, 한복 대여).\n"
            "- 자연스러운 방문 흐름 순서로 나열 (최종 동선 순서는 앱이 다시 계산한다).",
        ),
        (
            "human",
            '오늘: {today}\n요청: "{note}"\n선택 조건: {chips}\n\n[후보]\n{candidates}',
        ),
    ]
)

_COMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 '서울로' 코스 큐레이터다. 아래 확정된 장소들에 코스 서사를 입혀 JSON 으로만 답하라.\n"
            'form: {{"title":"...","subtitle":"...","description":"...","tags":["..."],'
            '"stops":[{{"name":"확정 장소명","preview":"한 줄","description":"현장 감성","duration":"예: 1시간"}}]}}\n'
            "- stops 의 name 과 개수는 확정 목록과 정확히 일치시킬 것 (순서 변경·추가 금지).\n"
            "- 실시간 혼잡도가 주어진 장소는 그 상황을 description 에 자연스럽게 반영.",
        ),
        ("human", '요청: "{note}"\n선택 조건: {chips}\n\n[확정된 장소]\n{stops}'),
    ]
)


def _chips(state: AgentState) -> CourseChips:
    raw = state.get("req", {}).get("chips") or {}
    try:
        return CourseChips(**raw)
    except Exception:  # noqa: BLE001 — 프론트 칩 값이 어긋나도 코스는 만들어야 한다
        logger.warning("칩 파싱 실패, 기본값 사용: %s", raw)
        return CourseChips()


def _cand(doc) -> dict:
    m = doc.metadata
    return {
        "name": m.get("display_name", ""),
        "lat": m.get("lat"),
        "lng": m.get("lng"),
        "category": m.get("category", ""),
        "area_name": m.get("area_name", ""),
        "text": doc.page_content,
    }


def _geo_rerank(
    pool: list[tuple[dict, float]], anchor: tuple[float, float], want: int
) -> list[dict]:
    """앵커 기준 반경 하드필터(부족하면 완화) + 거리 우선 블렌드 재정렬.

    pool: (후보, 의미거리) — 의미거리는 작을수록 유사(Chroma score).
    반환: 근접+유사 블렌드 상위 후보 dict[] (거리는 downstream 을 위해 dist_km 로 부착).
    """
    with_dist = [
        (c, sim, haversine_km(anchor[0], anchor[1], c["lat"], c["lng"]))
        for c, sim in pool
    ]
    # 반경 5→7→10km 로 완화하며 want 개 이상 확보. 그래도 없으면 반경 무시(폴백).
    near: list[tuple[dict, float, float]] = []
    for radius in _RADIUS_STEPS:
        near = [row for row in with_dist if row[2] <= radius]
        if len(near) >= want:
            break
    if not near:
        near = with_dist

    # 의미거리·지리거리를 각각 min-max 정규화 후 블렌드 (둘 다 작을수록 좋음 → 뒤집어 1=최상)
    sims = [s for _, s, _ in near]
    dists = [d for _, _, d in near]
    s_lo, s_hi = min(sims), max(sims)
    d_lo, d_hi = min(dists), max(dists)

    def _norm(v: float, lo: float, hi: float) -> float:
        return 1.0 if hi == lo else (hi - v) / (hi - lo)

    scored = [
        (dict(c, dist_km=round(d, 2)),
         _GEO_ALPHA * _norm(s, s_lo, s_hi) + (1 - _GEO_ALPHA) * _norm(d, d_lo, d_hi))
        for c, s, d in near
    ]
    scored.sort(key=lambda row: row[1], reverse=True)
    return [c for c, _ in scored]


async def retrieve_node(state: AgentState) -> dict:
    """칩+자연어 → RAG 후보. 코스는 지리 근접이 우선이라 앵커 반경 클러스터링 후 재랭킹.

    앵커: 위치 칩이 있으면 칩 중심좌표, 없으면 의미 유사도 1위 후보의 좌표
    (자연어 "종로에서…"도 상위 결과가 종로권이라 지오코딩 없이 클러스터가 잡힌다).
    """
    req = state.get("req", {})
    chips = _chips(state)

    query = " ".join(
        p for p in [req.get("note", ""), chips.purpose or "", chips.companion or "",
                    chips.location or "", chips.time or ""] if p
    ) or "서울 코스"

    # 지역 메타필터 없이 넉넉히 당긴다 — 4분면 경계에서 잘리지 않게, 반경으로 좁힌다
    scored = retriever.search_with_score(query, k=_FETCH_K, filters={"doc_type": "place"})
    pool = [
        (c, sim) for c, sim in ((_cand(d), s) for d, s in scored)
        if c["lat"] is not None and c["lng"] is not None
    ]
    if not pool:
        return {"candidates": [], "chips": chips.model_dump()}

    chip = chip_of(chips.location)
    anchor = (chip.lat, chip.lng) if chip else (pool[0][0]["lat"], pool[0][0]["lng"])

    want = chips.place_count + 6
    cands = _geo_rerank(pool, anchor, want)[:want]
    return {"candidates": cands, "chips": chips.model_dump()}


async def select_places_node(state: AgentState) -> dict:
    """AI 담당 구간 — 장소 선정 + 선정 이유 + 그곳에서 할 수 있는 일."""
    req = state.get("req", {})
    chips = _chips(state)
    cands = state.get("candidates", [])
    by_name = {c["name"]: c for c in cands}
    cand_lines = "\n".join(f"- {c['name']} ({c['category']}): {c['text'][:70]}" for c in cands)

    picked: list[dict] = []
    try:
        msg = await (_SELECT_PROMPT | get_llm()).ainvoke(
            {
                "today": date.today().isoformat(),
                "note": req.get("note", ""),
                "chips": chips.summary(),
                "candidates": cand_lines,
                "count": chips.place_count,
            }
        )
        data = parse_json_object(extract_text(msg.content))
        seen: set[str] = set()
        for row in data.get("stops", []) or []:
            name = row.get("name", "")
            if name in by_name and name not in seen:  # 화이트리스트 강제 (환각 차단)
                seen.add(name)
                picked.append(
                    {
                        **by_name[name],
                        "reason": row.get("reason", ""),
                        "activities": [a for a in row.get("activities", []) if a][:3],
                    }
                )
    except Exception as err:  # noqa: BLE001
        logger.warning("select LLM 실패, 후보 상위 사용: %s", err)

    if len(picked) < 2:  # 폴백: 후보 상위 (이유 없이)
        fallback = [{**c, "reason": "", "activities": []} for c in cands[: chips.place_count]]
        return {"selected": fallback, "source": "mock"}
    return {"selected": picked[: chips.place_count], "source": "ai"}


def _level(msg: str | None) -> str | None:
    return msg.split(":")[0].strip() if msg else None


async def enrich_node(state: AgentState) -> dict:
    """실시간 혼잡도 반영 — 혼잡도 선호 칩이 있으면 '붐빔' 장소를 후보로 교체."""
    chips = _chips(state)
    selected = state.get("selected", [])
    chosen = {s["name"] for s in selected}
    leftovers = [c for c in state.get("candidates", []) if c["name"] not in chosen]
    avoid_crowded = chips.congestion in ("여유", "보통")

    congestion: dict[str, str] = {}
    final: list[dict] = []
    for s in selected:
        lvl = _level(await get_congestion(s["area_name"])) if s.get("area_name") else None
        if lvl:
            congestion[s["name"]] = lvl
        if avoid_crowded and lvl == _CROWDED and leftovers:
            repl = leftovers.pop(0)
            rlvl = _level(await get_congestion(repl["area_name"])) if repl.get("area_name") else None
            if rlvl:
                congestion[repl["name"]] = rlvl
            final.append({**repl, "reason": f"{s['name']}이(가) 지금 붐벼서 한적한 대안으로 골랐어요.",
                          "activities": repl.get("activities", [])})
        else:
            final.append(s)
    return {"selected": final, "congestion": congestion}


def _card(item: NearbyItem, dist_km: float) -> dict:
    """dist_km 는 **해당 스톱 기준** 거리 (풀 앵커 기준인 item.dist_km 가 아니다)."""
    d = item.detail
    return {
        "title": d.title,
        "summary": d.summary or d.description[:100],
        "address": d.new_address or d.address,
        "lat": d.lat,
        "lng": d.lng,
        "dist_km": round(dist_km, 2),
        "period": f"{d.begin_date}~{d.end_date}" if d.begin_date or d.end_date else "",
        "use_time": d.use_time,
        "kind": item.kind,
    }


def _expired(item: NearbyItem, today: str) -> bool:
    """이미 끝난 행사 — Visit Seoul 목록에는 몇 해 전 축제도 그대로 남아 있다."""
    return bool(item.detail.end_date) and item.detail.end_date < today


def _assign(pool: list[NearbyItem], stop: dict, kinds: tuple[str, ...], today: str) -> list[dict]:
    """풀에서 이 스톱 반경 안의 해당 종류 항목만 가까운 순으로 뽑는다 (지난 행사 제외)."""
    near: list[tuple[float, NearbyItem]] = []
    for it in pool:
        d = it.detail
        if it.kind not in kinds or d.lat is None or d.lng is None or _expired(it, today):
            continue
        dist = haversine_km(stop["lat"], stop["lng"], d.lat, d.lng)
        if dist <= _NEARBY_RADIUS_KM:
            near.append((dist, it))
    near.sort(key=lambda row: row[0])
    return [_card(it, dist) for dist, it in near[:_NEARBY_PER_STOP]]


async def nearby_node(state: AgentState) -> dict:
    """Visit Seoul — 확정된 각 장소의 인근 식당 / 주변 문화행사·관광정보.

    목록 API 는 키워드 검색이라 **스톱 이름이 가장 좋은 단서**다 (keyword="경복궁" →
    별빛야행·집옥재·돌담길·서촌 카페). 스톱마다 따로 호출하면 rate limit(~1.4 req/s)에
    걸리므로, 스톱 이름들을 한 번에 키워드로 넘겨 풀을 만들고 반경으로 배분한다.
    """
    selected = state.get("selected", [])
    if not selected:
        return {"nearby": {}}
    chips = _chips(state)

    c_lat = sum(s["lat"] for s in selected) / len(selected)
    c_lng = sum(s["lng"] for s in selected) / len(selected)
    # 코스가 퍼져 있어도 모든 스톱 주변을 덮도록 풀 반경을 넓힌다
    spread = max(haversine_km(c_lat, c_lng, s["lat"], s["lng"]) for s in selected)
    pool_radius = spread + _NEARBY_RADIUS_KM

    terms = address_terms(chips.location)
    keywords = tuple(place_keyword(s["name"]) for s in selected) + ((terms[0],) if terms else ())

    try:
        pool = await search_nearby(
            lat=c_lat, lng=c_lng,
            keywords=keywords,
            kinds=(KIND_RESTAURANT, KIND_EVENT, KIND_ATTRACTION),
            radius_km=pool_radius,
            region_terms=terms,
            limit=_NEARBY_BUDGET,
            budget=_NEARBY_BUDGET,
        )
    except VisitSeoulError as err:
        logger.warning("visitseoul 주변 조회 실패: %s", err)
        pool = []

    today = date.today().isoformat()
    nearby = {
        s["name"]: {
            "restaurants": _assign(pool, s, (KIND_RESTAURANT,), today),
            "attractions": _assign(pool, s, (KIND_EVENT, KIND_ATTRACTION), today),
        }
        for s in selected
    }
    return {"nearby": nearby}


async def compose_node(state: AgentState) -> dict:
    """확정된 장소에 코스 서사를 입힌다. 장소·개수는 고정, 순서는 프론트가 다시 계산한다."""
    selected = state.get("selected", [])
    req = state.get("req", {})
    chips = _chips(state)
    congestion = state.get("congestion", {})
    nearby = state.get("nearby", {})

    stop_lines = "\n".join(
        f"- {s['name']} ({s.get('category','')})"
        f"{' · 지금 ' + congestion[s['name']] if s['name'] in congestion else ''}"
        f"{' · 선정 이유: ' + s['reason'] if s.get('reason') else ''}"
        for s in selected
    )

    narrative: dict[str, dict] = {}
    title = subtitle = description = ""
    tags: list[str] = []
    source = state.get("source", "ai")
    try:
        msg = await (_COMPOSE_PROMPT | get_llm()).ainvoke(
            {"note": req.get("note", "서울 코스"), "chips": chips.summary(), "stops": stop_lines}
        )
        data = parse_json_object(extract_text(msg.content))
        title = data.get("title", "")
        subtitle = data.get("subtitle", "")
        description = data.get("description", "")
        tags = data.get("tags", []) or []
        for s in data.get("stops", []):
            if s.get("name"):
                narrative[s["name"]] = s
    except Exception as err:  # noqa: BLE001
        logger.warning("compose LLM 실패, mock 서사: %s", err)
        source = "mock"

    stops = []
    for s in selected:
        n = narrative.get(s["name"], {})
        lvl = congestion.get(s["name"])
        tip = n.get("tip")
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
                "lat": s["lat"],
                "lng": s["lng"],
                "reason": s.get("reason", ""),
                "activities": s.get("activities", []),
                "congestion": lvl,
                "nearby": nearby.get(s["name"], {"restaurants": [], "attractions": []}),
            }
        )

    course = {
        "title": title or "서울 추천 코스",
        "subtitle": subtitle,
        "description": description,
        "stops": stops,
        "tags": tags or ["코스"],
    }
    return {"result": {"course": course, "source": source}, "source": source}
