"""course 파이프라인 — RAG 후보 → AI 장소 선정(+선정 이유) → 혼잡도 → 주변 정보 → 서사.

역할 분리:
 - AI: "어떤 장소를, 왜 골랐는지"와 "거기서 뭘 할 수 있는지(행동추천)".
   지금의 서울(오늘 날짜·시간대·실시간 혼잡도)에 맞춰 쓴다.
 - 주변 정보 3분할: 식당은 Visit Seoul 권역 캐시(meal_cache), 행사는 서울시 citydata
   실시간(events), 관광지는 임베딩 세트(seoul_places). nearby_node 는 식당·행사만 붙인다.
 - strangemap 프론트(courseRouting.ts): 방문 순서·지도 폴리라인·거리.
   → 폴리라인 경로는 프론트에서 작업 Ai 서빙은 해당 장소의 좌표값만 반환
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.core.geo import address_terms, chip_of, haversine_km, nearest_terms
from app.core.json_parse import parse_json_object
from app.core.scheduler import duration_label
from app.core.llm import extract_text, get_llm
from app.features.course.schema import CourseChips
from app.graph.state import AgentState
from app.rag import retriever
from app.tools.congestion import get_congestion
from app.tools.events import get_events
from app.tools.meal_cache import meal_card, pool_for_stops
from app.tools.visitseoul import (
    MEAL_KINDS,
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
# 거리 min-max 정규화 스케일 하한 — 후보가 좁게 모여 있을 때(수백 m) 미세한 거리 차이가
# 0~1 로 뻥튀기되어 다른 신호(유사도)를 눌러버리는 것을 막는다. 이 안은 다 도보권.
_DIST_SCALE_FLOOR_KM = 2.0
# 식사 추천 반경 — 앵커 장소 기준. 걸어갈 만한 1.5km 를 우선 보고, 3곳을 못 채울 때만
# 3km 로 넓힌다. Visit Seoul 음식 데이터가 자치구별로 크게 편중돼 있어서다
# (표본 실측: 종로 65 · 용산 51 · 마포 49 vs 관악 3 · 성북 5 — 관악·사당은 1.5km 로 못 채움).
MEAL_RADIUS_NEAR_KM = 1.5
MEAL_RADIUS_KM = 3.0

# 여행자 멀티데이 + 위치 "상관없음" → 날마다 다른 권역을 배정한다 (실제 여행 패턴).
# 1일차는 여행 목적에 맞는 권역으로 시작하고, 나머지는 회전 순서를 따른다.
_AREA_ROTATION = ("종로·중구", "홍대·마포", "성수·건대", "강남·서초", "용산·이태원", "잠실·송파")
_PURPOSE_FIRST_AREA = {
    "핫플레이스": "성수·건대", "체험·액티비티": "홍대·마포",
    "쇼핑": "강남·서초", "자연 힐링": "여의도·영등포",
}


def _day_areas(chips: CourseChips) -> dict[int, str] | None:
    """일차 → 권역 배정.

    동네를 여러 개(여행일수만큼) 고르면 그 순서대로 하루씩 배정한다.
    동네를 하나만 골랐으면 그 동네를 존중해 분산하지 않는다(None).
    "상관없음"(또는 미선택)이면 목적에 맞춰 자동으로 날마다 다른 권역을 돈다.
    """
    if chips.audience != "tourist" or chips.days <= 1:
        return None
    picked = [loc for loc in chips.locations if chip_of(loc)]
    if len(picked) >= 2:
        return {d: picked[(d - 1) % len(picked)] for d in range(1, chips.days + 1)}
    if picked:
        return None
    first = _PURPOSE_FIRST_AREA.get(chips.purposes[0] if chips.purposes else "", _AREA_ROTATION[0])
    order = [first, *[a for a in _AREA_ROTATION if a != first]]
    return {d: order[(d - 1) % len(order)] for d in range(1, chips.days + 1)}

_SELECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 '서울로' 코스 큐레이터다. 후보 목록에서만 장소를 골라 JSON 으로만 답하라.\n"
            'form: {{"days":[{{"day":1,"stops":[{{"name":"후보에 있는 장소명",'
            '"reason":"이 사람에게 왜 이 장소인지 한 문장",'
            '"activities":["거기서 할 수 있는 일 2~3개"]}}]}}]}}\n'
            "- 총 {days}일, 하루 정확히 {count}곳. 후보에 없는 이름은 절대 넣지 말고, "
            "같은 장소를 여러 날에 중복해 넣지 말 것.\n"
            "- 같은 날의 장소는 걸어서 이어질 만큼 가까운 곳끼리 묶을 것 (후보의 거리 참고).\n"
            "- 후보가 [N일차 후보] 로 나뉘어 있으면 각 일차는 그 일차의 후보에서만 고를 것.\n"
            "- 요청에 시간 범위가 있으면 그 시간에 문 닫는 장소는 고르지 말 것 (후보의 운영시간 참고).\n"
            "{purpose_rule}"
            "{companion_rule}"
            "{purpose_act}"
            "- reason 은 후보의 소개 글·운영시간·특징 등 주어진 데이터에 근거해 구체적으로 쓴다. "
            "선택 조건(동반·목적·시간대)과 오늘의 서울(날짜/계절)을 엮되, "
            "어느 장소에나 붙일 수 있는 일반론은 금지.\n"
            "- activities 는 그 장소에서 실제로 할 수 있는 행동인데, **누구와(동반) 왔는지 × 무엇을 하러(목적) "
            "왔는지에 따라 같은 장소라도 하는 일이 달라진다.** 위의 동반 렌즈와 목적별 행동 결을 곱해, "
            "그 장소의 소개 글에 실제로 등장하는 특징(시설·풍경·볼거리)에 걸어 행동을 2~3개 제안하라.\n"
            "- **행동은 서로 다른 유형으로 벌릴 것** — 관람/체험/먹거리/산책·사진/쇼핑 등 한 장소 안에서 같은 "
            "유형만 반복하지 말고, 코스 전체에서도 스톱마다 행동이 겹치지 않게 다양화한다.\n"
            "- 가게 상호를 지어내지 말 것(환각 금지). 그 장소·권역에서 실제로 할 만한 '행동 유형'으로 쓴다. "
            "예: 친구와+놀거리 잠실 → ['야구 경기 관람','보드게임 카페에서 한 판'] / "
            "연인과+데이트 잠실 → ['석촌호수 산책','팝업스토어 구경'] / "
            "부모님과+문화·예술·역사 고궁 → ['천천히 정원 산책','고궁 해설 투어'].\n"
            "- 하루 안에서는 자연스러운 방문 흐름 순서로 나열 (시간표는 서버가 다시 계산한다).",
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
            "- **각 장소의 description 은 그 장소에 주어진 '여기서 할 일'(activities)을 자연스럽게 녹여 쓸 것** — "
            "동행자와 목적에 따라 같은 장소라도 목적에 따라 행동 추천이 다르게 그려도록 현장감 있게 서술한다. "
            "activities 에 없는 엉뚱한 행동을 새로 지어내지 말 것.\n"
            "- title·subtitle·description(코스 전체)은 선택 조건(동반·목적)의 결에 맞는 하나의 흐름으로 엮는다.\n"
            "- 실시간 혼잡도가 주어진 장소는 그 상황을 description 에 자연스럽게 반영.\n"
            "- 방문 시각(예: 14:00~15:30)이 주어진 장소는 그 시간대에 맞는 서사를 쓰되 "
            "시각은 이미 확정된 값이니 바꾸거나 언급을 지어내지 말 것.\n"
            "- 장소에 일차(예: 2일차)가 붙어 있으면 일차 흐름이 이어지게 description 을 쓸 것.\n"
            "- description 은 선정 이유에 담긴 데이터(운영시간·특징)와 어긋나지 않게 쓸 것.",
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
        "op_start": int(m.get("op_start", 0)),
        "op_end": int(m.get("op_end", 24)),
        "text": doc.page_content,
    }


def _hours_label(c: dict) -> str:
    return "상시개방" if (c["op_start"], c["op_end"]) == (0, 24) else f"운영 {c['op_start']}시~{c['op_end']}시"


def _geo_rerank(
    pool: list[tuple[dict, float]], anchor: tuple[float, float], want: int,
) -> list[dict]:
    """앵커 기준 반경 하드필터(부족하면 완화) + 거리 우선 블렌드 재정렬.

    pool: (후보, 의미거리) — 의미거리는 작을수록 유사(Chroma score).
    반환: 블렌드 상위 후보 dict[] (거리는 downstream 을 위해 dist_km 로 부착).
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
    d_lo = min(dists)
    d_hi = max(max(dists), d_lo + _DIST_SCALE_FLOOR_KM)

    def _norm(v: float, lo: float, hi: float) -> float:
        return 1.0 if hi == lo else (hi - v) / (hi - lo)

    w_sim, w_dist = _GEO_ALPHA, 1 - _GEO_ALPHA
    scored = [
        (dict(c, dist_km=round(d, 2)),
         w_sim * _norm(s, s_lo, s_hi) + w_dist * _norm(d, d_lo, d_hi))
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
    rule = chips.purpose_rule()

    # 목적 칩은 라벨 그대로가 아니라 task 조건(PURPOSE_RULES)의 검색 확장어로 검색한다
    query = " ".join(
        p for p in [req.get("note", ""), rule.get("query", " ".join(chips.purposes)),
                    " ".join(chips.companions), " ".join(chips.locations), chips.time or ""] if p
    ) or "서울 코스"

    # 지역 메타필터 없이 넉넉히 당긴다 — 4분면 경계에서 잘리지 않게, 반경으로 좁힌다
    scored = retriever.search_with_score(query, k=_FETCH_K, filters={"doc_type": "place"})
    pool = [
        (c, sim) for c, sim in ((_cand(d), s) for d, s in scored)
        if c["lat"] is not None and c["lng"] is not None
    ]
    if not pool:
        return {"candidates": [], "chips": chips.model_dump()}

    window = chips.resolved_window()
    day_areas = _day_areas(chips)
    if day_areas:
        # 일차별 권역 앵커로 같은 풀을 재랭킹 → day_hint 를 붙인 일차별 후보 그룹.
        # 앞 일차가 쓴 장소는 다음 일차에서 제외한다.
        per = chips.stops_per_day() + 2
        taken: set[str] = set()
        cands = []
        for d, area in day_areas.items():
            area_chip = chip_of(area)
            pool_d = [(c, s) for c, s in pool if c["name"] not in taken]
            if not pool_d:
                break
            picks = _geo_rerank(pool_d, (area_chip.lat, area_chip.lng), per)[:per]
            taken.update(c["name"] for c in picks)
            cands.extend({**c, "day_hint": d} for c in picks)
        if window:  # 일차 그룹 순서는 지키고, 그룹 안에서만 문 여는 곳을 앞세운다
            cands.sort(key=lambda c: (c["day_hint"],
                                      not window.overlaps(c["op_start"], c["op_end"])))
        return {"candidates": cands, "chips": chips.model_dump(), "day_areas": day_areas}

    chip = next((chip_of(loc) for loc in chips.locations if chip_of(loc)), None)
    anchor = (chip.lat, chip.lng) if chip else (pool[0][0]["lat"], pool[0][0]["lng"])

    want = chips.days * chips.stops_per_day() + 6
    cands = _geo_rerank(pool, anchor, want)

    # 시간 범위가 있으면 그 시간에 문 여는 장소 우선 — 부족할 때만 나머지를 꼬리에 남긴다
    if window:
        open_c = [c for c in cands if window.overlaps(c["op_start"], c["op_end"])]
        closed = [c for c in cands if not window.overlaps(c["op_start"], c["op_end"])]
        cands = open_c if len(open_c) >= want else open_c + closed

    return {"candidates": cands[:want], "chips": chips.model_dump()}


async def select_places_node(state: AgentState) -> dict:
    """AI 담당 구간 — 장소 선정(일자별) + 데이터에 근거한 선정 이유 + 할 수 있는 일."""
    req = state.get("req", {})
    chips = _chips(state)
    cands = state.get("candidates", [])
    by_name = {c["name"]: c for c in cands}

    # 소개 글을 넉넉히 실어 reason·activities 가 실제 데이터(RAG 본문)를 근거로 쓰게 한다.
    # 장소가 지역구 단위라 본문에 "그 권역에 뭐가 있는지"가 담겨 있어야 행동 추천이 구체화된다.
    def _cand_line(c: dict) -> str:
        return (
            f"- {c['name']} ({c['category']} · {_hours_label(c)}"
            f"{' · 앵커에서 ' + str(c['dist_km']) + 'km' if c.get('dist_km') is not None else ''}"
            f"): {c['text'][:280]}"
        )

    # 일차별 권역 분산이면 후보를 [N일차 후보 — 권역] 으로 묶어 보여준다
    day_areas = state.get("day_areas") or {}
    if day_areas:
        lines, cur = [], None
        for c in cands:
            if c.get("day_hint") != cur:
                cur = c.get("day_hint")
                lines.append(f"[{cur}일차 후보 — {day_areas.get(cur, '')}]")
            lines.append(_cand_line(c))
        cand_lines = "\n".join(lines)
    else:
        cand_lines = "\n".join(_cand_line(c) for c in cands)
    pr = chips.purpose_rule()
    rule = pr.get("rule", "")
    act = pr.get("act", "")
    comp_rule = chips.companion_rule()
    per_day = chips.stops_per_day()

    picked: list[dict] = []
    try:
        msg = await (_SELECT_PROMPT | get_llm()).ainvoke(
            {
                "today": date.today().isoformat(),
                "note": req.get("note", ""),
                "chips": chips.summary(),
                "candidates": cand_lines,
                "count": per_day,
                "days": chips.days,
                "purpose_rule": f"- 목적 조건: {rule}\n" if rule else "",
                "companion_rule": f"- 동반 조건(행동 렌즈): {comp_rule}\n" if comp_rule else "",
                "purpose_act": f"- 목적별 행동 결(행동 렌즈): {act}\n" if act else "",
            }
        )
        data = parse_json_object(extract_text(msg.content))
        # 구형 응답({"stops":[...]})도 1일차로 수용
        day_rows = data.get("days") or [{"day": 1, "stops": data.get("stops", [])}]
        seen: set[str] = set()
        for i, day_row in enumerate(day_rows[: chips.days]):
            day_no = int(day_row.get("day") or i + 1)
            for row in (day_row.get("stops") or [])[:per_day]:
                name = row.get("name", "")
                if name in by_name and name not in seen:  # 화이트리스트 강제 (환각 차단)
                    seen.add(name)
                    cand = by_name[name]
                    # 일차별 권역 분산이면 LLM 이 섞어 담아도 day_hint 가 일차를 확정한다
                    day = cand.get("day_hint") or day_no
                    picked.append(
                        {
                            **cand,
                            "day": day if chips.days > 1 else None,
                            "reason": row.get("reason", ""),
                            "activities": [a for a in row.get("activities", []) if a][:3],
                        }
                    )
    except Exception as err:  # noqa: BLE001
        logger.warning("select LLM 실패, 후보 상위 사용: %s", err)

    if len(picked) < 2:  # 폴백: 후보 상위를 일자별로 나눠 담는다 (이유 없이)
        fallback = [
            {**c, "day": (c.get("day_hint") or i // per_day + 1) if chips.days > 1 else None,
             "reason": "", "activities": []}
            for i, c in enumerate(cands[: per_day * chips.days])
        ]
        return {"selected": fallback, "source": "mock"}
    return {"selected": picked, "source": "ai"}


def _level(msg: str | None) -> str | None:
    return msg.split(":")[0].strip() if msg else None


async def enrich_node(state: AgentState) -> dict:
    """실시간 혼잡도 반영 — 혼잡도 선호 칩이 있을 때만 조회하고, '붐빔' 장소는 후보로 교체.

    칩이 없으면 API 호출 없이 통과한다 (불필요한 혼잡도 주입으로 인한 레이턴시 제거).
    조회할 때는 선택 장소 + 교체 예비 후보를 한 번에 병렬 조회한다 (순차 대기 제거).
    """
    chips = _chips(state)
    selected = state.get("selected", [])
    if chips.congestion not in ("여유", "보통") or not selected:
        return {"congestion": {}}

    # 실시간 혼잡도는 "지금"의 값 — 멀티데이 코스에선 오늘 도는 1일차에만 반영한다
    day1 = [s for s in selected if (s.get("day") or 1) == 1]
    chosen = {s["name"] for s in selected}
    leftovers = [c for c in state.get("candidates", []) if c["name"] not in chosen]
    spares = leftovers[:3]  # 붐빔 교체용 예비 — 같은 배치에서 미리 조회해 추가 왕복 제거

    async def _lvl(stop: dict) -> str | None:
        if not stop.get("area_name"):
            return None
        return _level(await get_congestion(stop["area_name"]))

    targets = day1 + spares
    levels = await asyncio.gather(*(_lvl(t) for t in targets))
    lvl_by_name = {t["name"]: lvl for t, lvl in zip(targets, levels)}

    final: list[dict] = []
    spares_left = list(spares)
    for s in selected:
        if (s.get("day") or 1) == 1 and lvl_by_name.get(s["name"]) == _CROWDED and spares_left:
            # 예비 중 붐비지 않는 곳 우선, 전부 붐비면 첫 예비라도 사용 (기존 동작 유지)
            repl = next((c for c in spares_left if lvl_by_name.get(c["name"]) != _CROWDED),
                        spares_left[0])
            spares_left.remove(repl)
            final.append({**repl, "day": s.get("day"),
                          "reason": f"{s['name']}이(가) 지금 붐벼서 한적한 대안으로 골랐어요.",
                          "activities": repl.get("activities", [])})
            continue
        final.append(s)

    congestion = {s["name"]: lvl_by_name[s["name"]] for s in final if lvl_by_name.get(s["name"])}
    return {"selected": final, "congestion": congestion}


def _nearest_cards(pool: list[dict], stop: dict, radius_km: float, n: int) -> list[dict]:
    """식당 풀에서 이 스톱 반경 안 카드를 가까운 순 n개 (스톱 기준 dist_km 부착)."""
    near: list[tuple[float, dict]] = []
    for r in pool:
        if r.get("lat") is None or r.get("lng") is None:
            continue
        dist = haversine_km(stop["lat"], stop["lng"], r["lat"], r["lng"])
        if dist <= radius_km:
            near.append((dist, r))
    near.sort(key=lambda row: row[0])
    return [{**r, "dist_km": round(dist, 2)} for dist, r in near[:n]]


def _event_cards(events: list[dict], stop: dict, n: int) -> list[dict]:
    """스톱이 속한 명소의 실시간 행사 상위 n개 — 좌표가 있으면 거리도 붙인다.

    행사는 이미 그 스톱의 명소(area_name)에 속하므로 반경 필터로 걸러내지 않는다.
    """
    out: list[dict] = []
    for e in events[:n]:
        dist = (round(haversine_km(stop["lat"], stop["lng"], e["lat"], e["lng"]), 2)
                if e.get("lat") is not None and e.get("lng") is not None else None)
        out.append({**e, "dist_km": dist})
    return out


async def _live_meal_pool(selected: list[dict], chips: CourseChips) -> list[dict]:
    """권역 캐시가 비었을 때의 폴백 — 스톱 이름 + 동네 키워드로 식당을 라이브 조회.

    목록 API 는 키워드 검색이라 스톱 이름이 가장 좋은 단서다 (keyword="경복궁" →
    별빛야행·서촌 카페). 스톱마다 부르면 rate limit(~1.4 req/s)에 걸리므로 이름들을
    한 번에 키워드로 넘겨 식당 풀을 만든다. (예전 nearby 로직 — 이제 식당 전용.)
    """
    c_lat = sum(s["lat"] for s in selected) / len(selected)
    c_lng = sum(s["lng"] for s in selected) / len(selected)
    spread = max(haversine_km(c_lat, c_lng, s["lat"], s["lng"]) for s in selected)
    pool_radius = spread + max(_NEARBY_RADIUS_KM, MEAL_RADIUS_KM)

    terms = tuple(dict.fromkeys(t for loc in chips.locations for t in address_terms(loc)))
    keywords = tuple(place_keyword(s["name"]) for s in selected) + ((terms[0],) if terms else ())
    extra_terms = [
        t for s in selected for t in nearest_terms(s["lat"], s["lng"])
        if t not in keywords
    ]
    rest_keywords = keywords + tuple(dict.fromkeys(extra_terms))[:4]

    # 시간표 요청은 끼니마다 식당 3곳 이상이 필요해 예산을 더 태운다 (상세 1건당 ~0.75초).
    budget = min(8 + 6 * chips.days, 26) if chips.resolved_window() else _NEARBY_BUDGET
    try:
        items = await search_nearby(
            lat=c_lat, lng=c_lng, radius_km=pool_radius, region_terms=terms,
            keywords=rest_keywords, kinds=MEAL_KINDS, limit=budget, budget=budget,
        )
    except VisitSeoulError as err:
        logger.warning("visitseoul 식당 폴백 조회 실패: %s", err)
        return []

    pool: list[dict] = []
    seen: set[str] = set()
    for it in items:
        d = it.detail
        if d.lat is None or d.lng is None or d.title in seen:
            continue
        seen.add(d.title)
        pool.append(meal_card(d, it.kind))
    return pool


async def nearby_node(state: AgentState) -> dict:
    """확정된 각 장소의 주변 정보 — 식당(권역 캐시) + 실시간 행사(citydata).

    3분할: 코스 장소는 임베딩(seoul_places)이, 식당은 Visit Seoul 권역 캐시가,
    행사는 서울시 citydata 가 담당한다. 여기서는 식당·행사만 스톱에 붙인다.
    """
    selected = state.get("selected", [])
    if not selected:
        return {"nearby": {}}
    chips = _chips(state)

    # 식당 — 권역 캐시 우선. 캐시가 없거나 얇으면 라이브 조회로 보충 (제목 중복 제거).
    pool = pool_for_stops(selected)
    if len(pool) < get_settings().meal_pool_min:
        seen = {r["title"] for r in pool}
        pool += [r for r in await _live_meal_pool(selected, chips) if r["title"] not in seen]

    # 행사 — 스톱이 속한 명소(area_name)의 citydata 실시간 행사. 명소당 한 번만 조회.
    areas = [a for a in {s.get("area_name") for s in selected} if a]
    events_by_area = dict(zip(areas, await asyncio.gather(*(get_events(a) for a in areas))))

    nearby = {
        s["name"]: {
            "restaurants": _nearest_cards(pool, s, _NEARBY_RADIUS_KM, _NEARBY_PER_STOP),
            "attractions": _event_cards(events_by_area.get(s.get("area_name"), []), s, _NEARBY_PER_STOP),
        }
        for s in selected
    }
    return {"nearby": nearby, "meal_pool": pool}


async def compose_node(state: AgentState) -> dict:
    """확정된 장소에 코스 서사를 입힌다. 장소·개수는 고정, 순서는 프론트가 다시 계산한다."""
    selected = state.get("selected", [])
    req = state.get("req", {})
    chips = _chips(state)
    congestion = state.get("congestion", {})
    nearby = state.get("nearby", {})
    schedule = state.get("schedule", [])
    sched_by_name = {s["name"]: s for s in schedule if s.get("slot_type") == "place"}
    flex_names = {s["name"] for s in schedule if s.get("slot_type") == "flex"}

    def _slot_label(name: str) -> str:
        slot = sched_by_name.get(name)
        if slot:
            return f" · {slot['start_time']}~{slot['end_time']}"
        return " · 시간표 밖 자유 방문 제안" if name in flex_names else ""

    stop_lines = "\n".join(
        f"- {str(s['day']) + '일차 · ' if s.get('day') else ''}{s['name']} ({s.get('category','')})"
        f"{_slot_label(s['name'])}"
        f"{' · 지금 ' + congestion[s['name']] if s['name'] in congestion else ''}"
        f"{' · 선정 이유: ' + s['reason'] if s.get('reason') else ''}"
        f"{' · 여기서 할 일: ' + ', '.join(s['activities']) if s.get('activities') else ''}"
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

    def _place_stop(s: dict) -> dict:
        n = narrative.get(s["name"], {})
        lvl = congestion.get(s["name"])
        tip = n.get("tip")
        if lvl and lvl != "여유":
            note = f"현재 혼잡도 {lvl}"
            tip = f"{tip} · {note}" if tip else note
        slot = sched_by_name.get(s["name"])
        # 시간표가 있으면 체류시간은 계산값 — LLM 자유 텍스트가 아니라 시간표와 항상 일치한다
        duration = duration_label(slot["duration_min"]) if slot else n.get("duration", "1시간")
        return {
            "name": s["name"],
            "preview": n.get("preview", s.get("text", "")[:40]),
            "description": n.get("description", ""),
            "duration": duration,
            "tip": tip,
            "lat": s["lat"],
            "lng": s["lng"],
            "reason": s.get("reason", ""),
            "activities": s.get("activities", []),
            "congestion": lvl,
            "nearby": nearby.get(s["name"], {"restaurants": [], "attractions": []}),
            "start_time": slot["start_time"] if slot else None,
            "end_time": slot["end_time"] if slot else None,
            "slot_type": "flex" if s["name"] in flex_names else "place",
            "day": s.get("day"),
            "meal_options": [],
            "travel_min": (slot.get("travel_min") or None) if slot else None,
            "travel_mode": (slot.get("travel_mode") or None) if slot else None,
        }

    def _meal_stop(m: dict) -> dict:
        return {
            "name": m["name"],
            "preview": m.get("summary", ""),
            "description": "",
            "duration": "1시간",
            "tip": None,
            "lat": m.get("lat"),
            "lng": m.get("lng"),
            "reason": "",
            "activities": [],
            "congestion": None,
            "nearby": {"restaurants": [], "attractions": []},
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "slot_type": "meal",
            "day": m.get("day"),
            "meal_options": m.get("meal_options", []),
        }

    by_name = {s["name"]: s for s in selected}
    if schedule:
        # 시간표 순서 그대로 — 장소 사이에 식사 슬롯이 끼어든다
        stops = [
            _meal_stop(slot) if slot["slot_type"] == "meal" else _place_stop(by_name[slot["name"]])
            for slot in schedule
            if slot["slot_type"] == "meal" or slot["name"] in by_name
        ]
    else:
        stops = [_place_stop(s) for s in selected]

    course = {
        "title": title or "서울 추천 코스",
        "subtitle": subtitle,
        "description": description,
        "stops": stops,
        "tags": tags or ["코스"],
        "scheduled": bool(sched_by_name),
        "days": chips.days,
        "day_areas": state.get("day_areas") or {},
    }
    return {"result": {"course": course, "source": source}, "source": source}
