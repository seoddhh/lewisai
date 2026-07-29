"""LangGraph StateGraph 조립 + 컴파일(싱글턴) — 서울 코스 생성 에이전트.

흐름:
  START → parse_intent → plan → retrieve → select_places
        → fit_schedule → meals → enrich → nearby → compose → END

각 단계가 왜 그 자리인지:
  - plan          칩만으로 시간 골격(식사 앵커 + 장소 구간)을 만든다. 순수 계산이라 맨 앞.
                  이 골격이 select 프롬프트로 들어가 LLM 이 구간을 알고 장소를 고른다.
  - fit_schedule  구간 예산에 맞춰 장소를 앉히고 방문 순서·시각을 확정한다.
  - meals         고른 끼니에 실제 식당 후보를 붙인다. 시간표가 정해진 뒤라야 앵커가 잡힌다.
  - enrich        방문 시각을 알아야 그 시(時)의 혼잡도 예보를 붙일 수 있다.
  - nearby        스톱 주변 카드(식당·행사). 최종 payload 에만 쓰여 마지막.

에이전트의 임무는 코스 생성이다. 잡담 분기(chitchat)는 제거했다 — 클라이언트에 잡담
입력창이 없고, 칩 경로에서는 router 가 어차피 무동작이었다. 잡담이 필요하면
/agent/chitchat 라우트가 별도로 살아 있다.

지도 폴리라인·실경로 거리는 서버가 만들지 않는다 (strangemap courseRouting.ts 담당).
단, 방문 순서·시각은 서버가 확정한다.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.common import parse_intent_node
from app.graph.nodes.course import (
    compose_node,
    enrich_node,
    nearby_node,
    retrieve_node,
    select_places_node,
)
from app.graph.nodes.meals import meals_node
from app.graph.nodes.planning import plan_node
from app.graph.nodes.schedule import fit_schedule_node
from app.graph.state import AgentState


@lru_cache
def get_agent_graph():
    g = StateGraph(AgentState)

    g.add_node("parse_intent", parse_intent_node)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("select_places", select_places_node)
    g.add_node("fit_schedule", fit_schedule_node)
    g.add_node("meals", meals_node)
    g.add_node("enrich", enrich_node)
    g.add_node("nearby", nearby_node)
    g.add_node("compose", compose_node)

    g.add_edge(START, "parse_intent")
    g.add_edge("parse_intent", "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "select_places")
    g.add_edge("select_places", "fit_schedule")
    g.add_edge("fit_schedule", "meals")
    g.add_edge("meals", "enrich")
    g.add_edge("enrich", "nearby")
    g.add_edge("nearby", "compose")
    g.add_edge("compose", END)

    return g.compile()


async def run_agent(
    *, message: str | None = None, intent: str | None = None, req: dict | None = None
) -> dict[str, Any]:
    """그래프 1회 실행. 자연어(message) 또는 어댑터(intent+req) 진입 모두 지원."""
    state: dict[str, Any] = {}
    if message is not None:
        state["message"] = message
    if intent is not None:
        state["intent"] = intent
    if req is not None:
        state["req"] = req
    return await get_agent_graph().ainvoke(state)


# ── 코스 생성 과정 노출 (칩/자연어 → 실제 장소가 어떻게 나왔는지) ────────────

COURSE_STEP_LABELS: dict[str, str] = {
    "plan": "시간 골격 구성 (식사 시각·장소 구간)",
    "retrieve": "후보 장소 검색 (RAG)",
    "select_places": "AI 장소 선정",
    "fit_schedule": "시간표 배치 (구간 예산 배분)",
    "meals": "식사 장소 추천",
    "enrich": "방문 시각 예상 혼잡도 반영",
    "nearby": "주변 식당·문화행사 조회",
    "compose": "코스 서사 작성",
}


def course_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """최종 상태 → 챗봇에 보여줄 생성 단계에서의 장소선택 이유
    """
    selected = state.get("selected", [])
    nearby = state.get("nearby", {})
    schedule = state.get("schedule", [])
    n_nearby = sum(len(v.get("restaurants", [])) + len(v.get("attractions", []))
                   for v in nearby.values())
    sk = state.get("skeleton") or {}
    meal_slots = [s for s in schedule if s.get("slot_type") == "meal"]
    n_meal_opts = sum(len(s.get("meal_options") or []) for s in meal_slots)
    return [
        {"id": "s1", "tool": "plan", "label": COURSE_STEP_LABELS["plan"],
         "ok": bool(sk.get("segments")), "detail": _plan_detail(sk)},
        {"id": "s2", "tool": "retrieve", "label": COURSE_STEP_LABELS["retrieve"], "ok": True,
         "detail": f"후보 {len(state.get('candidates', []))}곳"},
        {"id": "s3", "tool": "select_places", "label": COURSE_STEP_LABELS["select_places"],
         "ok": bool(selected),
         "detail": ", ".join(s["name"] for s in selected),
         "picks": [{"name": s["name"], "reason": s.get("reason", ""),
                    "activities": s.get("activities", [])} for s in selected]},
        {"id": "s4", "tool": "fit_schedule", "label": COURSE_STEP_LABELS["fit_schedule"],
         "ok": bool(schedule), "detail": _schedule_detail(schedule)},
        {"id": "s5", "tool": "meals", "label": COURSE_STEP_LABELS["meals"],
         "ok": n_meal_opts > 0,
         "detail": (f"{len(meal_slots)}끼 · 식당 후보 {n_meal_opts}곳"
                    if meal_slots else "식사 미선택 — 건너뜀")},
        {"id": "s6", "tool": "enrich", "label": COURSE_STEP_LABELS["enrich"], "ok": True,
         "detail": ", ".join(f"{k} {v}" for k, v in state.get("congestion", {}).items())
                   or "예상 혼잡도 측정 지점 없음"},
        {"id": "s7", "tool": "nearby", "label": COURSE_STEP_LABELS["nearby"], "ok": n_nearby > 0,
         "detail": f"주변 정보 {n_nearby}건"},
        {"id": "s8", "tool": "compose", "label": COURSE_STEP_LABELS["compose"],
         "ok": state.get("source") == "ai", "detail": (state.get("result") or {})
             .get("course", {}).get("title", "")},
    ]


def _plan_detail(sk: dict) -> str:
    if not sk.get("segments"):
        return "시간 범위 없음 — 시간표 생략"
    meals = "·".join(m["label"] for m in sk.get("meals", [])) or "없음"
    return f"구간 {len(sk['segments'])}개 · 식사 {meals}"


def _schedule_detail(schedule: list[dict]) -> str:
    timed = [s for s in schedule if s.get("start_time")]
    if not timed:
        return "시간 범위 없음 — 건너뜀"
    meals = sum(1 for s in schedule if s.get("slot_type") == "meal")
    n_days = len({s.get("day") or 1 for s in schedule})
    day_label = f"{n_days}일 · " if n_days > 1 else ""
    return (f"{day_label}{timed[0]['start_time']}~{timed[-1]['end_time']} · "
            f"장소 {len(timed) - meals}곳 · 식사 {meals}회")


def _payload(state: dict[str, Any]) -> dict[str, Any]:
    """그래프 최종 상태 → chat 응답 payload.

    course: stops[] (name/lat/lng/reason/activities/nearby) + 생성 단계 트레이스.
    이 에이전트는 코스만 만든다 — 잡담 분기는 그래프에서 제거했다.
    """
    result = state.get("result") or {}
    return {
        "kind": "course",
        "course": result.get("course", {}),
        "source": result.get("source", state.get("source", "ai")),
        "steps": course_steps(state),
    }


async def run_course(note: str, chips: dict[str, Any]) -> dict[str, Any]:
    """칩(+자연어) → 코스 (결정적 진입)."""
    return _payload(await run_agent(intent="course", req={"note": note, "chips": chips}))


async def run_chat(message: str) -> dict[str, Any]:
    """자연어 → 코스. parse_intent 가 칩 형태로 구조화한 뒤 같은 파이프라인을 탄다."""
    return _payload(await run_agent(message=message, intent="course"))


async def _stream(initial: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """그래프 진행 상황/최종 payload 이벤트를 순서대로 yield.

    이벤트 종류(event):
      - progress: {"stage": plan|retrieve|select_places|fit_schedule|meals|enrich|nearby|compose}
      - final:    {"payload": {...}} 최종 구조화 응답(코스/steps 렌더링용).

    코스는 출력이 JSON 이라 토큰을 흘리지 않는다 (잡담 분기가 없어져 token 이벤트도 사라짐).
    """
    graph = get_agent_graph()
    state: dict[str, Any] = {}

    async for node, patch in _updates(graph, initial):
        if not isinstance(patch, dict):
            continue
        state.update(patch)
        if node in COURSE_STEP_LABELS:
            yield {"event": "progress", "stage": node,
                   "message": COURSE_STEP_LABELS[node] + " 완료"}

    yield {"event": "final", "payload": _payload(state)}


async def _updates(graph, initial: dict[str, Any]):
    async for data in graph.astream(initial, stream_mode="updates"):
        for node, patch in data.items():
            yield node, patch


async def stream_course(note: str, chips: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """칩 → 코스 스트리밍."""
    async for evt in _stream({"intent": "course", "req": {"note": note, "chips": chips}}):
        yield evt


async def stream_chat(message: str) -> AsyncIterator[dict[str, Any]]:
    """자연어 → 코스 스트리밍."""
    async for evt in _stream({"message": message, "intent": "course"}):
        yield evt
