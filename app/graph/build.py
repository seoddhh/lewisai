"""코스를 만드는 단계들을 순서대로 이어 붙이는 곳(LangGraph). 그래프는 한 번만 만들어 재사용한다.

  parse_intent → plan → retrieve → select_places
  → fit_schedule → meals → enrich → nearby → compose

순서가 이런 이유:
  - plan          칩만 보고 시간 뼈대(식사 시각·장소가 들어갈 구간)를 짠다. 계산만 하므로 맨 앞.
                  이 뼈대를 AI 에게 같이 줘야 구간에 맞는 장소를 고른다.
  - retrieve      뼈대에 맞는 후보 장소를 검색해 온다.
  - select_places AI 가 후보 중에서 실제로 갈 곳을 고르고 이유를 쓴다.
  - fit_schedule  고른 장소를 시간에 앉혀 방문 순서와 시각을 확정한다.
  - meals         정해진 식사 시각 주변에서 실제 식당을 찾는다. 시간표가 나온 뒤라야 가능하다.
  - enrich        방문 시각을 알아야 그 시간대의 예상 혼잡도를 붙일 수 있다.
  - nearby        장소 주변 식당·행사 카드. 마지막 응답에만 쓰이므로 맨 뒤.
  - compose       코스 제목과 소개 문구를 쓴다.

방문 순서와 시각까지는 서버가 정한다. 지도에 그릴 실제 경로와 거리는 프론트 담당이다.
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
    *, message: str | None = None, req: dict | None = None
) -> dict[str, Any]:
    """그래프를 한 번 돌린다. 사용자가 쓴 문장(message)이나 칩(req) 중 하나를 넣어 준다.

    둘 중 뭘로 들어왔는지는 parse_intent 가 req 유무로 판단한다.
    """
    state: dict[str, Any] = {}
    if message is not None:
        state["message"] = message
    if req is not None:
        state["req"] = req
    return await get_agent_graph().ainvoke(state)


# ── 코스가 만들어진 과정을 챗봇에 보여주기 위한 부분 ──────────────────────────

# 단계 이름 → 사용자에게 보여줄 문구
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
    """각 단계에서 무슨 일이 있었는지를 목록으로 만든다. 챗봇이 이걸 그대로 보여준다.

    특히 select_places 단계에는 어떤 장소를 왜 골랐는지(picks)가 들어간다.
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
    """그래프가 끝난 상태를 API 응답 모양으로 바꾼다."""
    result = state.get("result") or {}
    return {
        "kind": "course",
        "course": result.get("course", {}),
        "source": result.get("source", state.get("source", "ai")),
        "steps": course_steps(state),
    }


async def run_course(note: str, chips: dict[str, Any], *,
                     seed: int | None = None) -> dict[str, Any]:
    """칩으로 코스를 만든다.

    seed 는 후보를 뽑을 때의 무작위성만 정한다. 안 주면 칩 내용으로 정해져서
    같은 요청에는 같은 코스가 나오고, 주면 그 값에 따라 다른 코스가 나온다.
    """
    req: dict[str, Any] = {"note": note, "chips": chips}
    if seed is not None:
        req["seed"] = seed
    return _payload(await run_agent(req=req))


async def run_chat(message: str) -> dict[str, Any]:
    """사용자가 쓴 문장으로 코스를 만든다. parse_intent 가 문장을 칩 모양으로 바꾼 뒤 같은 길을 탄다."""
    return _payload(await run_agent(message=message))


async def _stream(initial: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """단계가 하나 끝날 때마다 진행 상황을, 다 끝나면 최종 결과를 내보낸다.

    보내는 이벤트는 두 가지다.
      - progress: 어느 단계가 끝났는지
      - final:    완성된 코스 전체

    코스는 결과가 JSON 이라 글자 단위로 흘려보내지 않는다. 단계별로만 알려준다.
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


async def stream_course(note: str, chips: dict[str, Any], *,
                        seed: int | None = None) -> AsyncIterator[dict[str, Any]]:
    """칩으로 코스를 만들면서 진행 상황을 내보낸다.

    프론트가 실제로 쓰는 경로가 이쪽이다. run_course 를 고칠 일이 있으면 여기도 같이 봐야 한다.
    (예전에 seed 를 여기에만 안 넘겨서 "다시 만들기"가 조용히 안 먹은 적이 있다.)
    """
    req: dict[str, Any] = {"note": note, "chips": chips}
    if seed is not None:
        req["seed"] = seed
    async for evt in _stream({"req": req}):
        yield evt


async def stream_chat(message: str) -> AsyncIterator[dict[str, Any]]:
    """자연어 → 코스 스트리밍."""
    async for evt in _stream({"message": message}):
        yield evt
