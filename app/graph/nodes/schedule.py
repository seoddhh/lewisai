"""schedule 노드 — 시간 범위가 있는 요청을 (일자별) 시간표 코스로 만든다.

nearby 뒤, compose 앞에 온다: 식사 슬롯이 nearby 의 식당 풀(meal_pool)을 쓰기 때문.
시간 범위가 없으면 아무것도 하지 않는다 (기존 응답과 완전히 동일).

멀티데이(여행자): 장소마다 select 가 붙인 day 로 묶어 하루씩 같은 시간 창으로
시간표를 만든다. 식사 슬롯에는 앵커 장소 3km 이내 Visit Seoul 식당을 3곳 이상
(가까운 순, 코스 전체에서 중복 없이) meal_options 로 싣는다 — 지어내지 않는다.
"""
from __future__ import annotations

import logging

from app.core.geo import chip_of, haversine_km
from app.core.scheduler import build_schedule, hhmm
from app.graph.nodes.course import MEAL_RADIUS_KM, MEAL_RADIUS_NEAR_KM, _chips
from app.graph.state import AgentState
from app.tools.visitseoul import KIND_BAR, KIND_CAFE, KIND_RESTAURANT

logger = logging.getLogger("lewisai.schedule")

_MEAL_OPTIONS_MAX = 3  # 끼니당 노출 상한 — 더 보여주면 뒤 끼니가 쓸 풀이 마른다 (중복 금지)

# 끼니별 구성 — (종류, 개수) 순서대로 채운다. 해당 종류가 반경 안에 없으면
# 남는 자리는 식당으로 메운다 (주점은 서울 전체 63곳뿐이라 자주 빈다).
_MEAL_RECIPE: dict[str, tuple[tuple[str, int], ...]] = {
    "점심": ((KIND_RESTAURANT, 2), (KIND_CAFE, 1)),
    "저녁": ((KIND_RESTAURANT, 2), (KIND_BAR, 1)),
}


def _pick(
    meal_pool: list[dict], anchor: dict, used: set[str], label: str, radius_km: float,
) -> list[dict]:
    """반경 안에서 끼니 구성대로 고른다. 구성이 안 되면 부족분을 식당으로 메운다."""
    by_kind: dict[str, list[dict]] = {}
    for r in meal_pool:
        if r["title"] in used or r.get("lat") is None:
            continue
        dist = haversine_km(anchor["lat"], anchor["lng"], r["lat"], r["lng"])
        if dist <= radius_km:
            by_kind.setdefault(r.get("kind", ""), []).append(
                {**r, "dist_km": round(dist, 2)}
            )
    for rows in by_kind.values():
        rows.sort(key=lambda r: r["dist_km"])

    recipe = _MEAL_RECIPE.get(label) or ((KIND_RESTAURANT, _MEAL_OPTIONS_MAX),)
    options: list[dict] = []
    for kind, want in recipe:
        options += by_kind.get(kind, [])[:want]
    # 부족분은 식당으로 (구성에서 이미 쓴 식당 다음 순번부터)
    if len(options) < _MEAL_OPTIONS_MAX:
        picked = {r["title"] for r in options}
        options += [
            r for r in by_kind.get(KIND_RESTAURANT, []) if r["title"] not in picked
        ][: _MEAL_OPTIONS_MAX - len(options)]
    return options


def _meal_options(
    meal_pool: list[dict], anchor: dict | None, used: set[str], label: str = "",
) -> list[dict]:
    """앵커 장소 주변을 끼니 구성대로 — 이미 다른 끼니에 쓴 곳은 제외.

    반경은 걸어갈 만한 1.5km 를 먼저 보고, 3곳을 못 채울 때만 3km 로 넓힌다.
    """
    if not anchor or anchor.get("lat") is None:
        return []
    options: list[dict] = []
    for radius in (MEAL_RADIUS_NEAR_KM, MEAL_RADIUS_KM):
        options = _pick(meal_pool, anchor, used, label, radius)
        if len(options) >= _MEAL_OPTIONS_MAX:
            break
    used.update(r["title"] for r in options)
    return options


async def schedule_node(state: AgentState) -> dict:
    """확정 장소 → 방문 순서·시각·식사 슬롯이 계산된 시간표 (멀티데이는 일자별).

    - 요청 시간대에 아예 운영하지 않는 장소는 문 여는 후보로 교체한다.
    - selected 를 시간표 순서로 재정렬해 내보낸다 (scheduled 코스는 서버 순서가 확정본).
    """
    chips = _chips(state)
    window = chips.resolved_window()
    selected = state.get("selected", [])
    if not window or not selected:
        return {"schedule": []}

    # 시간대에 문 닫는 장소 교체 — 후보(이미 문 여는 곳 우선 정렬)에서 채운다
    chosen = {s["name"] for s in selected}
    open_left = [
        c for c in state.get("candidates", [])
        if c["name"] not in chosen
        and window.overlaps(c.get("op_start", 0), c.get("op_end", 24))
    ]
    fixed: list[dict] = []
    for s in selected:
        if not window.overlaps(s.get("op_start", 0), s.get("op_end", 24)) and open_left:
            # 일차별 권역 분산이면 같은 일차(권역) 후보를 우선 꺼낸다
            i = next((i for i, c in enumerate(open_left)
                      if c.get("day_hint") in (None, s.get("day") or 1)), 0)
            repl = open_left.pop(i)
            fixed.append({**repl, "day": s.get("day"),
                          "reason": f"{s['name']}은(는) 요청 시간대({window.label()})에 "
                                    "운영하지 않아 대신 골랐어요.",
                          "activities": repl.get("activities", [])})
        else:
            fixed.append(s)

    chip = next((chip_of(loc) for loc in chips.locations if chip_of(loc)), None)
    day_areas = state.get("day_areas") or {}
    meal_pool = state.get("meal_pool", [])
    used_titles: set[str] = set()

    schedule: list[dict] = []
    ordered: list[dict] = []
    day_numbers = sorted({s.get("day") or 1 for s in fixed})
    for day_no in day_numbers:
        day_stops = [s for s in fixed if (s.get("day") or 1) == day_no]
        by_name = {s["name"]: s for s in day_stops}
        day_field = day_no if chips.days > 1 else None
        # 일자별 권역 분산이면 그 날의 권역 좌표가 출발 앵커
        day_chip = chip_of(day_areas.get(day_no)) or chip
        anchor = (day_chip.lat, day_chip.lng) if day_chip else None
        slots, flex = build_schedule(day_stops, window.start, window.end, anchor=anchor)

        for slot in slots:
            if slot["slot_type"] == "meal":
                # 앵커 주변이 비면 같은 날의 다른 스톱 앵커로 폴백 (3km 원칙은 유지)
                anchor_name, options = slot["anchor"], []
                for cand in ([by_name.get(slot["anchor"])]
                             + [s for s in reversed(day_stops) if s["name"] != slot["anchor"]]):
                    options = _meal_options(meal_pool, cand, used_titles, slot["label"])
                    if options:
                        anchor_name = cand["name"]
                        break
                top = options[0] if options else {}
                schedule.append({
                    "slot_type": "meal", "label": slot["label"], "day": day_field,
                    "start_time": hhmm(slot["start"]), "end_time": hhmm(slot["end"]),
                    # 3km 안 식당이 하나도 없으면 지어내지 않고 자리만 표시한다
                    "name": (f"{slot['label']} 식사 · {anchor_name} 주변" if options
                             else f"{slot['anchor']} 주변 식당에서 {slot['label']}"),
                    "summary": top.get("summary", ""),
                    "lat": top.get("lat"), "lng": top.get("lng"),
                    "anchor": slot["anchor"],
                    "meal_options": options,
                })
            else:
                stop = {**slot["stop"], "day": day_field}
                ordered.append(stop)
                schedule.append({
                    "slot_type": "place", "name": stop["name"], "day": day_field,
                    "start_time": hhmm(slot["start"]), "end_time": hhmm(slot["end"]),
                    "duration_min": slot["end"] - slot["start"],
                    "travel_min": slot.get("travel_min", 0),
                    "travel_mode": slot.get("travel_mode", ""),
                })

        for stop in flex:  # 시간표에 못 앉힌 장소 — 시간 없는 자유 방문 제안으로 꼬리에
            ordered.append({**stop, "day": day_field})
            schedule.append({"slot_type": "flex", "name": stop["name"], "day": day_field})

    n_meals = sum(1 for s in schedule if s["slot_type"] == "meal")
    n_flex = sum(1 for s in schedule if s["slot_type"] == "flex")
    logger.info("시간표 %s×%d일: 장소 %d · 식사 %d · flex %d",
                window.label(), len(day_numbers), len(ordered) - n_flex, n_meals, n_flex)
    return {"selected": ordered, "schedule": schedule}
