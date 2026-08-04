"""plan 이 만든 시간 뼈대에 확정된 장소를 앉혀 시간표를 만든다.

장소를 고른 뒤, 식당을 붙이기 전에 온다. 끼니 시각은 이미 정해져 있어서
식당은 시간표가 나온 다음에 채워도 된다.

시간 범위가 없는 요청이면 아무것도 하지 않는다.
여러 날 코스는 날짜별로 나눠 하루씩 같은 방식으로 시간표를 만든다.
"""
from __future__ import annotations

import logging

from app.core.geo import chip_of
from app.core.plan import allocate_stops, build_skeleton
from app.core.scheduler import build_day, hhmm
from app.graph.nodes.course import _chips
from app.graph.state import AgentState

logger = logging.getLogger("lewisai.schedule")


def _split_by_quota(stops: list[dict], quota: list[int]) -> tuple[list[list[dict]], list[dict]]:
    """장소 목록을 구간마다 정해진 개수만큼 앞에서부터 잘라 나눈다. 남은 것도 같이 돌려준다."""
    groups: list[list[dict]] = []
    i = 0
    for n in quota:
        groups.append(stops[i:i + n])
        i += n
    return groups, stops[i:]


async def fit_schedule_node(state: AgentState) -> dict:
    """고른 장소들로 시간표를 짠다.

    시간표 순서대로 장소 목록을 다시 정렬해 내보낸다. 방문 순서는 여기서 확정된다.
    시간이 모자라 못 넣은 장소는 시각 없이 "자유 방문"으로 뺀다.
    """
    chips = _chips(state)
    selected = state.get("selected", [])
    skeleton = state.get("skeleton") or build_skeleton(chips)
    if not skeleton.get("segments") or not selected:
        return {"schedule": []}

    chip = next((chip_of(loc) for loc in chips.locations if chip_of(loc)), None)
    day_areas = state.get("day_areas") or {}
    segments = skeleton["segments"]

    schedule: list[dict] = []
    ordered: list[dict] = []
    day_numbers = sorted({s.get("day") or 1 for s in selected})

    for day_no in day_numbers:
        day_stops = [s for s in selected if (s.get("day") or 1) == day_no]
        day_field = day_no if chips.days > 1 else None
        day_chip = chip_of(day_areas.get(day_no)) or chip
        anchor = (day_chip.lat, day_chip.lng) if day_chip else None

        quota, _ = allocate_stops(skeleton, len(day_stops))
        groups, leftover = _split_by_quota(day_stops, quota)
        slots, dropped = build_day(groups, segments, anchor=anchor)

        # 시간표 슬롯 + 식사 슬롯을 시각 순으로 합친다
        rows: list[tuple[int, dict]] = []
        for slot in slots:
            stop = {**slot["stop"], "day": day_field}
            ordered.append(stop)
            rows.append((slot["start"], {
                "slot_type": "place", "name": stop["name"], "day": day_field,
                "start_time": hhmm(slot["start"]), "end_time": hhmm(slot["end"]),
                "duration_min": slot["dwell_min"],
                "travel_min": slot["travel_min"], "travel_mode": slot["travel_mode"],
            }))
        for m in skeleton["meals"]:
            rows.append((m["start"], {
                "slot_type": "meal", "label": m["label"], "day": day_field,
                "start_time": hhmm(m["start"]), "end_time": hhmm(m["end"]),
                "duration_min": m["end"] - m["start"],
                # 이름·좌표·선택지는 meals 노드가 채운다 (여기선 자리만 잡는다)
                "name": f"{m['label']} 식사", "summary": "", "lat": None, "lng": None,
                "anchor": "", "meal_options": [],
            }))
        schedule += [r for _, r in sorted(rows, key=lambda x: x[0])]

        for stop in dropped + leftover:      # 시간표에 못 앉힌 장소 — 꼬리에
            ordered.append({**stop, "day": day_field})
            schedule.append({"slot_type": "flex", "name": stop["name"], "day": day_field})

    n_meals = sum(1 for s in schedule if s["slot_type"] == "meal")
    n_flex = sum(1 for s in schedule if s["slot_type"] == "flex")
    n_place = sum(1 for s in schedule if s["slot_type"] == "place")
    logger.info("시간표 %d일: 장소 %d · 식사 %d · flex %d",
                len(day_numbers), n_place, n_meals, n_flex)
    return {"selected": ordered, "schedule": schedule}
