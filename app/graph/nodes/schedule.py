"""fit_schedule 노드 — 시간 골격(plan)에 확정 장소를 앉힌다.

select 뒤, meals 앞에 온다. 예전에는 nearby(식당 풀) 뒤였는데, 식사가 사용자 선택이 되면서
식당 후보는 시간표가 정해진 **뒤에** 붙이면 된다 (끼니 시각이 칩으로 이미 고정이므로).

시간 창이 없으면 아무것도 하지 않는다 (자연어 경로 등).

멀티데이는 장소마다 select 가 붙인 day 로 묶어 하루씩 같은 골격으로 시간표를 만든다.
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
    """장소 목록을 구간별 배정 수대로 순서대로 자른다 → (구간별 그룹, 남은 것)."""
    groups: list[list[dict]] = []
    i = 0
    for n in quota:
        groups.append(stops[i:i + n])
        i += n
    return groups, stops[i:]


async def fit_schedule_node(state: AgentState) -> dict:
    """확정 장소 → 구간 예산에 맞춘 시간표 (멀티데이는 일자별).

    - selected 를 시간표 순서로 재정렬해 내보낸다 (서버가 방문 순서를 확정한다)
    - 구간 수용 한계를 넘거나 예산이 모자란 장소는 flex(시각 없는 자유 방문 제안)
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
