"""plan 노드 — 칩만으로 하루의 시간 골격을 만든다. 순수 계산, 0ms.

파이프라인 맨 앞(retrieve 앞)에 온다. 식사 시각이 칩으로 확정되므로 장소를 고르기
전에 하루가 어떻게 나뉘는지 알 수 있고, 그 골격을 select 프롬프트에 실어 보내면
LLM 이 구간을 알고 장소·체류시간을 고른다.

끼니를 안 골랐거나 시간 창이 없어도 **그래프를 분기하지 않는다** — 골격의 내용만
달라진다 (세그먼트 1개 / 빈 골격). 하위 노드는 "세그먼트 리스트" 하나로 동일하게 동작한다.
"""
from __future__ import annotations

import logging

from app.core.plan import allocate_stops, build_skeleton
from app.graph.nodes.course import _chips
from app.graph.state import AgentState

logger = logging.getLogger("lewisai.plan")


async def plan_node(state: AgentState) -> dict:
    chips = _chips(state)
    skeleton = build_skeleton(chips)
    quota, flex_n = allocate_stops(skeleton, chips.stops_per_day())
    logger.info(
        "시간 골격: 창=%s 끼니=%d 구간=%d 배정=%s flex=%d",
        skeleton["window"], len(skeleton["meals"]), len(skeleton["segments"]), quota, flex_n,
    )
    return {"skeleton": skeleton}
