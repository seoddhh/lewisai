"""칩만 보고 하루의 시간 뼈대를 만드는 노드. 계산만 하므로 순식간에 끝난다.

장소를 찾기 전에 와야 한다. 이 뼈대를 AI 에게 같이 줘야 구간에 맞는 장소를 고르기 때문이다.

끼니를 안 골랐거나 시간 범위가 없어도 그래프가 갈라지지는 않는다.
뼈대 내용만 달라지고(구간 하나 또는 빈 뼈대), 뒤 노드들은 똑같이 동작한다.

실제 계산은 app/core/plan.py 가 한다.
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
