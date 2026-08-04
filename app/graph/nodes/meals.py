"""고른 끼니마다 실제 식당 후보를 붙인다.

식당은 코스 장소와 달리 검색으로 찾지 않고, 미리 구워둔 `data/meal_cache/` 에서 읽는다.
파일을 읽는 것뿐이라 네트워크를 안 탄다.

## 가까운 곳 중에서 무작위로 뽑는 이유

거리순으로 위에서 3곳을 자르면 같은 동네에서는 늘 같은 식당만 나온다.
그렇다고 동네 전체에서 무작위로 뽑으면, 동네가 넓어서(홍대·마포는 끝에서 끝까지 11km)
홍대에서 점심 먹는데 은평구 식당이 나올 수 있다.

그래서 걸어갈 만한 거리 안에서 고르되, 그 안에서는 무작위로 뽑는다.
아침은 여는 곳이 적어서 반경을 넓혀야 할 때가 많다.
"""
from __future__ import annotations

import logging
import random
import re

from app.core.geo import haversine_km, nearest_chip
from app.features.course.schema import MEAL_TIMES
from app.graph.nodes.course import _chips
from app.graph.state import AgentState
from app.tools.meal_cache import load_area_pool
from app.tools.visitseoul import KIND_BAR, KIND_CAFE, KIND_RESTAURANT

logger = logging.getLogger("lewisai.meals")

OPTIONS_PER_MEAL = 3
RADIUS_NEAR_KM = 1.5      # 걸어갈 만한 거리. 여기서 먼저 찾는다
RADIUS_FAR_KM = 3.0       # 위에서 못 채웠을 때만 넓힌다 (아침에 자주 여기까지 간다)

# 끼니마다 어떤 종류를 몇 곳씩 넣을지. 그 종류가 근처에 없으면 일반 식당으로 채운다.
RECIPE: dict[str, tuple[tuple[str, int], ...]] = {
    "아침": ((KIND_RESTAURANT, 2), (KIND_CAFE, 1)),
    "점심": ((KIND_RESTAURANT, 2), (KIND_CAFE, 1)),
    "저녁": ((KIND_RESTAURANT, 2), (KIND_BAR, 1)),
}

_HOUR = re.compile(r"(\d{1,2})\s*[:시]")


def _open_at(card: dict, hour: int) -> bool:
    """운영시간 문구를 보고 그 시각에 문을 여는지 본다. 읽어낼 수 없으면 통과시킨다."""
    hits = _HOUR.findall(card.get("use_time") or "")
    if not hits:
        return True
    o, c = int(hits[0]), int(hits[-1])
    if c <= o:
        c = 24
    return o <= hour < c


def _pick(pool: list[dict], anchor: dict, used: set[str], label: str,
          radius_km: float, rng: random.Random) -> list[dict]:
    """반경 안에서 끼니 구성(RECIPE)대로 무작위로 뽑는다. 구성이 안 되면 일반 식당으로 채운다."""
    hour = MEAL_TIMES[label]
    by_kind: dict[str, list[dict]] = {}
    for r in pool:
        if r["title"] in used or r.get("lat") is None:
            continue
        if not _open_at(r, hour):
            continue
        dist = haversine_km(anchor["lat"], anchor["lng"], r["lat"], r["lng"])
        if dist <= radius_km:
            by_kind.setdefault(r.get("kind", ""), []).append({**r, "dist_km": round(dist, 2)})

    options: list[dict] = []
    for kind, want in RECIPE.get(label, ((KIND_RESTAURANT, OPTIONS_PER_MEAL),)):
        rows = by_kind.get(kind, [])
        options += rng.sample(rows, min(want, len(rows)))
    if len(options) < OPTIONS_PER_MEAL:
        picked = {r["title"] for r in options}
        rest = [r for r in by_kind.get(KIND_RESTAURANT, []) if r["title"] not in picked]
        options += rng.sample(rest, min(OPTIONS_PER_MEAL - len(options), len(rest)))
    return sorted(options, key=lambda r: r["dist_km"])


def _options_for(pool: list[dict], anchor: dict | None, used: set[str],
                 label: str, rng: random.Random) -> list[dict]:
    """기준 장소 주변에서 식당을 찾는다. 가까운 반경부터 보고, 모자랄 때만 넓힌다."""
    if not anchor or anchor.get("lat") is None:
        return []
    options: list[dict] = []
    for radius in (RADIUS_NEAR_KM, RADIUS_FAR_KM):
        options = _pick(pool, anchor, used, label, radius, rng)
        if len(options) >= OPTIONS_PER_MEAL:
            break
    used.update(r["title"] for r in options)
    return options


async def meals_node(state: AgentState) -> dict:
    """시간표의 식사 자리에 실제 식당 후보를 채운다. 끼니를 안 골랐으면 아무것도 안 한다."""
    schedule = state.get("schedule", [])
    meal_slots = [s for s in schedule if s.get("slot_type") == "meal"]
    if not meal_slots:
        return {}

    selected = state.get("selected", [])
    if not selected:
        return {}
    _chips(state)  # 칩이 이상하면 여기서 로그가 남는다

    # 코스가 걸쳐 있는 동네의 캐시만 합친다. 이름이 겹치는 식당은 하나만 남긴다
    pool: list[dict] = []
    seen_title: set[str] = set()
    for area in {nearest_chip(s["lat"], s["lng"]) for s in selected
                 if s.get("lat") is not None}:
        for card in load_area_pool(area):
            if card["title"] not in seen_title:
                seen_title.add(card["title"])
                pool.append(card)

    rng = random.Random()          # 시드를 안 준다. 같은 요청이어도 식당은 매번 달라진다
    used: set[str] = set()         # 한 코스 안에서 같은 식당이 두 번 나오지 않게
    by_day_place = [s for s in schedule if s.get("slot_type") == "place"]
    name_to_stop = {s["name"]: s for s in selected}

    for slot in meal_slots:
        # 어디를 기준으로 찾을지 정한다. 같은 날 이 식사 바로 앞 장소가 1순위,
        # 없으면 바로 뒤 장소, 그것도 없으면 그날 아무 장소나 쓴다.
        day = slot.get("day")
        same_day = [p for p in by_day_place if p.get("day") == day]
        before = [p for p in same_day if p["start_time"] <= slot["start_time"]]
        after = [p for p in same_day if p["start_time"] > slot["start_time"]]
        cands = ([before[-1]] if before else []) + (after[:1] if after else []) + same_day
        options: list[dict] = []
        anchor_name = ""
        for c in cands:
            stop = name_to_stop.get(c["name"])
            options = _options_for(pool, stop, used, slot["label"], rng)
            if options:
                anchor_name = c["name"]
                break
        top = options[0] if options else {}
        slot["meal_options"] = options
        slot["anchor"] = anchor_name
        slot["summary"] = top.get("summary", "")
        slot["lat"], slot["lng"] = top.get("lat"), top.get("lng")
        # 3km 안에 하나도 없으면 지어내지 않고 자리만 표시한다
        slot["name"] = (f"{slot['label']} 식사 · {anchor_name} 주변" if options
                        else f"{slot['label']} 식사")

    n = sum(len(s["meal_options"]) for s in meal_slots)
    logger.info("식사 후보: 슬롯 %d개 · 식당 %d곳 (풀 %d곳)", len(meal_slots), n, len(pool))
    return {"schedule": schedule, "meal_pool": pool}
