"""meals 노드 — 고른 끼니에 실제 식당 후보를 붙인다.

식사는 목적 칩이 아니라 별도 축이므로 임베딩(places.json)이 아니라
`data/meal_cache/` 권역 캐시(1,234곳)에서 온다. 파일 읽기라 네트워크 왕복이 없다.

## 왜 거리 제약 + 랜덤인가

이전 규칙은 "앵커 1.5km 안 거리순 상위 3곳" 이었다. 같은 권역·같은 앵커면
**결정적으로 늘 같은 답**이 나와서 "항상 같은 식당만 나온다" 는 체감의 원인이었다.

그렇다고 권역 전체 랜덤은 안 된다 — 권역이 너무 넓다 (실측 중심~최원거리:
홍대·마포 11.0km · 강북·성북 9.2km · 성수·건대 8.7km). 홍대에서 점심인데
은평구 식당이 나올 수 있다.

그래서 **반경 안에서 뽑되 그 안에서는 랜덤**이다. 실측 재고(임의 장소 120곳 기준
1.5km 안 중앙값): 점심 28곳 · 저녁 28곳 · 아침 8곳. 점심·저녁은 랜덤 뽑기에
충분하고, 아침만 얇아 3km 폴백이 필요하다.
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
RADIUS_NEAR_KM = 1.5      # 걸어갈 만한 거리 — 우선
RADIUS_FAR_KM = 3.0       # 못 채울 때만 (아침은 재고가 얇아 자주 여기까지 간다)

# 끼니별 구성 — (종류, 개수). 해당 종류가 반경 안에 없으면 식당으로 메운다.
RECIPE: dict[str, tuple[tuple[str, int], ...]] = {
    "아침": ((KIND_RESTAURANT, 2), (KIND_CAFE, 1)),
    "점심": ((KIND_RESTAURANT, 2), (KIND_CAFE, 1)),
    "저녁": ((KIND_RESTAURANT, 2), (KIND_BAR, 1)),
}

_HOUR = re.compile(r"(\d{1,2})\s*[:시]")


def _open_at(card: dict, hour: int) -> bool:
    """use_time 원문으로 그 시각 영업 여부를 판정. 파싱 불가면 통과시킨다."""
    hits = _HOUR.findall(card.get("use_time") or "")
    if not hits:
        return True
    o, c = int(hits[0]), int(hits[-1])
    if c <= o:
        c = 24
    return o <= hour < c


def _pick(pool: list[dict], anchor: dict, used: set[str], label: str,
          radius_km: float, rng: random.Random) -> list[dict]:
    """반경 안에서 끼니 구성대로 랜덤 선택. 구성이 안 되면 식당으로 메운다."""
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
    """앵커 주변 후보 — 1.5km 를 먼저 보고 3곳을 못 채울 때만 3km 로 넓힌다."""
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
    """시간표의 식사 슬롯에 실제 식당 후보를 채운다. 끼니 미선택이면 즉시 반환."""
    schedule = state.get("schedule", [])
    meal_slots = [s for s in schedule if s.get("slot_type") == "meal"]
    if not meal_slots:
        return {}

    selected = state.get("selected", [])
    if not selected:
        return {}
    _chips(state)  # 칩 파싱 실패 로깅 유지

    # 코스가 걸친 권역의 캐시만 합친다 (제목 중복 제거)
    pool: list[dict] = []
    seen_title: set[str] = set()
    for area in {nearest_chip(s["lat"], s["lng"]) for s in selected
                 if s.get("lat") is not None}:
        for card in load_area_pool(area):
            if card["title"] not in seen_title:
                seen_title.add(card["title"])
                pool.append(card)

    rng = random.Random()          # 완전 랜덤 — 같은 요청도 매번 다른 식당이 나온다
    used: set[str] = set()         # 한 코스 안에서는 중복 없음
    by_day_place = [s for s in schedule if s.get("slot_type") == "place"]
    name_to_stop = {s["name"]: s for s in selected}

    for slot in meal_slots:
        # 앵커 = 같은 날, 이 식사 직전 장소. 없으면 직후 장소, 그것도 없으면 아무 장소
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
