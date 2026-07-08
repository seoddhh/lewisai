"""코스 동선 최적화 모듈 — strangemap `src/lib/courseRouting.ts` 의 파이썬 이식.

역할 분리 원칙:
 - "어떤 장소를 고를까"(의미 매칭)는 AI(Gemini)가 담당하고,
 - "어떤 순서로 이을까"(기하 최적화)는 이 모듈이 결정적으로 푼다.
 AI가 돌려준 순서를 그대로 쓰지 않으므로 지그재그 동선이 구조적으로 사라진다.

알고리즘 선택(스톱 수 n 기준):
 - n ≤ 8   : 전수 순열 탐색 — 운영시간 제약까지 반영한 "제약 포함 정확해".
 - n ≤ 13  : Held-Karp DP — 거리 기준 정확해 + 시간창은 방향(정/역) 선택으로 보정.
 - n > 13  : 최근접 이웃 + 2-opt 휴리스틱.

거리 모델: 직선(haversine) × detourFactor(기본 1.3)로 실보행 거리를 근사.
순수 표준 라이브러리만 사용(외부 의존성 없음).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Any

# stop 은 dict: {"lat": float, "lng": float, "operating_hours": {"start": int, "end": int} | None, ...}
Stop = dict[str, Any]

EXACT_MAX = 8
HELD_KARP_MAX = 13


@dataclass
class OptimizeResult:
    ordered: list[Stop]
    total_km: float  # 직선거리 합(km, 보정 전) — UI 표기·상한 판정 기준
    violations: int  # 운영시간 위반 스톱 수 (start_hour 미지정 시 0)
    method: str  # "exact" | "held-karp" | "2-opt"


# ── 거리 ────────────────────────────────────────────────────────────────────

def haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    R = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lng = math.radians(b_lng - a_lng)
    x = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(d_lng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def build_distance_matrix(stops: list[Stop]) -> list[list[float]]:
    """대칭 거리 행렬 (n×n, km)."""
    n = len(stops)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(stops[i]["lat"], stops[i]["lng"], stops[j]["lat"], stops[j]["lng"])
            m[i][j] = d
            m[j][i] = d
    return m


def _path_km(order: list[int], m: list[list[float]]) -> float:
    return sum(m[order[i - 1]][order[i]] for i in range(1, len(order)))


# ── 시간창(운영시간) 평가 ────────────────────────────────────────────────────

def count_time_violations(
    order: list[int],
    stops: list[Stop],
    m: list[list[float]],
    start_hour: float,
    dwell_hours: float,
    walk_speed_kmh: float,
    detour_factor: float,
) -> int:
    """순서대로 방문할 때 각 스톱 도착 시각을 추정해 운영시간 위반 수를 센다.

    도착 추정: start_hour + Σ(구간 보행시간) + 방문한 스톱 수 × dwell.
    자정을 넘기는 도착은 상시(0~24) 장소가 아니면 위반으로 본다.
    """
    violations = 0
    clock = start_hour
    for i in range(len(order)):
        if i > 0:
            clock += (m[order[i - 1]][order[i]] * detour_factor) / walk_speed_kmh
        hours = stops[order[i]].get("operating_hours")
        if hours and not (hours.get("start") == 0 and hours.get("end") == 24):
            if clock >= 24 or clock < hours["start"] or clock >= hours["end"]:
                violations += 1
        clock += dwell_hours
    return violations


# ── 정확해 1: 전수 순열 (n ≤ 8) — 시간창 제약 포함 ──────────────────────────

def _exact_open_path(
    stops: list[Stop],
    m: list[list[float]],
    time_opts: dict | None,
) -> tuple[list[int], float, int]:
    n = len(stops)
    best: tuple[list[int], float, int] | None = None
    for order in permutations(range(n)):
        order = list(order)
        # 시간창이 없으면 정/역방향 거리가 같으므로 절반만 평가
        if time_opts is None and order[0] > order[n - 1]:
            continue
        km = _path_km(order, m)
        violations = (
            count_time_violations(order, stops, m, **time_opts) if time_opts is not None else 0
        )
        if (
            best is None
            or violations < best[2]
            or (violations == best[2] and km < best[1])
        ):
            best = (order, km, violations)
    assert best is not None
    return best


# ── 정확해 2: Held-Karp 오픈 패스 (n ≤ 13) — 거리 기준 ──────────────────────

def _held_karp_open_path(m: list[list[float]]) -> list[int]:
    n = len(m)
    full = 1 << n
    inf = float("inf")
    dp = [[inf] * n for _ in range(full)]
    parent = [[-1] * n for _ in range(full)]
    for i in range(n):
        dp[1 << i][i] = 0.0  # 시작점 자유

    for mask in range(1, full):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == inf:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                nm = mask | (1 << nxt)
                nd = dp[mask][last] + m[last][nxt]
                if nd < dp[nm][nxt]:
                    dp[nm][nxt] = nd
                    parent[nm][nxt] = last

    best_end, best_km = 0, inf
    for last in range(n):
        if dp[full - 1][last] < best_km:
            best_km = dp[full - 1][last]
            best_end = last

    order: list[int] = []
    mask, cur = full - 1, best_end
    while cur != -1:
        order.insert(0, cur)
        p = parent[mask][cur]
        mask &= ~(1 << cur)
        cur = p
    return order


# ── 휴리스틱: 최근접 이웃 + 2-opt (n > 13) ───────────────────────────────────

def _nearest_neighbor_order(m: list[list[float]], start: int) -> list[int]:
    n = len(m)
    visited = [False] * n
    order = [start]
    visited[start] = True
    while len(order) < n:
        last = order[-1]
        best, best_d = -1, float("inf")
        for i in range(n):
            if not visited[i] and m[last][i] < best_d:
                best_d, best = m[last][i], i
        order.append(best)
        visited[best] = True
    return order


def _two_opt_improve(order: list[int], m: list[list[float]]) -> list[int]:
    """2-opt: 경로 교차를 제거할 때까지 구간 반전을 반복 (오픈 패스 버전)."""
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                a = 0.0 if i == 0 else m[order[i - 1]][order[i]]
                b = m[order[j]][order[j + 1]]
                c = 0.0 if i == 0 else m[order[i - 1]][order[j]]
                d = m[order[i]][order[j + 1]]
                if c + d < a + b - 1e-9:
                    order[i : j + 1] = order[i : j + 1][::-1]
                    improved = True
    return order


# ── 공개 API ─────────────────────────────────────────────────────────────────

def _time_opts(
    start_hour: float | None, dwell_hours: float, walk_speed_kmh: float, detour_factor: float
) -> dict | None:
    if start_hour is None:
        return None
    return {
        "start_hour": start_hour,
        "dwell_hours": dwell_hours,
        "walk_speed_kmh": walk_speed_kmh,
        "detour_factor": detour_factor,
    }


def optimize_course_order(
    stops: list[Stop],
    start_hour: float | None = None,
    dwell_hours: float = 1.0,
    walk_speed_kmh: float = 4.0,
    detour_factor: float = 1.3,
) -> OptimizeResult:
    """오픈 패스(출발지 복귀 없음) 방문 순서 최적화.

    스톱 수에 따라 정확해/휴리스틱을 자동 선택하고,
    start_hour 가 있으면 운영시간 위반 최소화를 거리보다 우선한다.
    """
    if len(stops) <= 2:
        km = (
            haversine_km(stops[0]["lat"], stops[0]["lng"], stops[1]["lat"], stops[1]["lng"])
            if len(stops) == 2
            else 0.0
        )
        return OptimizeResult(ordered=list(stops), total_km=km, violations=0, method="exact")

    m = build_distance_matrix(stops)
    topts = _time_opts(start_hour, dwell_hours, walk_speed_kmh, detour_factor)

    if len(stops) <= EXACT_MAX:
        order, km, violations = _exact_open_path(stops, m, topts)
        return OptimizeResult(
            ordered=[stops[i] for i in order], total_km=km, violations=violations, method="exact"
        )

    if len(stops) <= HELD_KARP_MAX:
        order = _held_karp_open_path(m)
        method = "held-karp"
    else:
        order = _two_opt_improve(_nearest_neighbor_order(m, 0), m)
        method = "2-opt"

    # 시간창이 있으면 정방향/역방향 중 위반이 적은 쪽 선택 (거리는 동일)
    violations = 0
    if topts is not None:
        fwd = count_time_violations(order, stops, m, **topts)
        rev = count_time_violations(list(reversed(order)), stops, m, **topts)
        if rev < fwd:
            order = list(reversed(order))
        violations = min(fwd, rev)
    return OptimizeResult(
        ordered=[stops[i] for i in order], total_km=_path_km(order, m), violations=violations, method=method
    )


def _remove_cheapest_stop(ordered: list[Stop]) -> list[Stop]:
    """최적화된 순서에서 "빼면 경로가 가장 짧아지는" 스톱 하나를 제거."""
    n = len(ordered)
    best_idx, best_saving = 0, float("-inf")
    for i in range(n):
        prev = ordered[i - 1] if i - 1 >= 0 else None
        nxt = ordered[i + 1] if i + 1 < n else None
        in_cost = (
            (haversine_km(prev["lat"], prev["lng"], ordered[i]["lat"], ordered[i]["lng"]) if prev else 0.0)
            + (haversine_km(ordered[i]["lat"], ordered[i]["lng"], nxt["lat"], nxt["lng"]) if nxt else 0.0)
        )
        bridge = haversine_km(prev["lat"], prev["lng"], nxt["lat"], nxt["lng"]) if prev and nxt else 0.0
        saving = in_cost - bridge
        if saving > best_saving:
            best_saving, best_idx = saving, i
    return [s for i, s in enumerate(ordered) if i != best_idx]


def plan_course(
    stops: list[Stop],
    max_stops: int = 5,
    min_stops: int = 3,
    max_total_km: float = 10.0,
    start_hour: float | None = None,
    dwell_hours: float = 1.0,
    walk_speed_kmh: float = 4.0,
    detour_factor: float = 1.3,
) -> OptimizeResult:
    """코스 계획: 순서 최적화 → 총거리 상한 초과 시 "빼면 가장 절약되는" 스톱부터
    제거하고 재최적화. min_stops 까지 줄여도 초과하면 그 상태로 반환한다(빈손 방지).
    """

    def _opt(cur: list[Stop]) -> OptimizeResult:
        return optimize_course_order(cur, start_hour, dwell_hours, walk_speed_kmh, detour_factor)

    result = _opt(list(stops))

    # 스톱 수 상한: 초과분은 제거 시 경로 절약이 가장 큰 것부터 뺀다
    while len(result.ordered) > max_stops:
        result = _opt(_remove_cheapest_stop(result.ordered))

    # 총거리 상한
    while result.total_km > max_total_km and len(result.ordered) > min_stops:
        result = _opt(_remove_cheapest_stop(result.ordered))
    return result


# ── 후보 군집 선택 (ai-recommend 사전 거리 반영) ─────────────────────────────

def select_cluster(
    cands: list[Stop],
    category_weight: dict[str, float],
    radius_km: float = 3.0,
    max_count: int = 12,
    skip_below: int = 5,
) -> list[Stop]:
    """가까운 장소끼리 군집을 만들어, 목적 적합도가 가장 높은 한 군집만 후보로 선택한다.

    → AI가 고른 스톱들이 흩어지지 않고 자연스러운 한 동선으로 이어지도록 "거리"를 사전 반영.
    각 후보 dict 는 category/lat/lng 를 가져야 한다.
    """
    if len(cands) <= skip_below:
        return cands

    def w_score(c: Stop) -> float:
        return category_weight.get(c.get("category", ""), 0.0)

    best: tuple[list[Stop], Stop, float] | None = None
    for anchor in cands:
        members = [
            c for c in cands
            if haversine_km(anchor["lat"], anchor["lng"], c["lat"], c["lng"]) <= radius_km
        ]
        score = len(members) + sum(w_score(c) for c in members) + w_score(anchor) * 2
        if best is None or score > best[2]:
            best = (members, anchor, score)

    members, anchor, _ = best  # type: ignore[misc]
    # 목적 적합 우선 → 앵커 근접 우선으로 정렬 후 상한 적용
    return sorted(
        members,
        key=lambda c: (
            -w_score(c),
            haversine_km(anchor["lat"], anchor["lng"], c["lat"], c["lng"]),
        ),
    )[:max_count]
