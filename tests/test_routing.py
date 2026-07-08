"""app/core/routing.py 포팅 정확성 검증.

strangemap courseRouting.ts 와 동일한 결정론적 동작을 확인한다:
 - haversine 거리
 - 오픈-패스 TSP 가 지그재그(교차)를 제거하는지
 - plan_course 의 스톱 수·총거리 상한
 - 운영시간 시간창 위반 최소화
 - select_cluster 군집 선택
"""
from __future__ import annotations

import math

from app.core.routing import (
    haversine_km,
    optimize_course_order,
    plan_course,
    select_cluster,
)


def _pt(lat, lng, **extra):
    return {"lat": lat, "lng": lng, **extra}


def test_haversine_known_distance():
    # 서울시청(37.5665,126.9780) ↔ 남산공원(37.5512,126.9882) ≈ 1.9km
    d = haversine_km(37.5665, 126.9780, 37.5512, 126.9882)
    assert 1.5 < d < 2.3, d
    assert haversine_km(37.5, 127.0, 37.5, 127.0) == 0.0


def test_open_path_removes_zigzag():
    # 경도로 일렬 배치된 5개 점을 뒤섞어 입력 → 최적 순서는 단조(좌→우 또는 우→좌)여야 한다.
    stops = [
        _pt(37.55, 126.90, name="A"),
        _pt(37.55, 126.98, name="E"),
        _pt(37.55, 126.92, name="B"),
        _pt(37.55, 126.96, name="D"),
        _pt(37.55, 126.94, name="C"),
    ]
    res = optimize_course_order(stops)
    names = [s["name"] for s in res.ordered]
    assert names in (["A", "B", "C", "D", "E"], ["E", "D", "C", "B", "A"]), names
    assert res.method == "exact"
    # 단조 경로 총거리 ≈ 양 끝(A↔E) 직선거리 (중간점들이 직선 위)
    end_to_end = haversine_km(37.55, 126.90, 37.55, 126.98)
    assert math.isclose(res.total_km, end_to_end, rel_tol=1e-6), (res.total_km, end_to_end)


def test_plan_course_caps_max_stops():
    # 7곳 입력, maxStops=5 → 5곳으로 축소
    stops = [_pt(37.55, 126.90 + i * 0.01, name=f"P{i}") for i in range(7)]
    res = plan_course(stops, max_stops=5, min_stops=3, max_total_km=100)
    assert len(res.ordered) == 5


def test_plan_course_caps_total_km():
    # 아주 멀리 떨어진 점들 → 총거리 상한(3km)에 걸려 min_stops(3)까지 축소
    stops = [_pt(37.50 + i * 0.05, 126.90 + i * 0.05, name=f"P{i}") for i in range(6)]
    res = plan_course(stops, max_stops=6, min_stops=3, max_total_km=3.0)
    assert len(res.ordered) == 3


def test_time_window_prefers_open_places():
    # 두 점 A(상시), B(9~18시). start_hour=20 이면 B 방문 시각이 영업 종료 후 → 위반 1.
    # 순서를 바꿔도 B는 위반이지만, 위반 최소가 거리보다 우선함을 확인(2점이라 위반 그대로).
    stops = [
        _pt(37.55, 126.95, name="A", operating_hours={"start": 0, "end": 24}),
        _pt(37.56, 126.96, name="B", operating_hours={"start": 9, "end": 18}),
        _pt(37.57, 126.97, name="C", operating_hours={"start": 9, "end": 18}),
    ]
    late = optimize_course_order(stops, start_hour=20.0)
    early = optimize_course_order(stops, start_hour=10.0)
    # 이른 시작(10시)은 영업시간 내 방문 가능 → 위반 0
    assert early.violations == 0
    # 늦은 시작(20시)은 9~18시 장소 방문 불가 → 위반 발생
    assert late.violations >= 1


def test_select_cluster_picks_dense_group():
    # 강북 군집 3곳 + 강남 외딴 2곳. 카테고리 가중 없이도 밀집 군집(강북)이 선택돼야 한다.
    cands = [
        _pt(37.58, 126.97, category="관광·역사"),
        _pt(37.579, 126.975, category="관광·역사"),
        _pt(37.577, 126.972, category="관광·역사"),
        _pt(37.49, 127.03, category="상권·역세권"),
        _pt(37.50, 127.04, category="상권·역세권"),
    ]
    picked = select_cluster(cands, {"관광·역사": 2.0}, radius_km=3.0, skip_below=0)
    # 선택된 군집은 모두 강북(위도>37.55) 밀집 지역
    assert all(p["lat"] > 37.55 for p in picked), picked
    assert len(picked) == 3


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
