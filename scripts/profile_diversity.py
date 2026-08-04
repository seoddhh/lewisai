"""코스 다양성·개인화 계측 — 개선 전후를 숫자로 비교하기 위한 기준선.

    uv run python -m scripts.profile_diversity --stage retrieve   # LLM 없이 후보만 (빠름·무료)
    uv run python -m scripts.profile_diversity --stage full       # 코스 생성까지 (LLM 콜)

`scripts/profile_course.py` 는 레이턴시를 재고, 이 스크립트는 **결과의 질**을 잰다.
지금까지 "다양성이 좋아졌다"를 판단할 수단이 없어서 프롬프트에 규칙만 쌓였다.

## 재는 것

| 지표 | 뜻 | 나쁜 값 |
|---|---|---|
| 종류 종수 | 코스 하나에 들어간 서로 다른 `category` 수 | 1~2 (같은 종류만) |
| 재생성 중복률 | 같은 칩으로 seed 만 바꿔 두 번 만들었을 때 겹치는 장소 비율 | 100% (재생성이 무의미) |
| 동반자 차이율 | 동반자만 바꿨을 때 달라진 장소 비율 | 0% (개인화가 문구뿐) |
| 목적 차이율 | 목적만 바꿨을 때 달라진 장소 비율 | 0% (목적 필터가 무력) |
| 권역 적중률 | 요청한 권역 안에 있는 장소 비율 | 낮음 (필터가 풀려 딴 동네가 섞임) |

`--stage retrieve` 는 `retrieve_node` 까지만 돌려 후보를 본다. LLM 콜이 없어 몇 초면 끝나고
쿼터를 안 쓰므로, 검색 단계 개선(쿼터·시드·가중치)의 효과는 이걸로 반복해서 보면 된다.
`select` 이후의 다양성 교정까지 보려면 `--stage full` 을 쓴다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

from app.graph.nodes.course import retrieve_node, select_places_node
from app.graph.nodes.planning import plan_node

# 계측 시나리오 — 프론트 위저드가 실제로 보내는 조합. 동반자/목적만 다른 쌍을 넣어
# "축 하나를 바꿨을 때 결과가 달라지는가"를 직접 비교할 수 있게 했다.
BASE = {"audience": "local", "locations": ["종로·중구"], "time": "오후",
        "congestion": "상관없음", "pace": "relaxed"}
CASES: dict[str, dict] = {
    "연인_문화": {**BASE, "companions": ["연인과"], "purposes": ["문화·예술"]},
    "아이_문화": {**BASE, "companions": ["아이와"], "purposes": ["문화·예술"]},
    "부모님_문화": {**BASE, "companions": ["부모님과"], "purposes": ["문화·예술"]},
    "연인_자연": {**BASE, "companions": ["연인과"], "purposes": ["자연·힐링"]},
    "연인_쇼핑": {**BASE, "companions": ["연인과"], "purposes": ["쇼핑"]},
    "친구_핫플_성수": {**BASE, "locations": ["성수·건대"],
                  "companions": ["친구와"], "purposes": ["핫플레이스"]},
    "혼자_문화_관악": {**BASE, "locations": ["관악·사당"],
                  "companions": ["혼자"], "purposes": ["문화·예술"]},
}


async def _names(chips: dict, *, seed: int | None, stage: str) -> list[dict]:
    """한 요청 → 장소 dict 목록. stage=retrieve 면 후보, full 이면 확정 스톱."""
    req = {"chips": chips, "note": ""}
    if seed is not None:
        req["seed"] = seed
    state: dict = {"req": req}
    state |= await plan_node(state)
    state |= await retrieve_node(state)
    if stage == "retrieve":
        # 후보 상위 N — 실제로 코스에 들어갈 만큼만 본다
        per = 3 if chips.get("pace") == "relaxed" else 5
        return state.get("candidates", [])[:per]
    state |= await select_places_node(state)
    return state.get("selected", [])


def _overlap(a: list[dict], b: list[dict]) -> float:
    """두 결과의 장소 겹침 비율 (0~1). 분모는 작은 쪽."""
    na, nb = {p["name"] for p in a}, {p["name"] for p in b}
    return len(na & nb) / max(1, min(len(na), len(nb)))


def _index_warning() -> None:
    """인덱스가 현재 스키마보다 오래됐는지 알린다.

    메타데이터는 **인제스트 시점에 굳는다.** 파일을 v2 로 바꿔도 재인제스트 전에는
    옛 메타(27종 category·개인화 축 없음)를 읽으므로, 여기 수치가 왜 안 변하는지
    한참 헤매게 된다. 그 시간을 없애려고 한 줄 찍는다.
    """
    from app.core.vectorstore import get_vectorstore
    try:
        sample = get_vectorstore().get(limit=1, include=["metadatas"])
        meta = (sample.get("metadatas") or [{}])[0] or {}
    except Exception as err:  # noqa: BLE001 — 인덱스가 없어도 측정 자체는 진행
        print(f"! 인덱스를 읽지 못했다({err})\n")
        return
    stale = []
    if "coarse_category" in meta:
        stale.append("27종 category (v1)")
    if "energy" not in meta:
        stale.append("개인화 축 없음")
    if stale:
        print(f"! 인덱스가 옛 스키마다 — {' · '.join(stale)}. "
              "재인제스트 전까지 아래 수치는 v1 기준이다.\n")


async def run(stage: str) -> int:
    print(f"stage={stage} · 시나리오 {len(CASES)}개\n")
    _index_warning()
    results: dict[str, list[dict]] = {}

    print("── 코스별 구성 ──")
    for label, chips in CASES.items():
        stops = await _names(chips, seed=1, stage=stage)
        results[label] = stops
        cats = Counter(p.get("category") or "?" for p in stops)
        area_hit = sum(1 for p in stops if p.get("area") == chips["locations"][0])
        indoor = sum(1 for p in stops if p.get("indoor"))
        print(f"{label:14s} 종류 {len(cats)}종 {dict(cats)} · "
              f"권역적중 {area_hit}/{len(stops)} · 실내 {indoor}/{len(stops)}")
        print(f"{'':14s} {[p['name'] for p in stops]}")

    print("\n── 재생성 중복률 (같은 칩, seed 만 다름 — 100%면 재생성이 무의미) ──")
    for label, chips in list(CASES.items())[:3]:
        a = await _names(chips, seed=1, stage=stage)
        b = await _names(chips, seed=999, stage=stage)
        print(f"{label:14s} {_overlap(a, b):.0%}   {sorted({p['name'] for p in a} ^ {p['name'] for p in b})[:4]}")

    print("\n── 축 하나만 바꿨을 때 차이율 (0%면 그 축이 검색에 반영 안 됨) ──")
    pairs = [("동반자 연인→아이", "연인_문화", "아이_문화"),
             ("동반자 연인→부모님", "연인_문화", "부모님_문화"),
             ("목적 문화→자연", "연인_문화", "연인_자연"),
             ("목적 문화→쇼핑", "연인_문화", "연인_쇼핑")]
    for name, x, y in pairs:
        print(f"{name:18s} 차이 {1 - _overlap(results[x], results[y]):.0%}")

    thin = [lb for lb, s in results.items() if len({p.get('category') for p in s}) < 2]
    if thin:
        print(f"\n경고: 종류가 1종뿐인 코스 — {thin}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("retrieve", "full"), default="retrieve",
                    help="retrieve=후보만(LLM 없음·빠름) / full=코스 생성까지")
    raise SystemExit(asyncio.run(run(ap.parse_args().stage)))
