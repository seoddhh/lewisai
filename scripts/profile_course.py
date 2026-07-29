"""코스 생성 레이턴시 계측 — 노드별 벽시계 시간 + 임베딩/검색 호출 수.

레이턴시 최적화(docs/next-tasks.md 2-1)의 전후 비교용. 개선을 넣기 전에 이 스크립트로
기준선을 남기고, 넣은 뒤 같은 명령을 다시 돌려 비교한다.

    .venv/bin/python -m scripts.profile_course              # 기본 3개 시나리오
    .venv/bin/python -m scripts.profile_course --case local1
    .venv/bin/python -m scripts.profile_course --repeat 3   # 시나리오당 3회 평균

노드 시간은 astream(stream_mode="updates") 의 yield 간격으로 잰다. 노드가 순차 실행이라
"이전 노드 완료 → 이번 노드 완료" 간격이 곧 그 노드의 소요 시간이다. 병렬 분기를 넣은
뒤에는 이 가정이 깨지므로 합계(total)를 기준으로 볼 것.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

from app.graph.build import get_agent_graph

# ── 계측 대상 시나리오 ─────────────────────────────────────────────────────
# 실제 프론트 위저드가 보내는 칩 조합을 흉내낸다. day_areas 분기(여행자 멀티데이 +
# 위치 상관없음)를 타는 케이스와 안 타는 케이스를 모두 넣어야 병렬화 효과를 구분할 수 있다.
CASES: dict[str, dict[str, Any]] = {
    "local1": {  # 현지인 1일 — day_areas 없음. 일차 병렬화가 안 먹는 최다 케이스
        "note": "",
        "chips": {
            "audience": "local", "companions": ["연인과"], "time": "오후",
            "purposes": ["데이트", "문화·예술"], "locations": ["종로·중구"],
            "place_count": 4, "days": 1, "meals": ["점심", "저녁"],
        },
    },
    "tourist2": {  # 여행자 2일 + 위치 상관없음 — day_areas 분기
        "note": "",
        "chips": {
            "audience": "tourist", "companions": ["친구와"],
            "purposes": ["관광 명소", "핫플레이스"], "locations": ["상관없음"],
            "place_count": 4, "days": 2, "meals": ["점심", "저녁"],
        },
    },
    "tourist3": {  # 여행자 3일 — 최악 케이스 (출력 토큰이 가장 많다)
        "note": "",
        "chips": {
            "audience": "tourist", "companions": ["혼자"],
            "purposes": ["문화·예술", "자연·힐링"], "locations": ["상관없음"],
            "place_count": 5, "days": 3, "meals": ["아침", "점심", "저녁"],
        },
    },
}


class Counters:
    """전역 호출 카운터 — 임베딩 왕복이 몇 번 일어나는지가 핵심 관심사."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.embed_sec = 0.0
        self.search_calls = 0

    def reset(self) -> None:
        self.__init__()

    def line(self) -> str:
        return (f"임베딩 {self.embed_calls}회 / {self.embed_sec:.2f}s · "
                f"벡터검색 {self.search_calls}회")


COUNTERS = Counters()


def _instrument() -> None:
    """임베딩·벡터검색 호출을 감싸 카운트한다 (프로세스 전역, 1회만)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from app.rag import retriever

    orig_embed = GoogleGenerativeAIEmbeddings.embed_query

    def counted_embed(self, text: str):
        t0 = time.perf_counter()
        try:
            return orig_embed(self, text)
        finally:
            COUNTERS.embed_calls += 1
            COUNTERS.embed_sec += time.perf_counter() - t0

    GoogleGenerativeAIEmbeddings.embed_query = counted_embed

    orig_search = retriever.search_with_score

    def counted_search(*args, **kwargs):
        COUNTERS.search_calls += 1
        return orig_search(*args, **kwargs)

    retriever.search_with_score = counted_search


async def run_once(case: dict[str, Any]) -> tuple[dict[str, float], float, dict[str, Any]]:
    """1회 실행 → (노드별 초, 총 초, 최종 상태)."""
    COUNTERS.reset()
    graph = get_agent_graph()
    initial = {"intent": "course", "req": case}

    timings: dict[str, float] = {}
    state: dict[str, Any] = {}
    t0 = last = time.perf_counter()
    async for data in graph.astream(initial, stream_mode="updates"):
        now = time.perf_counter()
        for node, patch in data.items():
            timings[node] = timings.get(node, 0.0) + (now - last)
            if isinstance(patch, dict):
                state.update(patch)
        last = now
    return timings, time.perf_counter() - t0, state


def _report(name: str, timings: dict[str, float], total: float, state: dict) -> None:
    print(f"\n── {name} — 총 {total:.2f}s · {COUNTERS.line()}")
    course = (state.get("result") or {}).get("course", {})
    print(f"   후보 {len(state.get('candidates', []))}곳 · 선정 {len(state.get('selected', []))}곳"
          f" · 슬롯 {len(state.get('schedule', []))}개 · source={state.get('source')}"
          f" · title={course.get('title', '')!r}")
    # 품질 가드 — compose 를 콜 단위로 쪼갠 뒤 "빠르지만 빈 카드"가 되는 회귀를 잡는다.
    places = [s for s in course.get("stops", []) if s.get("slot_type") != "meal"]
    empty = [s["name"] for s in places if not s.get("description")]
    print(f"   카드 {len(places)}곳 중 설명 없음 {len(empty)}곳"
          f"{': ' + ', '.join(empty) if empty else ''}"
          f" · day_descriptions {len(course.get('day_descriptions') or {})}개")
    for node, sec in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        bar = "█" * max(1, round(sec / max(total, 0.01) * 40))
        print(f"   {node:<14} {sec:6.2f}s {sec / total * 100:4.1f}%  {bar}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=[*CASES, "all"], default="all")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    _instrument()
    names = list(CASES) if args.case == "all" else [args.case]

    for name in names:
        totals: list[float] = []
        agg: dict[str, list[float]] = {}
        for i in range(args.repeat):
            timings, total, state = await run_once(CASES[name])
            totals.append(total)
            for node, sec in timings.items():
                agg.setdefault(node, []).append(sec)
            _report(f"{name} #{i + 1}", timings, total, state)
        if args.repeat > 1:
            mean = {n: sum(v) / len(v) for n, v in agg.items()}
            print(f"\n== {name} 평균 {sum(totals) / len(totals):.2f}s "
                  f"(최소 {min(totals):.2f} / 최대 {max(totals):.2f})")
            for node, sec in sorted(mean.items(), key=lambda kv: kv[1], reverse=True):
                print(f"   {node:<14} {sec:6.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
