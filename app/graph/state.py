"""AgentState — LangGraph 노드 간 공유 상태."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # 진입
    message: str            # 자연어 입력 (/agent/chat)
    intent: str             # course | chitchat
    req: dict[str, Any]     # 구조화된 요청 (parse_intent 결과 또는 어댑터 주입)

    # course 파이프라인 중간 상태
    chips: dict[str, Any]       # 경로 생성 칩 (CourseChips)
    # 시간 골격 — 칩만으로 만든 하루의 뼈대 (식사 앵커 + 장소 구간). plan 노드 산출물.
    # retrieve 보다 먼저 확정되므로 select 프롬프트에 실어 LLM 이 구간을 알고 고르게 한다.
    skeleton: dict[str, Any]
    candidates: list[dict]      # RAG 후보 장소 (정규화된 dict[], 권역 분산 시 day_hint 포함)
    day_areas: dict[int, str]   # 일차 → 권역 (여행자 멀티데이 + 위치 상관없음일 때만)
    selected: list[dict]        # AI 가 고른 장소 dict[] (+reason/activities)
    congestion: dict[str, str]  # name → 방문 시각 예상 혼잡도 레벨(예보)
    nearby: dict[str, dict]     # name → {restaurants:[], attractions:[]} (Visit Seoul)
    meal_pool: list[dict]       # 식사 후보 풀 (meal_cache 권역 캐시 — meals 노드가 채움)
    schedule: list[dict]        # 시간표 슬롯 (place|meal|flex) — 시간 범위 요청일 때만 채워짐
    # 폴리라인·실경로 거리는 서버가 만들지 않는다 (strangemap courseRouting.ts 담당).
    # 단, 시간 범위 요청(schedule)에선 방문 순서·시각을 서버가 확정한다.

    # 출력
    result: dict[str, Any]  # 최종 응답 payload (기능별 스키마로 직렬화 가능)
    source: str             # ai | mock
    error: str
