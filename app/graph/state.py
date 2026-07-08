"""AgentState — LangGraph 노드 간 공유 상태."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # 진입
    message: str            # 자연어 입력 (/agent/chat)
    intent: str             # place_intro | recommend | course | chitchat
    req: dict[str, Any]     # 구조화된 요청 (parse_intent 결과 또는 어댑터 주입)

    # course 파이프라인 중간 상태
    candidates: list[dict]  # RAG 후보 장소 (정규화된 dict[])
    selected: list[dict]    # LLM 이 고른 장소 dict[] (name/lat/lng/operating_hours/category)
    congestion: dict[str, str]  # name → 혼잡도 레벨
    ordered: list[dict]     # plan_course 로 최적화된 stop dict[]
    total_km: float         # 최적화 동선 총거리(km)

    # 출력
    result: dict[str, Any]  # 최종 응답 payload (기능별 스키마로 직렬화 가능)
    source: str             # ai | mock
    error: str
