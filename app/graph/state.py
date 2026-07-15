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
    candidates: list[dict]      # RAG 후보 장소 (정규화된 dict[])
    selected: list[dict]        # AI 가 고른 장소 dict[] (+reason/activities)
    congestion: dict[str, str]  # name → 실시간 혼잡도 레벨
    nearby: dict[str, dict]     # name → {restaurants:[], attractions:[]} (Visit Seoul)
    # 방문 순서·폴리라인·거리는 서버가 만들지 않는다 (strangemap courseRouting.ts 담당)

    # 출력
    result: dict[str, Any]  # 최종 응답 payload (기능별 스키마로 직렬화 가능)
    source: str             # ai | mock
    error: str
