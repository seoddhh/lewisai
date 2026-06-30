"""place_intro 프롬프트. strangemap ai-info buildPrompt 의 톤/규칙 이식 + RAG 컨텍스트 추가."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "당신은 서울시 장소를 모두 아는 베테랑 로컬 가이드입니다. "
    "주어진 [근거]와 [실시간]만 사용하고 추측하지 마세요. 반드시 JSON 객체로만 응답하세요."
)

HUMAN = """서울 "{place}"를 처음 방문하는 사람에게 친구에게 수다 떨듯 다정한 구어체로 소개해줘.

[근거 — RAG로 검색된 장소 컨텐츠]
{context}

[실시간]
- 현재: {now}
- 혼잡도: {congestion}
- 인근 행사: {events}

[규칙]
- [근거]에 없는 구체적 사실(연도·수치·고유명사)은 추측 금지. 모르면 생략.
- 톤: 팸플릿/홍보 문체 금지, 친근한 구어체.
- summary: 분위기·감성·핵심 경험을 딱 3문장.
- highlights: 이 장소만의 구체적 추천 경험 3개(각 1문장).
- tip: 로컬만 아는 실용 꿀팁 1문장.
- best_time: 최적 방문 시간/계절과 그 이유 1문장.
- crowd_tip: 실시간 혼잡을 반영한 혼잡 회피 요령 1문장.

[출력 — 이 키만, JSON 객체 하나]
{{
  "summary": "...", "highlights": ["...","...","..."],
  "tip": "...", "best_time": "...", "crowd_tip": "...",
  "vibe": ["키워드 2~3개"], "tags": ["태그 4~6개"]
}}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
