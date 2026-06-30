"""recommend 프롬프트. strangemap ai-recommend buildPrompt 이식. 화이트리스트(후보 목록) 강제."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "당신은 서울시 컨텐츠 추천 전문가입니다. 제공된 후보 장소 목록 안에서만 "
    "사용자 상황에 맞는 활동 3가지를 추천하세요. JSON 배열로만 응답하세요."
)

HUMAN = """서울에서 오늘 할 수 있는 활동 3가지를 추천해줘. JSON 배열만 응답.

[사용자 상황]
- 현재: {now}
- 누구랑: {companion}
- 나이대: {age_group}
- 원하는 시간대: {time}
- 목적: {purpose}
- 원하는 위치: {region}
- 혼잡도 선호: {congestion}

[추천 가능한 장소 — 반드시 이 목록의 장소만. 목록 밖은 절대 불가]
{candidates}

[규칙]
- place 필드에는 위 목록의 장소명을 그대로 사용.
- 목록에 없는 장소명 사용 금지.

[출력 — JSON 배열, 정확히 3개]
[{{"title":"...","place":"목록의 장소명","duration":"...","description":"...","reason":"...","tags":["..."]}}]"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
