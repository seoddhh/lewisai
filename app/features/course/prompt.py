"""course 프롬프트(1차 기본판). 검색된 장소/코스 청크로 동선을 구성."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "당신은 서울 테마 코스를 짜는 큐레이터입니다. 주어진 후보 장소만으로 "
    "3~4개 정류장의 도보 동선을 구성하세요. JSON 객체로만 응답하세요."
)

HUMAN = """사용자 요청: "{note}" (지역: {region}, 시간대: {time})

[후보 장소·코스 — 이 안에서만 정류장을 고름]
{candidates}

[규칙]
- 정류장(stops) 3~4개. name 은 후보 목록의 장소명만.
- 각 정류장에 preview(짧은 한 줄)·description(현장 감성)·duration 작성.
- 왜 이 순서인지 자연스러운 흐름이 되도록.

[출력 — JSON 객체]
{{"title":"...","subtitle":"...","description":"...",
  "stops":[{{"name":"후보 장소명","preview":"...","description":"...","duration":"..."}}],
  "tags":["..."]}}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
