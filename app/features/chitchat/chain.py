"""간단한 응답에 사용"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.core.llm import extract_text, get_llm

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 서울 여행 도우미입니다. 짧고 친근하게 1~2문장으로 답하세요."),
        ("human", "{message}"),
    ]
)


class ChitchatRequest(BaseModel):
    message: str


class ChitchatResponse(BaseModel):
    reply: str


async def run(req: ChitchatRequest) -> ChitchatResponse:
    chain = PROMPT | get_llm()
    try:
        msg = await chain.ainvoke({"message": req.message})
        return ChitchatResponse(reply=extract_text(msg.content).strip())
    except Exception:  # noqa: BLE001
        return ChitchatResponse(reply="서울 어디가 궁금하세요? 장소·코스·상황 추천을 도와드릴게요.")
