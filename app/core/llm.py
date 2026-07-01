"""제미나이 api 사용 챗 모델 구조"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings


@lru_cache
def get_llm() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.gemini_model,
        google_api_key=s.google_api_key,
        temperature=s.llm_temperature,
        max_output_tokens=s.llm_max_tokens,
    )


def extract_text(content: str | list) -> str:
    """AIMessage.content를 텍스트로 정규화.

    gemini-3.x 계열은 content를 문자열이 아니라
    [{"type": "text", "text": "...", "extras": {...}}] 형태의 블록 리스트로 반환한다.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)
