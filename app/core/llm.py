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
