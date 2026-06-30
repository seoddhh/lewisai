"""LLM 팩토리 — 공급자(nim|openai|gemini|local)에 따라 LangChain 챗 모델 반환.

nim/openai/local 은 모두 OpenAI 호환 엔드포인트라 ChatOpenAI(base_url) 한 경로로 묶는다.
gemini 만 네이티브 langchain-google-genai 를 쓴다.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings


@lru_cache
def get_llm() -> BaseChatModel:
    s = get_settings()

    if s.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=s.gemini_model,
            google_api_key=s.google_api_key,
            temperature=s.llm_temperature,
            max_output_tokens=s.llm_max_tokens,
        )

    # nim | openai | local  → OpenAI 호환
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key or "not-needed",  # 로컬 서버는 키 불필요
        model=s.llm_model,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
    )
