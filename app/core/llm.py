"""LLM 프로바이더 레이어 — 제미나이 우선, 쿼터 소진 시 클로드 자동 폴백.

모든 체인은 기존처럼 `PROMPT | get_llm()` 형태로 사용한다.
get_llm()이 반환하는 FallbackLLM 이 프로바이더 선택/폴백/로깅을 담당한다.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from app.config import get_settings

logger = logging.getLogger("lewisai.llm")

# 쿼터/한도 계열로 판정할 메시지 마커 (프로바이더별 표현이 제각각이라 문자열 검사 병행)
_QUOTA_MARKERS = (
    "429",
    "quota",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "rate_limit",
    "too many requests",
    "timed out",
    "timeout",
)
_QUOTA_CLASS_NAMES = ("ResourceExhausted", "TooManyRequests", "DeadlineExceeded")


def is_quota_error(err: BaseException) -> bool:
    """제미나이 쿼터 소진/429/타임아웃 계열 에러인지 판정 (폴백 트리거 조건)."""
    seen: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _QUOTA_CLASS_NAMES:
            return True
        msg = str(cur).lower()
        if any(m in msg for m in _QUOTA_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class FallbackLLM(Runnable):
    """primary(제미나이) 실패 시 secondary(클로드)로 폴백하는 Runnable.

    - 폴백 트리거: is_quota_error() == True 인 예외만. 그 외 예외는 그대로 전파.
    - 어떤 프로바이더가 응답했는지 logger 로 기록 (비용/품질 모니터링).
    - bind_tools() 는 양쪽 모델에 각각 bind 한 새 FallbackLLM 을 반환
      (Gemini function-calling ↔ Claude tool_use 스펙 변환은 LangChain이 처리).
    """

    def __init__(self, primary: Runnable, secondary: Runnable | None = None):
        self.primary = primary
        self.secondary = secondary

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage:
        try:
            msg = self.primary.invoke(input, config, **kwargs)
            logger.info("llm.provider=gemini ok")
            return msg
        except Exception as err:
            if self.secondary is None or not is_quota_error(err):
                raise
            logger.warning("llm.provider=gemini failed (%s) → claude fallback", err)
            msg = self.secondary.invoke(input, config, **kwargs)
            logger.info("llm.provider=claude ok (fallback)")
            return msg

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage:
        try:
            msg = await self.primary.ainvoke(input, config, **kwargs)
            logger.info("llm.provider=gemini ok")
            return msg
        except Exception as err:
            if self.secondary is None or not is_quota_error(err):
                raise
            logger.warning("llm.provider=gemini failed (%s) → claude fallback", err)
            msg = await self.secondary.ainvoke(input, config, **kwargs)
            logger.info("llm.provider=claude ok (fallback)")
            return msg

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FallbackLLM":
        secondary = (
            self.secondary.bind_tools(tools, **kwargs) if self.secondary else None
        )
        return FallbackLLM(self.primary.bind_tools(tools, **kwargs), secondary)


@lru_cache
def _gemini() -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.gemini_model,
        google_api_key=s.google_api_key,
        temperature=s.llm_temperature,
        max_output_tokens=s.llm_max_tokens,
        streaming=True,  # ainvoke도 내부 스트리밍 경로 → LangGraph messages 모드가 토큰 조각을 잡음
    )


@lru_cache
def _claude() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    s = get_settings()
    return ChatAnthropic(
        model=s.claude_model,
        api_key=s.anthropic_api_key,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        streaming=True,  # 폴백 경로도 토큰 단위로 흘리도록
    )


@lru_cache
def get_llm() -> FallbackLLM:
    s = get_settings()
    secondary = (
        _claude() if (s.llm_fallback_enabled and s.anthropic_api_key) else None
    )
    return FallbackLLM(_gemini(), secondary)


def extract_text(content: str | list) -> str:
    """AIMessage.content를 텍스트로 정규화.

    gemini-3.x 계열은 content를 문자열이 아니라
    [{"type": "text", "text": "...", "extras": {...}}] 형태의 블록 리스트로 반환한다.
    클로드도 블록 리스트를 쓰므로 (tool_use/thinking 블록은 건너뜀) 양쪽 모두 커버된다.
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
