"""LLM 프로바이더 레이어 — 업스테이지 솔라(단일 프로바이더).

모든 체인은 기존처럼 `PROMPT | get_llm()` 형태로 사용한다.

이력: 이전에는 제미나이 우선 + 쿼터 소진 시 클로드 폴백(FallbackLLM) 구조였으나
솔라로 전환하면서 폴백 경로를 비활성화했다. 관련 코드는 삭제하지 않고
주석/미사용 상태로 남겨둔다(복구 가능하도록).
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

    NOTE: 솔라 단일 프로바이더로 전환하면서 현재는 사용하지 않는다.
    추후 솔라+백업 프로바이더 구성이 필요해지면 그대로 재사용 가능.

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
def _solar() -> BaseChatModel:
    """업스테이지 솔라 챗 모델. OpenAI 호환 엔드포인트를 ChatUpstage 가 감싼다."""
    from langchain_upstage import ChatUpstage

    s = get_settings()
    if not s.upstage_api_key:
        raise RuntimeError("UPSTAGE_API_KEY 가 설정되지 않았습니다 (.env 확인).")
    return ChatUpstage(
        model=s.upstage_model,
        api_key=s.upstage_api_key,
        base_url=s.upstage_base_url,
        temperature=s.llm_temperature,
        reasoning_effort=s.llm_reasoning_effort,
        streaming=True,  # ainvoke도 내부 스트리밍 경로 → LangGraph messages 모드가 토큰 조각을 잡음
    )


@lru_cache
def _gemini() -> BaseChatModel:
    """제미나이 챗 모델. 솔라 크레딧 소진 동안 임시로 이 경로를 사용."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY 가 설정되지 않았습니다 (.env 확인).")
    return ChatGoogleGenerativeAI(
        model=s.gemini_model,
        google_api_key=s.google_api_key,
        temperature=s.llm_temperature,
        max_output_tokens=s.llm_max_tokens,
        streaming=True,
    )


@lru_cache
def get_llm() -> BaseChatModel:
    return _solar()


# --- 이전 프로바이더(클로드 폴백) — 솔라 전환으로 비활성 -----------------------
# @lru_cache
# def _claude() -> BaseChatModel:
#     from langchain_anthropic import ChatAnthropic
#
#     s = get_settings()
#     return ChatAnthropic(
#         model=s.claude_model,
#         api_key=s.anthropic_api_key,
#         temperature=s.llm_temperature,
#         max_tokens=s.llm_max_tokens,
#         streaming=True,
#     )


def extract_text(content: str | list) -> str:
    """AIMessage.content를 텍스트로 정규화.

    솔라(OpenAI 호환)는 content를 문자열로 반환하므로 대개 그대로 통과한다.
    제미나이 3.x / 클로드는 [{"type": "text", "text": "..."}] 블록 리스트를 쓰므로
    (tool_use/thinking 블록은 건너뜀) 프로바이더를 되돌려도 그대로 동작한다.
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
