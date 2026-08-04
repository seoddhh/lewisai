"""챗 LLM 을 만들어 주는 곳. 모든 체인은 `PROMPT | get_llm()` 형태로 쓴다.

어떤 모델을 쓸지는 `LLM_PROVIDER`(upstage|gemini|claude) 하나로 정해진다. 프로바이더가
막히면 코드가 아니라 .env 한 줄만 고쳐서 갈아탈 수 있게 하려는 것이다.
docs/llm-provider-eval.md 의 3-arm 평가도 이 값 하나만 바꿔가며 돌린다.
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

# 프로바이더마다 쿼터 초과 에러의 표현이 달라서, 클래스 이름과 메시지 문자열을 같이 본다.
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
    """쿼터 소진·429·타임아웃 계열 에러인지 본다. 원인 예외까지 거슬러 올라가며 확인한다."""
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
    """모델 두 개를 묶어, 앞쪽이 쿼터로 막히면 뒤쪽으로 넘기는 래퍼.

    지금은 프로바이더를 하나만 쓰기 때문에 아무 데서도 감싸지 않는다.
    백업 프로바이더가 다시 필요해지면 그대로 쓰려고 남겨둔 코드다.

    쿼터 계열 에러(is_quota_error)일 때만 넘긴다. 다른 에러는 그대로 올려보낸다.
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
    """업스테이지 솔라 모델. LLM_PROVIDER 가 upstage(기본)일 때 쓰인다."""
    from langchain_upstage import ChatUpstage

    s = get_settings()
    if not s.upstage_api_key:
        raise RuntimeError("UPSTAGE_API_KEY 가 설정되지 않았습니다 (.env 확인).")
    return ChatUpstage(
        model=s.upstage_model,
        api_key=s.upstage_api_key,
        base_url=s.upstage_base_url,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,  # 기본값이 짧아서, 안 주면 3일치 JSON이 중간에 잘린다
        reasoning_effort=s.llm_reasoning_effort,
        streaming=True,  # 켜둬야 LangGraph 가 토큰을 조각으로 흘려보낼 수 있다
    )


@lru_cache
def _gemini() -> BaseChatModel:
    """제미나이 모델. LLM_PROVIDER=gemini 일 때 쓰인다."""
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
def _claude() -> BaseChatModel:
    """클로드 모델. LLM_PROVIDER=claude 일 때 쓰인다.

    기본값 claude-haiku-4-5 는 솔라·제미나이와 같은 빠른/저가 티어라 평가 비교가 공정하다.
    이 세대는 temperature 를 그대로 받는다 — 상위 세대(Opus 4.7 계열)로 바꾸면
    temperature 가 거부되므로(400) 그때는 여기서 빼야 한다.
    """
    from langchain_anthropic import ChatAnthropic

    s = get_settings()
    if not s.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 가 설정되지 않았습니다 (.env 확인).")
    return ChatAnthropic(
        model=s.claude_model,
        api_key=s.anthropic_api_key,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        streaming=True,
    )


@lru_cache
def get_llm() -> BaseChatModel:
    """지금 설정된 챗 모델을 돌려준다. 체인은 전부 이 함수만 부른다."""
    provider = get_settings().llm_provider.lower()
    if provider == "gemini":
        return _gemini()
    if provider == "claude":
        return _claude()
    return _solar()


def extract_text(content: str | list) -> str:
    """모델 응답(AIMessage.content)에서 사람이 읽을 본문만 뽑아 문자열로 만든다.

    프로바이더마다 응답 모양이 다르다. 솔라는 문자열 하나로 주지만,
    제미나이는 [{"type": "text", ...}, ...] 처럼 블록 목록으로 준다.
    본문이 아닌 블록(생각 과정 등)은 건너뛰므로 어느 쪽이든 같은 결과가 나온다.
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
