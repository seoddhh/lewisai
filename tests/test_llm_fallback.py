"""FallbackLLM (제미나이 → 클로드 폴백) 단위 테스트."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app.core.llm import FallbackLLM, extract_text, is_quota_error


class FakeModel:
    """invoke/ainvoke/bind_tools 만 흉내내는 최소 스텁."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls = 0
        self.bound_tools = None

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return AIMessage(content=self.reply)

    async def ainvoke(self, input, config=None, **kwargs):
        return self.invoke(input, config, **kwargs)

    def bind_tools(self, tools, **kwargs):
        bound = FakeModel(self.reply, self.error)
        bound.bound_tools = tools
        return bound


class QuotaError(Exception):
    pass


class ResourceExhausted(Exception):
    """google.api_core.exceptions.ResourceExhausted 와 동명의 가짜 클래스."""


def test_is_quota_error_truth_table():
    assert is_quota_error(QuotaError("429 Too Many Requests"))
    assert is_quota_error(QuotaError("Quota exceeded for model"))
    assert is_quota_error(QuotaError("RESOURCE_EXHAUSTED"))
    assert is_quota_error(QuotaError("rate limit reached"))
    assert is_quota_error(QuotaError("Request timed out"))
    assert is_quota_error(ResourceExhausted("whatever"))  # 클래스명 매칭
    assert not is_quota_error(ValueError("invalid argument"))
    assert not is_quota_error(QuotaError("400 bad request"))


def test_is_quota_error_follows_cause_chain():
    inner = QuotaError("429")
    outer = RuntimeError("wrapped provider error")
    outer.__cause__ = inner
    assert is_quota_error(outer)


async def test_fallback_on_quota_error():
    primary = FakeModel(error=QuotaError("429 quota exceeded"))
    secondary = FakeModel(reply="클로드 응답")
    llm = FallbackLLM(primary, secondary)
    msg = await llm.ainvoke("hi")
    assert msg.content == "클로드 응답"
    assert primary.calls == 1


async def test_non_quota_error_reraised():
    primary = FakeModel(error=ValueError("bad prompt"))
    secondary = FakeModel(reply="클로드 응답")
    llm = FallbackLLM(primary, secondary)
    with pytest.raises(ValueError):
        await llm.ainvoke("hi")


async def test_no_secondary_reraises_quota_error():
    llm = FallbackLLM(FakeModel(error=QuotaError("429")), None)
    with pytest.raises(QuotaError):
        await llm.ainvoke("hi")


async def test_primary_success_skips_fallback():
    primary = FakeModel(reply="제미나이 응답")
    secondary = FakeModel(reply="클로드 응답")
    llm = FallbackLLM(primary, secondary)
    msg = await llm.ainvoke("hi")
    assert msg.content == "제미나이 응답"


def test_sync_invoke_fallback():
    llm = FallbackLLM(FakeModel(error=QuotaError("429")), FakeModel(reply="ok"))
    assert llm.invoke("hi").content == "ok"


def test_bind_tools_binds_both_sides():
    tools = [{"name": "dummy"}]
    llm = FallbackLLM(FakeModel(), FakeModel()).bind_tools(tools)
    assert isinstance(llm, FallbackLLM)
    assert llm.primary.bound_tools == tools
    assert llm.secondary.bound_tools == tools


def test_extract_text_variants():
    # 순수 문자열
    assert extract_text("안녕") == "안녕"
    # 제미나이 3.x 블록 리스트
    gemini = [{"type": "text", "text": "서울", "extras": {}}, {"type": "text", "text": "여행"}]
    assert extract_text(gemini) == "서울여행"
    # 클로드 블록 리스트 (tool_use/thinking 블록은 건너뜀)
    claude = [
        {"type": "thinking", "thinking": "..."},
        {"type": "text", "text": "종로 코스"},
        {"type": "tool_use", "id": "x", "name": "t", "input": {}},
    ]
    assert extract_text(claude) == "종로 코스"
    # 문자열 조각 리스트
    assert extract_text(["a", "b"]) == "ab"
