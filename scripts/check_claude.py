"""클로드 연결 테스트 — 평가 arm C 가 실제로 돌 수 있는지 확인한다.

    PYTHONPATH=. uv run python scripts/check_claude.py

확인 항목
  1) ANTHROPIC_API_KEY 로드 여부
  2) 설정된 CLAUDE_MODEL 이 계정에서 실제로 호출 가능한지 (Models API)
  3) get_llm() 단발 호출 — 앱 체인과 완전히 같은 경로
  4) 스트리밍 호출 — LangGraph messages 모드가 토큰을 잡으려면 필수
  5) JSON 형식 준수 — select/compose 노드가 요구하는 출력 모양
  6) 토큰 사용량과 코스 1건당 예상 원가

`scripts/check_upstage.py` 의 클로드판이다. 세 arm 을 같은 방식으로 점검하려는 것.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from app.config import get_settings
from app.core.llm import extract_text, get_llm

OK, FAIL, WARN = "✅", "❌", "⚠️"

# claude-haiku-4-5 기준 (per MTok). 모델을 바꾸면 이 값도 같이 고칠 것.
USD_IN, USD_OUT = 1.00, 5.00
KRW = 1400


def _fail(msg: str) -> int:
    print(f"{FAIL} {msg}")
    return 1


async def main() -> int:
    s = get_settings()

    # 1) 키
    if not s.anthropic_api_key:
        return _fail("ANTHROPIC_API_KEY 가 비어 있습니다. .env 에 추가하세요.")
    print(f"{OK} ANTHROPIC_API_KEY 로드됨 (…{s.anthropic_api_key[-4:]})")
    print(f"   model={s.claude_model}  temperature={s.llm_temperature}  max_tokens={s.llm_max_tokens}")

    # 2) 모델이 계정에서 실제로 호출 가능한지. 오타·권한 문제를 여기서 걸러낸다
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=s.anthropic_api_key)
        info = client.models.retrieve(s.claude_model)
    except Exception as err:
        return _fail(f"모델 조회 실패 ({s.claude_model}): {err}")
    print(f"{OK} 모델 확인: {info.display_name} "
          f"(컨텍스트 {info.max_input_tokens:,} · 최대 출력 {info.max_tokens:,})")

    # 3) 앱과 같은 경로로 단발 호출. LLM_PROVIDER 를 강제로 claude 로 두고 부른다
    if s.llm_provider.lower() != "claude":
        print(f"{WARN} LLM_PROVIDER={s.llm_provider} 입니다. "
              f"이 테스트는 클로드를 직접 만들어 확인합니다 "
              f"(앱 전체를 arm C 로 돌리려면 .env 를 LLM_PROVIDER=claude 로).")
        from app.core.llm import _claude
        llm = _claude()
    else:
        llm = get_llm()

    t0 = time.perf_counter()
    try:
        msg = await llm.ainvoke("서울 종로 코스를 한 문장으로 추천해줘.")
    except Exception as err:
        return _fail(f"단발 호출 실패: {err}")
    dt = time.perf_counter() - t0
    print(f"{OK} 단발 호출 {dt:.1f}s — {extract_text(msg.content)[:60]}…")

    usage = msg.usage_metadata or {}
    n_in, n_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    print(f"   토큰 입력 {n_in} · 출력 {n_out}")

    # 4) 스트리밍. 안 되면 /agent/chat/stream 이 통째로 죽는다
    t0, chunks = time.perf_counter(), 0
    try:
        async for _ in llm.astream("서울 3대 명소를 쉼표로만 나열해줘."):
            chunks += 1
    except Exception as err:
        return _fail(f"스트리밍 실패: {err}")
    print(f"{OK} 스트리밍 {time.perf_counter() - t0:.1f}s — 조각 {chunks}개")

    # 5) JSON 준수. select 노드가 이 모양을 파싱하므로 여기서 깨지면 arm C 는 무의미하다
    try:
        raw = extract_text((await llm.ainvoke(
            '{"stops":[{"name":"...","reason":"..."}]} 형식의 JSON 만 출력해. '
            '설명·코드펜스 없이. 경복궁과 북촌한옥마을 두 곳을 넣어.'
        )).content)
        from app.core.json_parse import parse_json_object

        data = parse_json_object(raw)
        n = len(data.get("stops", []))
    except Exception as err:
        print(f"{WARN} JSON 파싱 실패 — 평가 시 salvage 경로를 탈 수 있습니다: {err}")
    else:
        print(f"{OK} JSON 파싱 성공 — stops {n}곳")

    # 6) 코스 1건당 예상 원가. 콜 5회(select 1 + compose 전역 1 + 청크 2 + 여유 1) 가정
    if n_in and n_out:
        per_call = (n_in / 1e6) * USD_IN + (n_out / 1e6) * USD_OUT
        print(f"\n참고 · 코스 1건당 예상 원가 (콜 5회 가정, 이 호출의 토큰 기준)")
        print(f"   ${per_call * 5:.5f} ≈ {per_call * 5 * KRW:.2f}원")
        print(f"   ※ 실제 프롬프트는 후보 목록이 붙어 훨씬 길다. 실측은 평가 하네스에서.")

    print(f"\n{OK} arm C 실행 가능합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
