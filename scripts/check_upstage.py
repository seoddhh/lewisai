"""업스테이지 솔라 연결 테스트.

    PYTHONPATH=. uv run python scripts/check_upstage.py

확인 항목
  1) UPSTAGE_API_KEY 로드 여부
  2) GET /v1/models — 계정에서 실제 호출 가능한 모델 alias 목록
     (solar-pro2 / solar-open2 등 어떤 이름이 유효한지 여기서 확인)
  3) 설정된 UPSTAGE_MODEL 이 그 목록에 있는지
  4) get_llm() 단발 호출 (앱 체인과 동일 경로)
  5) 스트리밍 호출 — LangGraph messages 모드가 토큰을 잡으려면 필수
"""
from __future__ import annotations

import asyncio
import sys

import httpx

from app.config import get_settings
from app.core.llm import extract_text, get_llm

OK, FAIL = "✅", "❌"


def _fail(msg: str) -> int:
    print(f"{FAIL} {msg}")
    return 1


async def main() -> int:
    s = get_settings()

    # 1) 키
    if not s.upstage_api_key:
        return _fail("UPSTAGE_API_KEY 가 비어 있습니다. .env 에 추가하세요.")
    print(f"{OK} UPSTAGE_API_KEY 로드됨 (…{s.upstage_api_key[-4:]})")
    print(f"   base_url={s.upstage_base_url}  model={s.upstage_model}")

    # 2) 사용 가능한 모델 목록
    try:
        r = httpx.get(
            f"{s.upstage_base_url}/models",
            headers={"Authorization": f"Bearer {s.upstage_api_key}"},
            timeout=15.0,
        )
        r.raise_for_status()
    except Exception as err:  # 네트워크/인증 실패
        return _fail(f"GET /models 실패: {err}")

    aliases = sorted(m["id"] for m in r.json().get("data", []))
    print(f"{OK} 호출 가능한 모델 {len(aliases)}개: {', '.join(aliases)}")

    # 3) 설정 모델이 목록에 있는지 — solar-open2 는 Private Beta 라
    #    /models 에 안 잡힐 수 있다. 실제 판정은 아래 호출 성공 여부로 한다.
    if s.upstage_model in aliases:
        print(f"{OK} UPSTAGE_MODEL={s.upstage_model} 목록에 있음")
    else:
        print(
            f"⚠️  UPSTAGE_MODEL={s.upstage_model} 은 /models 목록에 없습니다 "
            f"(Private Beta 면 정상). 실제 호출로 확인합니다."
        )

    llm = get_llm()

    # 4) 단발 호출
    try:
        msg = await llm.ainvoke("한 문장으로 자기소개해줘.")
    except Exception as err:
        return _fail(f"ainvoke 실패: {err}")
    print(f"{OK} ainvoke 응답: {extract_text(msg.content).strip()[:120]}")

    # 5) 스트리밍 — 프로바이더 SSE 연결 자체를 확인한다. 코스 출력은 JSON 이라
    #    토큰을 흘리지 않으므로(/agent/chat/stream 은 노드 progress 만 흘린다)
    #    런타임에 쓰이는 경로는 아니다.
    try:
        chunks = [c async for c in llm.astream("서울로 여행 팁 한 줄만.")]
    except Exception as err:
        return _fail(f"astream 실패: {err}")
    streamed = "".join(extract_text(c.content) for c in chunks)
    if len(chunks) < 2:
        return _fail(f"스트리밍 조각이 {len(chunks)}개뿐 — 토큰 단위로 흐르지 않습니다.")
    print(f"{OK} astream {len(chunks)}조각: {streamed.strip()[:120]}")

    print("\n🎉 솔라 연결 정상 — 앱 체인에서 그대로 사용 가능합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
