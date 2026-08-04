"""심사 모델(NVIDIA NIM 경유) 연결 테스트 — docs/LLM-eval.md 의 심사 단계 준비물.

    PYTHONPATH=. uv run python scripts/check_judge.py
    PYTHONPATH=. uv run python scripts/check_judge.py --model z-ai/glm-5.2   # 다른 후보와 비교

확인 항목
  1) NVIDIA_API_KEY 로드 여부
  2) GET /v1/models — 계정에서 실제 호출 가능한 모델 목록에 있는지 확인한다
  3) 설정된 모델이 그 목록에 있는지
  4) 단발 호출
  5) **판정 JSON 형식 준수** — 실제 평가에서 쓸 출력 모양 그대로 요구해 본다
  6) **순서 편향 점검** — 같은 두 글을 순서만 바꿔 두 번 판정시켜, 같은 글을 고르는지 본다
  7) 토큰 사용량 → 평가 전체(270회) 예상치

## 왜 이 스크립트가 따로 있는가

심사 모델은 `get_llm()` 을 타지 않는다. `get_llm()` 은 `@lru_cache` 로 평가 대상 모델
하나에 묶여 있고, 심사 모델은 서비스 코드가 아니라 평가 전용이기 때문이다. 그래서 연결
방식도 앱과 분리돼 있고, 그 분리된 경로가 실제로 도는지 여기서 따로 확인한다.

**6번이 이 스크립트의 핵심이다.** 심사 모델은 먼저 보여준 쪽을 고르는 경향이 있어서,
그 경향이 크면 판정 결과 자체를 못 쓴다. 평가를 다 돌리고 나서 알면 늦으므로 미리 본다.
여기서 통과해도 표본 2건짜리 맛보기일 뿐이라, 본 평가에서는 모든 판정을 순서를 바꿔
두 번씩 수행하고 두 판정이 일치할 때만 결과로 인정한다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv

OK, FAIL, WARN = "✅", "❌", "⚠️"

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
# 심사 모델 기본값. 비교 대상 3사(업스테이지·구글·앤스로픽) 어디에도 속하지 않고,
# NIM 경로에서 판정 1회가 약 2초로 빨라서 골랐다 (glm-5.2 는 약 100초였다).
DEFAULT_MODEL = "openai/gpt-oss-120b"

# 본 평가의 판정 호출 수: 페르소나 15 × seed 3 × 3쌍 × 순서 바꿔 2회
TOTAL_JUDGE_CALLS = 15 * 3 * 3 * 2

# 실제 평가에서 쓸 판정 지시문의 축소판. 형식이 지켜지는지만 본다.
JUDGE_SYSTEM = (
    "너는 서울 여행 코스 두 개를 비교하는 심사위원이다. JSON 으로만 답하라.\n"
    'form: {"reason":"판단 근거 한두 문장",'
    '"feasibility":"A"|"B"|"tie",'
    '"condition_fit":"A"|"B"|"tie",'
    '"korean":"A"|"B"|"tie",'
    '"overall":"A"|"B"|"tie"}\n'
    "- reason 을 먼저 쓰고 그 다음에 판정을 쓴다.\n"
    "- feasibility: 실제로 그대로 다닐 수 있는가. "
    "condition_fit: 사용자가 고른 조건이 반영됐는가. korean: 한국어가 자연스러운가.\n"
    "- 우열을 가리기 어려우면 tie 를 쓴다. 억지로 고르지 말 것."
)

# 품질 차이가 분명한 두 글. 어느 쪽이 나은지는 사람이 봐도 명확해야
# 순서 편향 점검이 의미를 가진다.
GOOD = (
    "경복궁에서 수문장 교대의식을 보고, 국립민속박물관에서 조선시대 생활사 전시를 관람한다. "
    "도보 10분 거리의 북촌한옥마을로 이동해 골목 사진을 찍는다."
)
POOR = (
    "다채로운 매력이 가득한 곳에서 여유로운 시간을 보낼 수 있어요. "
    "아름다운 풍경과 다양한 즐길 거리가 있어 특별한 추억을 만들기 좋습니다."
)


def _fail(msg: str) -> int:
    print(f"{FAIL} {msg}")
    return 1


def _pair_prompt(a: str, b: str) -> str:
    return (
        "요청 조건: 종로·중구 / 연인과 / 문화·예술 / 오후\n\n"
        f"[A]\n{a}\n\n[B]\n{b}\n\n"
        "위 형식의 JSON 으로만 답하라."
    )


def _parse(text: str) -> dict | None:
    """판정 응답에서 JSON 객체만 건져낸다. 코드펜스로 감싸 오는 모델이 있다."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t.lstrip("`")
        t = t[4:] if t.startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL))
    args = ap.parse_args()

    load_dotenv()
    key = os.getenv("NVIDIA_API_KEY", "")

    # 1) 키
    if not key:
        return _fail(
            "NVIDIA_API_KEY 가 비어 있습니다.\n"
            "   build.nvidia.com 에서 발급한 뒤 .env 에 NVIDIA_API_KEY=nvapi-... 로 추가하세요."
        )
    print(f"{OK} NVIDIA_API_KEY 로드됨 (…{key[-4:]})")
    print(f"   base_url={args.base_url}  model={args.model}")

    # 2) 모델 목록 — NIM 카탈로그의 정확한 ID 는 여기서만 확인된다
    try:
        r = httpx.get(
            f"{args.base_url}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20.0,
        )
        r.raise_for_status()
    except Exception as err:
        return _fail(f"GET /models 실패: {err}")

    ids = sorted(m["id"] for m in r.json().get("data", []))
    stem = args.model.split("/")[-1].split("-")[0].lower()
    near = [i for i in ids if stem in i.lower()]
    print(f"{OK} 호출 가능한 모델 {len(ids)}개 · '{stem}' 포함 {len(near)}개: {', '.join(near) or '없음'}")

    # 3) 설정 모델 확인
    if args.model in ids:
        print(f"{OK} 모델 {args.model} 목록에 있음")
    else:
        print(f"{WARN} 모델 {args.model} 은 목록에 없습니다. 실제 호출로 확인합니다.")
        if near:
            print(f"   후보: {', '.join(near)}  →  --model 로 지정하거나 .env 의 JUDGE_MODEL 을 고치세요.")

    from langchain_openai import ChatOpenAI

    judge = ChatOpenAI(
        base_url=args.base_url,
        api_key=key,
        model=args.model,
        temperature=0,      # 판정은 매번 같아야 한다
        max_tokens=1024,
    )

    # 4) 단발 호출
    try:
        msg = await judge.ainvoke("한 문장으로 자기소개해줘.")
    except Exception as err:
        return _fail(f"ainvoke 실패: {err}")
    print(f"{OK} ainvoke 응답: {str(msg.content).strip()[:120]}")

    # 5) 판정 JSON 형식 준수 — 실제 평가에서 쓸 모양 그대로 요구한다
    tokens_in = tokens_out = 0
    verdicts: list[dict] = []
    for label, (a, b) in (("정순서 A=좋은글", (GOOD, POOR)), ("역순서 B=좋은글", (POOR, GOOD))):
        try:
            res = await judge.ainvoke(
                [("system", JUDGE_SYSTEM), ("human", _pair_prompt(a, b))]
            )
        except Exception as err:
            return _fail(f"판정 호출 실패 ({label}): {err}")

        usage = getattr(res, "usage_metadata", None) or {}
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)

        got = _parse(str(res.content))
        if got is None:
            return _fail(f"판정 응답이 JSON 이 아닙니다 ({label}): {str(res.content)[:200]}")

        need = {"reason", "feasibility", "condition_fit", "korean", "overall"}
        if missing := need - got.keys():
            return _fail(f"판정 JSON 에 필드 누락 ({label}): {sorted(missing)}")
        if got["overall"] not in {"A", "B", "tie"}:
            return _fail(f"overall 값이 A/B/tie 가 아닙니다 ({label}): {got['overall']}")

        verdicts.append(got)
        print(f"{OK} 판정 JSON 정상 ({label}) → overall={got['overall']} · {got['reason'][:60]}")

    # 6) 순서 편향 — 좋은 글이 A 였다가 B 로 자리를 바꿨으니, 판정도 A→B 로 뒤집혀야 정상이다.
    #    두 번 다 같은 자리를 골랐다면 내용이 아니라 위치를 보고 있다는 뜻이다.
    first, second = verdicts[0]["overall"], verdicts[1]["overall"]
    if first == "A" and second == "B":
        print(f"{OK} 순서 편향 없음 — 자리를 바꿔도 같은 글을 골랐습니다 (A→B)")
    elif first == second and first != "tie":
        print(
            f"{FAIL} 순서 편향 의심 — 두 번 다 '{first}' 자리를 골랐습니다.\n"
            f"   내용이 아니라 위치를 보고 판정할 가능성이 있습니다.\n"
            f"   본 평가에서는 모든 판정을 순서를 바꿔 두 번 하고, 일치할 때만 채택하세요."
        )
    else:
        print(f"{WARN} 판정이 엇갈리거나 무승부입니다 ({first} / {second}). 표본 2건이라 단정은 못 합니다.")

    # 7) 사용량 → 평가 전체 예상치
    per_call_in = tokens_in / 2
    per_call_out = tokens_out / 2
    print(
        f"\n판정 1회 평균 토큰: 입력 {per_call_in:.0f} · 출력 {per_call_out:.0f}\n"
        f"평가 전체 예상 ({TOTAL_JUDGE_CALLS}회 = 페르소나 15 × seed 3 × 3쌍 × 순서 2회):\n"
        f"  입력 {per_call_in * TOTAL_JUDGE_CALLS / 1000:.1f}K · "
        f"출력 {per_call_out * TOTAL_JUDGE_CALLS / 1000:.1f}K 토큰"
    )
    print(
        "  ※ 실제 판정은 코스 전문 두 개가 들어가므로 입력 토큰이 이보다 훨씬 큽니다.\n"
        "  ※ NVIDIA NIM 무료 사용은 테스트·평가 목적만 허용됩니다. 이 용도는 그 범위 안입니다."
    )

    print(f"\n🎉 심사 모델 연결 정상 — docs/LLM-eval.md 6절의 심사 단계에 사용 가능합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
