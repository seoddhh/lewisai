"""심사 모델 1:1 비교 — docs/LLM-eval.md 6절.

    PYTHONPATH=. uv run python -m scripts.eval_judge
    PYTHONPATH=. uv run python -m scripts.eval_judge --challengers solar-pro4
    PYTHONPATH=. uv run python -m scripts.eval_judge --limit 5 --dry-run

`eval/runs/<모델>/` 에 저장된 결과를 읽어 기준 모델과 도전 모델을 짝지어 판정한다.
코스를 다시 만들지 않으므로, 판정 기준을 고쳐서 몇 번이든 다시 돌릴 수 있다.

## 판정 방식

같은 페르소나·같은 seed 의 두 코스를 나란히 놓고 어느 쪽이 나은지 고르게 한다.
축은 세 개이고, 총점 하나로 합치지 않는다 — 축마다 이기는 모델이 다를 수 있고
그 사실 자체가 결정에 필요한 정보이기 때문이다.

  실행 가능성  : 이 코스를 실제로 그대로 다닐 수 있는가
  조건 반영도  : 사용자가 고른 조건이 결과에 실제로 반영됐는가
  한국어       : 읽었을 때 어색하지 않은가

## 순서 편향을 어떻게 막는가

심사 모델은 먼저 보여준 쪽을 고르는 경향이 있다. 그래서 **같은 쌍을 두 번 판정한다.**
한 번은 기준 모델을 왼쪽에, 한 번은 오른쪽에 둔다. 두 판정이 같은 코스를 가리킬 때만
그 결과를 인정하고, 엇갈리면 무승부로 처리한다. 엇갈린 비율(`flip_rate`)도 함께 남긴다 —
이 값이 높으면 그 축의 판정 자체를 믿을 수 없다는 뜻이다.

## 무엇을 보여주고 무엇을 감추는가

모델 이름은 절대 넣지 않는다. 보여주는 것은 요청 조건과 코스 본문(제목·소개·장소별
선정 이유·할 일)뿐이다. 소요 시간이나 토큰 수 같은 운영 지표는 코드로 이미 재고 있어서
심사 모델에 넣을 이유가 없고, 넣으면 판정이 그쪽에 끌린다.

## 한쪽이 실패한 쌍은 제외한다

`source=mock` 인 코스는 대체 응답이라 내용이 빈약하다. 이걸 비교에 넣으면 상대가
자동으로 이겨서 승률이 부풀려진다. 실패는 이미 별도 지표(생성 실패율)로 세고 있으므로
여기서는 제외하고, 제외한 건수를 결과에 남긴다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

RUNS = Path("eval/runs")
OUT = Path("eval/scores_judge.json")
BASE_DEFAULT = "solar-open2"
AXES = ("feasibility", "condition_fit", "korean", "overall")
AXIS_KR = {"feasibility": "실행 가능성", "condition_fit": "조건 반영도",
           "korean": "한국어", "overall": "종합"}

RUBRIC = """너는 서울 여행 코스 두 개를 비교하는 심사위원이다. JSON 으로만 답하라.

form: {"reason":"판단 근거 두 문장 이내","feasibility":"A"|"B"|"tie","condition_fit":"A"|"B"|"tie","korean":"A"|"B"|"tie","overall":"A"|"B"|"tie"}

판정 기준
- feasibility(실행 가능성): 장소 순서와 동선이 자연스러운가. 할 일이 그 장소에서 실제로
  할 수 있는 일인가. 막연한 표현("여유롭게 즐기기")보다 구체적인 행동이 낫다.
- condition_fit(조건 반영도): 아래 '요청 조건'이 코스에 실제로 반영됐는가. 동반자·목적·
  시간대·권역이 장소 선택과 설명에 드러나는가. 조건과 무관한 일반론이면 감점이다.
- korean(한국어): 문장이 자연스럽고 군더더기가 없는가. "다채로운", "다양한", "아름다운"
  같은 상투어가 반복되면 감점이다.
- overall(종합): 위 셋을 종합해 어느 코스를 사용자에게 주겠는가.

규칙
- reason 을 먼저 쓰고 그 다음에 판정을 쓴다.
- 우열을 가리기 어려우면 tie 를 쓴다. 억지로 고르지 말 것.
- 글이 길다고 좋은 것이 아니다. 길이가 아니라 내용으로 판단하라.
- 장소 개수가 다른 것은 사용자가 고른 조건이므로 그 자체로는 감점 사유가 아니다."""


def _course_text(rec: dict) -> str:
    c = rec.get("course") or {}
    lines = [f"제목: {c.get('title', '')}"]
    if c.get("subtitle"):
        lines.append(f"부제: {c['subtitle']}")
    if c.get("description"):
        lines.append(f"소개: {c['description']}")
    for i, s in enumerate((c.get("stops") or []), 1):
        if s.get("slot_type") != "place":
            continue
        acts = " / ".join(s.get("activities") or [])
        lines.append(f"{i}. {s.get('name', '')}"
                     + (f"\n   선정 이유: {s['reason']}" if s.get("reason") else "")
                     + (f"\n   할 일: {acts}" if acts else ""))
    return "\n".join(lines)


def _conditions(rec: dict, personas: dict) -> str:
    p = personas.get(rec["persona"], {})
    req = p.get("request", {})
    chips, msg = req.get("chips"), req.get("message", "")
    if chips is None:
        return f'자연어 요청: "{msg}"'
    keep = ("audience", "days", "place_count", "locations", "purposes",
            "companions", "time", "meals", "pace")
    got = {k: chips[k] for k in keep if chips.get(k) not in (None, [], "")}
    out = json.dumps(got, ensure_ascii=False)
    return out + (f'\n자유 입력: "{msg}"' if msg else "")


def _parse(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        got = json.loads(t[a:b + 1])
    except json.JSONDecodeError:
        return None
    return got if all(got.get(k) in {"A", "B", "tie"} for k in AXES) else None


async def _judge_once(llm, cond: str, left: str, right: str) -> dict | None:
    prompt = (f"요청 조건:\n{cond}\n\n[코스 A]\n{left}\n\n[코스 B]\n{right}\n\n"
              "위 형식의 JSON 으로만 답하라.")
    try:
        res = await llm.ainvoke([("system", RUBRIC), ("human", prompt)])
    except Exception as err:  # noqa: BLE001
        print(f"    ! 판정 호출 실패: {type(err).__name__}", file=sys.stderr)
        return None
    return _parse(str(res.content))


def _winner(v: str, base_is_left: bool) -> str:
    """판정 글자를 '기준 모델 승 / 도전 모델 승 / 무승부' 로 바꾼다."""
    if v == "tie":
        return "tie"
    picked_left = v == "A"
    return "base" if picked_left == base_is_left else "chal"


async def compare(llm, base_rec: dict, chal_rec: dict, cond: str) -> dict:
    """같은 쌍을 자리를 바꿔 두 번 판정한다. 두 판정이 일치할 때만 승패로 인정한다."""
    bt, ct = _course_text(base_rec), _course_text(chal_rec)
    r1 = await _judge_once(llm, cond, bt, ct)   # 기준 모델이 왼쪽(A)
    r2 = await _judge_once(llm, cond, ct, bt)   # 기준 모델이 오른쪽(B)
    if r1 is None or r2 is None:
        return {"ok": False}

    out = {"ok": True, "reason_1": r1.get("reason", ""), "reason_2": r2.get("reason", "")}
    for ax in AXES:
        w1 = _winner(r1[ax], base_is_left=True)
        w2 = _winner(r2[ax], base_is_left=False)
        out[ax] = w1 if w1 == w2 else "tie"
        out[f"{ax}_flip"] = w1 != w2      # 자리를 바꿨더니 판정이 뒤집혔는가
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--challengers", default="", help="쉼표 구분. 비우면 base 외 전부")
    ap.add_argument("--personas", default="eval/personas.json")
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"))
    ap.add_argument("--base-url", default=os.getenv("NVIDIA_BASE_URL",
                                                    "https://integrate.api.nvidia.com/v1"))
    ap.add_argument("--limit", type=int, default=0, help="쌍마다 앞에서 N개만")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    pmeta = {p["id"]: p for p in
             json.loads(Path(args.personas).read_text(encoding="utf-8"))["personas"]}

    def load(arm: str) -> dict[tuple[str, Any], dict]:
        out = {}
        for f in sorted((RUNS / arm).glob("P*.json")):
            d = json.load(open(f, encoding="utf-8"))
            out[(d["persona"], d["seed"])] = d
        return out

    base = load(args.base)
    if not base:
        print(f"❌ 기준 모델 결과가 없습니다: {RUNS / args.base}")
        return 1
    chals = ([c.strip() for c in args.challengers.split(",") if c.strip()]
             or [d.name for d in RUNS.iterdir() if d.is_dir() and d.name != args.base])

    print(f"기준 모델: {args.base} ({len(base)}건) · 심사 모델: {args.model}")
    plan = {}
    for arm in chals:
        c = load(arm)
        keys = [k for k in base if k in c
                and base[k]["source"] == "ai" and c[k]["source"] == "ai"]
        skipped = len([k for k in base if k in c]) - len(keys)
        keys.sort()
        if args.limit:
            keys = keys[: args.limit]
        plan[arm] = (c, keys, skipped)
        print(f"  vs {arm:24} 비교 {len(keys)}쌍 · 실패로 제외 {skipped}쌍 "
              f"· 판정 호출 {len(keys) * 2}회")
    total = sum(len(v[1]) * 2 for v in plan.values())
    print(f"총 판정 호출 {total}회")
    if args.dry_run:
        return 0

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(base_url=args.base_url, api_key=os.environ["NVIDIA_API_KEY"],
                     model=args.model, temperature=0, max_tokens=1024)

    results: dict[str, Any] = {"base": args.base, "judge_model": args.model, "pairs": {}}
    t0 = time.perf_counter()
    for arm, (c, keys, skipped) in plan.items():
        tally = {ax: Counter() for ax in AXES}
        flips = Counter()
        details, bad = [], 0
        for i, k in enumerate(keys, 1):
            cond = _conditions(base[k], pmeta)
            v = await compare(llm, base[k], c[k], cond)
            if not v["ok"]:
                bad += 1
                continue
            for ax in AXES:
                tally[ax][v[ax]] += 1
                flips[ax] += int(v[f"{ax}_flip"])
            details.append({"persona": k[0], "seed": k[1],
                            **{ax: v[ax] for ax in AXES},
                            "reason": v["reason_1"]})
            if i % 10 == 0 or i == len(keys):
                o = tally["overall"]
                print(f"  [{arm}] {i}/{len(keys)} · 종합 기준 {o['base']} / "
                      f"무 {o['tie']} / 도전 {o['chal']}")
        n = max(1, sum(tally["overall"].values()))
        results["pairs"][arm] = {
            "n": n, "skipped_failed": skipped, "judge_errors": bad,
            "axes": {ax: {"base": tally[ax]["base"], "tie": tally[ax]["tie"],
                          "chal": tally[ax]["chal"],
                          "chal_win_rate": round(tally[ax]["chal"] / n * 100, 1),
                          "flip_rate": round(flips[ax] / n * 100, 1)} for ax in AXES},
            "details": details,
        }

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n총 {time.perf_counter() - t0:.0f}초 → {args.out}\n")

    for arm, r in results["pairs"].items():
        print(f"── {args.base} vs {arm}  (n={r['n']}, 제외 {r['skipped_failed']})")
        for ax in AXES:
            a = r["axes"][ax]
            print(f"   {AXIS_KR[ax]:8} 기준 {a['base']:2} / 무 {a['tie']:2} / 도전 {a['chal']:2}"
                  f"  → 도전 승률 {a['chal_win_rate']:5.1f}%  (순서 뒤집힘 {a['flip_rate']:.0f}%)")
    print("\n※ 표본 45 기준, 승률이 우연으로 설명되지 않으려면 약 65% 이상이어야 한다.")
    print("※ 순서 뒤집힘 비율이 높은 축은 판정 자체를 신뢰할 수 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
