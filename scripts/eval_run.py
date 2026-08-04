"""모델 평가 실행 하네스 — docs/LLM-eval.md 9절 4단계.

    # 모델은 환경변수로 고른다. 한 프로세스 = 한 모델이다.
    LLM_PROVIDER=upstage UPSTAGE_MODEL=solar-open2 \
      PYTHONPATH=. uv run python -m scripts.eval_run --arm solar-open2

    PYTHONPATH=. uv run python -m scripts.eval_run --arm solar-pro4 --only P01,P04 --seeds 11
    PYTHONPATH=. uv run python -m scripts.eval_run --arm gemini --throttle 60   # 분당 1콜

`eval/personas.json` 을 읽어 페르소나 × seed 만큼 코스를 만들고, 결과를
`eval/runs/<arm>/<페르소나>_<seed>.json` 으로 저장한다. 채점과 심사는 저장된
파일을 나중에 읽어서 한다 — 생성이 비싸고 채점은 싸기 때문이다.

## 무엇을 기록하는가

payload(코스 원문) 말고도 채점에 필요한 것을 함께 남긴다. 나중에 다시 만들 수 없는 값들이다.

  - 노드별 소요 시간 (astream 의 yield 간격)
  - LLM 호출 수 · 호출별 소요 시간 · 입출력 토큰
  - 임베딩 호출 수 · 벡터검색 호출 수
  - 후보 목록 — "없는 장소를 지어냈는지" 채점하려면 **그때 후보가 무엇이었는지**가 있어야 한다
  - 모델 이름과 실행 시각 — 나중에 어느 조건의 결과인지 구분하는 유일한 근거

## 왜 HTTP 가 아니라 그래프를 직접 부르는가

서버를 띄우면 노드별 시간과 후보 목록을 꺼낼 수 없다. `/agent/chat` 응답에는
최종 코스만 들어 있고 중간 상태가 없다. 채점에 중간 상태가 필요하므로 직접 부른다.

## seed 에 대한 주의

칩으로 들어온 요청은 `req["seed"]` 가 후보 뽑기의 무작위성을 정한다. 그러나 **자연어
요청은 `parse_intent` 가 `req` 를 통째로 새로 만들면서 seed 를 지운다.** 그래서 자연어
페르소나는 seed 로 결과를 흔들 수 없고, 대신 `parse_intent` 자체가 온도 0.7 의 LLM
호출이라 반복하면 다른 결과가 나온다. 기록에 `seed_applied` 로 구분해 남긴다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PERSONAS = Path("eval/personas.json")
OUT_ROOT = Path("eval/runs")


# ── 계측 ────────────────────────────────────────────────────────────────────
class Counters:
    """한 번 실행하는 동안의 외부 호출 기록. 실행마다 reset 한다."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.embed_sec = 0.0
        self.search_calls = 0
        self.llm_calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.__init__()

    def summary(self) -> dict[str, Any]:
        return {
            "embed_calls": self.embed_calls,
            "embed_sec": round(self.embed_sec, 3),
            "search_calls": self.search_calls,
            "llm_calls": len(self.llm_calls),
            "llm_sec_total": round(sum(c["sec"] for c in self.llm_calls), 3),
            "llm_sec_max": round(max((c["sec"] for c in self.llm_calls), default=0.0), 3),
            "input_tokens": sum(c["in"] for c in self.llm_calls),
            "output_tokens": sum(c["out"] for c in self.llm_calls),
            "truncated": sum(1 for c in self.llm_calls if c["truncated"]),
            "per_call": self.llm_calls,
        }


COUNTERS = Counters()
_THROTTLE = {"sec": 0.0, "last": 0.0, "lock": None}


def _tokens(msg) -> tuple[int, int]:
    u = getattr(msg, "usage_metadata", None) or {}
    if u:
        return u.get("input_tokens", 0), u.get("output_tokens", 0)
    meta = getattr(msg, "response_metadata", None) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    return tu.get("prompt_tokens", 0), tu.get("completion_tokens", 0)


def _truncated(msg) -> bool:
    meta = getattr(msg, "response_metadata", None) or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason") or ""
    return str(reason).lower() in {"length", "max_tokens"}


def _instrument() -> None:
    """임베딩·검색·LLM 호출을 감싸 센다. 프로세스당 1회만 부른다.

    LLM 은 **활성 인스턴스의 클래스**를 감싼다. 클래스 이름을 박아두면 프로바이더를
    바꿨을 때 조용히 0 으로 찍힌다 (profile_course 에서 실제로 그랬다).
    노드들이 `(prompt | llm).ainvoke(...)` 로 부르므로 `ainvoke` 를 감싸면 전부 잡힌다.
    """
    from app.core.embeddings import get_embeddings
    from app.core.llm import get_llm
    from app.rag import retriever

    ecls = type(get_embeddings())
    orig_embed = ecls.embed_query

    def counted_embed(self, text: str):
        t0 = time.perf_counter()
        try:
            return orig_embed(self, text)
        finally:
            COUNTERS.embed_calls += 1
            COUNTERS.embed_sec += time.perf_counter() - t0

    ecls.embed_query = counted_embed

    orig_search = retriever.search_with_score

    def counted_search(*a, **kw):
        COUNTERS.search_calls += 1
        return orig_search(*a, **kw)

    retriever.search_with_score = counted_search

    llm = get_llm()
    # OpenAI 호환 모델은 스트리밍일 때 토큰 사용량을 기본으로 안 준다
    # (stream_options.include_usage 를 켜야 온다). 비용을 재려면 필수라 여기서만 켠다.
    if "stream_usage" in type(llm).model_fields and getattr(llm, "streaming", False):
        llm.stream_usage = True

    lcls = type(llm)
    orig_invoke = lcls.ainvoke

    async def counted_invoke(self, *a, **kw):
        # 쿼터가 빡빡한 모델은 호출 간격을 벌린다. 앱 코드가 아니라 여기서만 건다.
        if _THROTTLE["sec"] > 0:
            async with _THROTTLE["lock"]:
                gap = _THROTTLE["sec"] - (time.perf_counter() - _THROTTLE["last"])
                if gap > 0:
                    await asyncio.sleep(gap)
                _THROTTLE["last"] = time.perf_counter()
        t0 = time.perf_counter()
        err = None
        try:
            msg = await orig_invoke(self, *a, **kw)
        except Exception as e:  # noqa: BLE001 — 실패도 기록해야 실패율을 낼 수 있다
            err, msg = type(e).__name__, None
            raise
        finally:
            tin, tout = _tokens(msg) if msg is not None else (0, 0)
            COUNTERS.llm_calls.append({
                "sec": round(time.perf_counter() - t0, 3),
                "in": tin, "out": tout,
                "truncated": _truncated(msg) if msg is not None else False,
                "error": err,
            })
        return msg

    lcls.ainvoke = counted_invoke


# ── 실행 ────────────────────────────────────────────────────────────────────
def _initial(persona: dict, seed: int | None) -> tuple[dict[str, Any], str, bool]:
    """그래프에 넣을 처음 상태 → (state, 진입방식, seed 가 실제로 먹는지)."""
    req = persona["request"]
    chips, message = req.get("chips"), req.get("message", "")
    if chips is None:
        # 자연어 진입 — parse_intent 가 req 를 새로 만들면서 seed 를 지운다
        return {"message": message}, "message", False
    r: dict[str, Any] = {"note": message, "chips": chips}
    if seed is not None:
        r["seed"] = seed
    return {"req": r}, "chips", seed is not None


async def run_one(persona: dict, seed: int | None) -> dict[str, Any]:
    from app.graph.build import _payload, course_steps, get_agent_graph

    COUNTERS.reset()
    initial, entry, seed_applied = _initial(persona, seed)

    timings: dict[str, float] = {}
    state: dict[str, Any] = {}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = last = time.perf_counter()
    error = None
    try:
        async for data in get_agent_graph().astream(initial, stream_mode="updates"):
            now = time.perf_counter()
            for node, patch in data.items():
                timings[node] = round(timings.get(node, 0.0) + (now - last), 3)
                if isinstance(patch, dict):
                    state.update(patch)
            last = now
    except Exception as e:  # noqa: BLE001 — 한 건이 죽어도 나머지는 돌려야 한다
        error = f"{type(e).__name__}: {e}"
    total = round(time.perf_counter() - t0, 3)

    payload = _payload(state) if state.get("result") or state.get("selected") else {}
    return {
        "persona": persona["id"],
        "persona_name": persona.get("name"),
        "seed": seed,
        "seed_applied": seed_applied,
        "entry": entry,
        "started_at": started,
        "total_sec": total,
        "timings": timings,
        "counters": COUNTERS.summary(),
        "error": error,
        "source": payload.get("source"),
        "course": payload.get("course"),
        "steps": payload.get("steps"),
        # 채점용 중간 상태 — 이게 없으면 "없는 장소를 지어냈는지"를 판정할 수 없다
        "candidates": [
            {k: c.get(k) for k in ("name", "category", "area", "lat", "lng", "day_hint")}
            for c in (state.get("candidates") or [])
        ],
        "day_areas": state.get("day_areas") or {},
        "skeleton_segments": len((state.get("skeleton") or {}).get("segments") or []),
    }


def _meta() -> dict[str, Any]:
    from app.config import get_settings
    from app.core.llm import get_llm

    s = get_settings()
    llm = get_llm()
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = ""
    return {
        "llm_provider": s.llm_provider,
        "llm_model": getattr(llm, "model", None) or getattr(llm, "model_name", None),
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.active_embedding_model,
        "chroma_collection": s.chroma_collection,
        "temperature": s.llm_temperature,
        "max_tokens": s.llm_max_tokens,
        "commit": sha,
    }


async def _warmup() -> None:
    """임베딩 모델을 미리 메모리에 올린다.

    첫 호출은 모델 로드까지 하느라 2.8초쯤 걸리고 그 다음부터는 0.15초다.
    예열하지 않으면 첫 페르소나만 검색 시간이 20배로 찍혀 비교가 망가진다.
    """
    from app.core.embeddings import get_embeddings

    t0 = time.perf_counter()
    await asyncio.to_thread(get_embeddings().embed_query, "예열")
    COUNTERS.reset()
    print(f"· 임베딩 예열 {time.perf_counter() - t0:.1f}s")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="결과를 저장할 이름 (예: solar-open2)")
    ap.add_argument("--personas", default=str(PERSONAS))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--seeds", default="", help="쉼표로 구분. 비우면 personas.json 의 seeds")
    ap.add_argument("--only", default="", help="쉼표로 구분한 페르소나 번호 (예: P01,P04)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만")
    ap.add_argument("--throttle", type=float, default=0.0, help="LLM 호출 간 최소 간격(초)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.personas).read_text(encoding="utf-8"))
    personas = data["personas"]
    seeds = ([int(x) for x in args.seeds.split(",") if x.strip()]
             if args.seeds else data.get("seeds") or [None])
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        personas = [p for p in personas if p["id"] in want]
    if args.limit:
        personas = personas[: args.limit]

    jobs = [(p, s) for p in personas for s in seeds]
    outdir = Path(args.out) / args.arm
    print(f"arm={args.arm} · 페르소나 {len(personas)} × seed {len(seeds)} = {len(jobs)}회 → {outdir}")

    if args.dry_run:
        for p, s in jobs:
            print(f"  {p['id']} seed={s}  {p.get('name')}")
        return 0

    _THROTTLE["sec"] = args.throttle
    _THROTTLE["lock"] = asyncio.Lock()
    if args.throttle:
        print(f"· LLM 호출 간격 {args.throttle}초 강제")

    _instrument()
    meta = _meta()
    print(f"· 모델 {meta['llm_provider']}/{meta['llm_model']} · 임베딩 {meta['embedding_model']}"
          f" · 컬렉션 {meta['chroma_collection']} · 커밋 {meta['commit']}")
    await _warmup()

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "_meta.json").write_text(
        json.dumps({**meta, "arm": args.arm, "personas": len(personas), "seeds": seeds},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    ok = fail = 0
    t_all = time.perf_counter()
    for i, (p, seed) in enumerate(jobs, 1):
        rec = await run_one(p, seed)
        rec["meta"] = meta
        (outdir / f"{p['id']}_seed{seed}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

        c = rec["counters"]
        stops = len((rec.get("course") or {}).get("stops") or [])
        mark = "✅" if rec["source"] == "ai" and not rec["error"] else "❌"
        if mark == "✅":
            ok += 1
        else:
            fail += 1
        print(f"  [{i:>2}/{len(jobs)}] {mark} {p['id']} seed={seed} "
              f"{rec['total_sec']:5.1f}s · LLM {c['llm_calls']}콜 "
              f"({c['input_tokens']}/{c['output_tokens']}tok) · 임베딩 {c['embed_calls']}회 "
              f"· 스톱 {stops} · source={rec['source']}"
              + (f" · {rec['error'][:60]}" if rec["error"] else ""))

    print(f"\n완료: 성공 {ok} · 실패 {fail} · 총 {time.perf_counter() - t_all:.0f}초 → {outdir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
