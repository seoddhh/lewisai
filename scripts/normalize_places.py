"""임베딩 장소 세트 스키마 v2 — 정규화·검증의 단일 출처.

    uv run python -m scripts.normalize_places --check    # 진단만 (파일 안 건드림)
    uv run python -m scripts.normalize_places            # v1 → v2 마이그레이션 + 정렬
    uv run python -m scripts.normalize_places --strict   # 개인화 축까지 필수로 검사

멱등이다 — 이미 v2 인 파일에 다시 돌려도 결과가 같다. **스키마 변환·검증만 하고 내용은
만들지 않는다.** 개인화 축(indoor/night_ok/stay_min/energy)과 highlights 는 규칙으로
만들면 안 되는 값이라 여기서 채우지 않는다 — 규칙으로 유도하면 카테고리를 라벨만 바꿔
다시 쓰게 되고, 그러면 검색 축이 하나로 붙어 다양성이 죽는다. 현재 1,651곳은 태깅이
끝나 있고(`--strict` 통과), 장소를 새로 추가할 때만 이 값들을 손으로 채우면 된다.

## v1 → v2 변경
- `coarse_category`(6종) → **`category`**. 27종 세부 분류는 버린다 — 다양성 판정에 못 쓰고
  (`쇼핑`·`상권·역세권`·`복합공간·쇼핑`이 다른 값이라 백화점 연속을 못 막는다), 세부 구체성은
  description 이 이미 품고 있다. 종류 축은 **장소당 하나**여야 셀 수 있다.
- **개인화 축 4개 신설** — `indoor` / `night_ok` / `stay_min` / `energy`.
  카테고리에서 유도되지 않는 값이라야 검색이 실제로 갈린다(실내 공원도, 밤에 여는 미술관도
  있다). 동반자 재랭킹(아이와→실내·짧게, 부모님과→이동 최소)과 밤 코스·우천 대응이 여기 걸린다.
- **`highlights` 신설** — 그 장소의 시설·볼거리 명사 2~4개. select 프롬프트가 요구하는
  "소개 글에 실제로 등장하는 특징에 걸어 행동을 쓰라"의 입력이다. 지금은 description
  평균 95자 한 문장뿐이라 걸 곳이 없어 LLM 이 일반론을 쓴다.
- **`district` 신설(선택)** — 주소의 자치구. `area`(9권역 칩) 산출 근거이자 검증 축이다.
  서울시 문화공간 데이터(GNGU)에는 있고 기존 807곳에는 없어서 선택 필드로 둔다.
- `purpose_tags` 는 남기되 **`데이트` 를 뺀다** — 807곳 중 587곳(72.7%)에 붙어 있어
  필터로서 정보량이 0 이었다. 데이트는 장소 속성이 아니라 정렬 렌즈로 다룬다
  (`PURPOSE_RULES["데이트"].query` 는 임베딩 검색에 그대로 쓰이므로 의미 검색은 유지된다).

## 삭제한 필드와 근거 (v1 에서)
- `place_id`/`source`/`lang`/`tags`/`curated` : 읽는 코드가 없다.
- `ragText` : 나머지 필드의 조립물. `ingest._embed_text` 가 만든다.
- `hours_known` : `hours: null`(미확인)로 표현한다.
- `isFilming`+`contentTitle` → `filming`(제목 문자열). 두 필드의 참/거짓이 100% 일치했다.
- `areaName` → `congestion_key`. displayName 과 겹친 게 아니라 역할이 다르다(citydata 지점명).

## 지우면 안 되는 필드
- `hours_note`(옛 `operatingRaw`) : 메타로는 안 실리지만 임베딩 본문에 실려 LLM 까지 간다.
  581건 중 562건이 `hours` 정수 두 개로 복원 불가하고(휴관일 102건·분 단위 162건),
  "매주 월요일 휴관" 은 여기에만 있다.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PLACES = Path("data/embed/places.json")

# 출력 키 순서 — 정체 → 분류 → 개인화 축 → 시간 → 서사 → 부가. 눈으로 훑을 순서다.
KEY_ORDER = [
    "name", "district", "area", "lat", "lng",
    "category", "purpose_tags",
    "indoor", "night_ok", "stay_min", "energy",
    "hours", "hours_note",
    "description", "highlights",
    "congestion_key", "same_place_group", "filming",
]
# v1 잔재 — 보이면 버린다 (마이그레이션 리포트용)
DROPPED = ["place_id", "source", "lang", "tags", "ragText", "curated",
           "hours_known", "displayName", "areaName", "operatingHours",
           "operatingRaw", "isFilming", "contentTitle", "coarse_category", "coarse"]

AREAS = {"종로·중구", "강북·성북", "홍대·마포", "용산·이태원", "여의도·영등포",
         "강남·서초", "성수·건대", "잠실·송파", "관악·사당"}
# 종류 축 6종 — 코스 안 다양성 규칙(_SELECT_PROMPT)이 이 값을 센다. 빈 값·오타를
# 통과시키면 "섞어라"가 전부 '기타'가 되어 조용히 무력화된다.
CATEGORIES = {"자연", "역사", "문화", "쇼핑", "명소", "체험"}
# 목적 축 — 검색 필터(pt_*)로 쓰는 7종.
# "데이트"는 한 번 뺐다가 되살렸다 — 원래 99.3%(1651/1663)에 붙어 있어 필터 기능이 없었는데,
# 그건 태그가 틀려서가 아니라 **아무 데나 붙였기** 때문이다. 묘지·추모공원·구민체육센터·
# 주민센터처럼 데이트로 가지 않는 곳을 빼면 실제 축이 된다. 다만 이름 규칙으로는 못 거른다
# (실측: 규칙으로 245곳만 걸리고 그중 '별마당 도서관'·'동해문화예술관' 같은 오탐이 섞였다).
# 그래서 소개문을 보고 장소별로 판단해 붙였고, 아래 유병률 상한이 그 결과를 검증한다.
PURPOSES = {"자연·힐링", "문화·예술", "관광 명소", "체험·놀거리", "핫플레이스", "쇼핑", "데이트"}
ENERGY = {"calm", "lively"}
# 한 축의 유병률이 이 이상이면 필터로서 정보량이 없다 — 검색 필터로 쓰는 태그만 검사한다.
MAX_TAG_PREVALENCE = 0.60
# 유병률 검사에서 빼는 목적 — 필터로 쓰지 않으므로 높아도 무방하다.
# "데이트"는 정확히 붙여도 75%가 해당된다(공원·고궁·전시관·쇼핑몰이 다 데이트 장소다).
# 필터에서 빼는 대신 태그는 정확히 유지한다 — 임베딩 본문·프롬프트에 실리는 값이라서.
# app/features/course/schema.py:FILTERABLE_PURPOSES 와 짝이다.
NON_FILTER_PURPOSES = {"데이트"}


def _hours(p: dict) -> dict | None:
    """운영시간. 모르면 None — 이게 곧 v1 의 hours_known=False 다.

    hours_known=True 이면서 (0,24)인 135건은 "진짜 상시"라 값을 유지한다.
    False 인 226건은 (0,24)로 채워져 있을 뿐이므로 None 으로 지운다 — 이 구분이
    사라지면 시간창 필터가 문 닫은 곳을 통과시키고, LLM 이 "상시개방"이라 쓴다.
    """
    if "hours" in p:                                    # 이미 v2
        return p["hours"]
    if not p.get("hours_known"):
        return None
    h = p.get("operatingHours") or {}
    return {"start": int(h.get("start", 0)), "end": int(h.get("end", 24))}


def normalize(p: dict) -> dict:
    out = {
        "name": (p.get("name") or p.get("displayName", "")).strip(),
        "area": p["area"],
        "lat": round(float(p["lat"]), 6),
        "lng": round(float(p["lng"]), 6),
        # v1 의 coarse_category 가 v2 의 category 다 (27종 세부 분류는 버린다)
        "category": p.get("category") if p.get("category") in CATEGORIES
        else p.get("coarse_category") or p.get("coarse", ""),
        "purpose_tags": [t for t in p.get("purpose_tags", []) if t in PURPOSES],
        "hours": _hours(p),
        "description": p["description"].strip(),
    }
    if district := (p.get("district") or "").strip():
        out["district"] = district
    # 개인화 축 — 없으면 키를 만들지 않아 "아직 미태깅"이 그대로 보인다.
    for axis in ("indoor", "night_ok"):
        if isinstance(p.get(axis), bool):
            out[axis] = p[axis]
    if isinstance(p.get("stay_min"), int):
        out["stay_min"] = p["stay_min"]
    if p.get("energy") in ENERGY:
        out["energy"] = p["energy"]
    if highlights := [h.strip() for h in p.get("highlights", []) if h.strip()]:
        out["highlights"] = highlights
    # 운영시간 원문 — 휴관일·계절제·분 단위는 여기에만 있다. 공백을 접어 한 줄로 만들되
    # 내용은 손대지 않는다 (해석해서 요약하는 건 LLM 단계의 일이다).
    if note := re.sub(r"\s+", " ", (p.get("hours_note") or p.get("operatingRaw") or "")).strip():
        out["hours_note"] = note
    # 빈 값은 키를 만들지 않는다 (읽을 때 잡음이 되고, ingest 는 get 으로 받는다)
    if key := (p.get("congestion_key") or p.get("areaName") or "").strip():
        out["congestion_key"] = key
    if group := (p.get("same_place_group") or "").strip():
        out["same_place_group"] = group
    if title := (p.get("filming") or p.get("contentTitle") or "").strip():
        out["filming"] = title
    return {k: out[k] for k in KEY_ORDER if k in out}


def validate(places: list[dict], *, strict: bool) -> list[str]:
    """구조 계약 검사. 하나라도 깨지면 쓰지 않는다 (조용히 망가진 데이터가 최악이다)."""
    errs: list[str] = []
    for n, c in Counter(p["name"] for p in places).items():
        if c > 1:
            errs.append(f"이름 중복: {n} ({c}건) — ingest 의 doc id·화이트리스트 키가 겹친다")
    for p in places:
        n = p["name"]
        if not n:
            errs.append(f"이름 없음: {p}")
        if p["area"] not in AREAS:
            errs.append(f"{n}: 권역 칩 아님 ({p['area']!r})")
        if p["category"] not in CATEGORIES:
            errs.append(f"{n}: 종류 6종 아님 ({p['category']!r})")
        if not 37.4 <= p["lat"] <= 37.72 or not 126.7 <= p["lng"] <= 127.2:
            errs.append(f"{n}: 서울 좌표 밖 (lat={p['lat']}, lng={p['lng']}) — 위경도 뒤바뀜 의심")
        if p["hours"] and not 0 <= p["hours"]["start"] < 24:
            errs.append(f"{n}: 운영 시작 시각 범위 밖 ({p['hours']})")
        if not p["description"]:
            errs.append(f"{n}: description 없음 — 임베딩 본문이 이름뿐이 된다")
        if "[" in p["description"] or "]" in p["description"]:
            errs.append(f"{n}: description 에 대괄호 — 임베딩 잡음")
        if strict:
            errs += _strict_errors(p)

    # 유병률 상한 — 한 태그가 과반을 넘으면 그 필터는 아무것도 안 거른다
    tot = len(places) or 1
    for t, c in Counter(t for p in places for t in p["purpose_tags"]).most_common():
        if t not in NON_FILTER_PURPOSES and c / tot > MAX_TAG_PREVALENCE:
            errs.append(f"목적 태그 '{t}' 유병률 {c / tot:.0%} — 필터로서 정보량이 없다")
    return errs


def _strict_errors(p: dict) -> list[str]:
    """--strict: 태깅이 끝난 파일에만 통과해야 하는 검사."""
    n, errs = p["name"], []
    for axis in ("indoor", "night_ok", "stay_min", "energy"):
        if axis not in p:
            errs.append(f"{n}: 개인화 축 '{axis}' 미태깅 — 값을 채워야 한다")
    if not 15 <= p.get("stay_min", 0) <= 240:
        errs.append(f"{n}: stay_min 범위 밖 ({p.get('stay_min')}) — 15~240분")
    if len(p.get("highlights", [])) < 2:
        errs.append(f"{n}: highlights 2개 미만 — activities 를 걸 명사가 없다")
    if not p["purpose_tags"]:
        errs.append(f"{n}: purpose_tags 없음 — 목적 칩 어디에도 안 걸린다")
    return errs


def report(places: list[dict], raw: list[dict]) -> None:
    """남은 것·버린 것·밀도. 내용 결함은 LLM 단계 몫이라 여기선 세기만 한다."""
    tot = len(places) or 1
    known = sum(1 for p in places if p["hours"])
    print(f"장소 {len(places)}건 · 운영시간 확인 {known} · 미확인 {len(places) - known}")

    tagged = sum(1 for p in places if "energy" in p)
    print(f"개인화 축 태깅: {tagged}/{len(places)}"
          f"{'  ← 미태깅 장소 있음' if tagged < len(places) else ''}")
    for k in ("highlights", "district", "hours_note", "congestion_key",
              "same_place_group", "filming"):
        print(f"  {k}: {sum(1 for p in places if k in p)}건")

    print("\n종류(category) — 다양성 규칙이 세는 축:")
    for c, n in Counter(p["category"] for p in places).most_common():
        print(f"  {c:4s} {n:5d} {n / tot * 100:5.1f}%")

    print("목적 태그 — 검색 필터 축:")
    for t, n in Counter(t for p in places for t in p["purpose_tags"]).most_common():
        flag = "  ← 무필터" if n / tot > MAX_TAG_PREVALENCE else ""
        print(f"  {t:8s} {n:5d} {n / tot * 100:5.1f}%{flag}")

    print("\n권역 × 종류 밀도 (칸당 6곳이 목표 — 하루 5곳 + 중복 제거 여유):")
    cats = ["자연", "역사", "문화", "쇼핑", "명소", "체험"]
    grid: dict[str, Counter] = defaultdict(Counter)
    for p in places:
        grid[p["area"]][p["category"]] += 1
    print("              " + "".join(f"{c:>6s}" for c in cats) + "     합")
    starved = 0
    for a in sorted(AREAS):
        row = [grid[a][c] for c in cats]
        starved += sum(1 for v in row if v < 6)
        print(f"{a:12s}" + "".join(f"{v:6d}" for v in row) + f"{sum(row):7d}")
    print(f"6곳 미만 칸: {starved}/{len(AREAS) * len(cats)}")

    if dropped := {k for p in raw for k in p if k in DROPPED}:
        print(f"\n버린 v1 필드: {', '.join(sorted(dropped))}")


def run(*, check: bool, strict: bool) -> int:
    raw = json.loads(PLACES.read_text(encoding="utf-8"))
    places = [normalize(p) for p in raw]

    if errs := validate(places, strict=strict):
        print(f"검증 실패 {len(errs)}건 — 파일을 쓰지 않는다:")
        for e in errs[:20]:
            print(f"  ! {e}")
        if len(errs) > 20:
            print(f"  … 그 외 {len(errs) - 20}건")
        return 1

    report(places, raw)
    if check:
        print("\n--check: 파일을 쓰지 않았다.")
        return 0

    before = len(PLACES.read_bytes())
    PLACES.write_text(json.dumps(places, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    after = len(PLACES.read_bytes())
    print(f"\n{PLACES}: {before:,}B → {after:,}B ({(1 - after / before) * 100:+.0f}%)")
    print("→ page_content 가 바뀌므로 재인제스트 필요: uv run python -m scripts.run_ingest --reset")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="진단만 하고 파일은 안 건드린다")
    ap.add_argument("--strict", action="store_true", help="개인화 축·highlights 까지 필수 검사")
    a = ap.parse_args()
    raise SystemExit(run(check=a.check, strict=a.strict))
