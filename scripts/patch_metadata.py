"""메타데이터 패치 — places.json 의 메타를 Chroma 인덱스에 반영한다 (임베딩 호출 0회).

    PYTHONPATH=. uv run python scripts/patch_metadata.py                  # dry-run (기본)
    PYTHONPATH=. uv run python scripts/patch_metadata.py --apply
    PYTHONPATH=. uv run python scripts/patch_metadata.py --apply --prune  # 고아 문서까지 삭제

## 왜 필요한가

런타임(`retrieve` → `_cand`)은 `data/embed/places.json` 을 읽지 않는다. Chroma 에 **복사된**
메타데이터(lat/lng/category/area/pt_*/운영시간)를 읽는다. 그런데 인제스트는 쿼터를 아끼려고
**이미 존재하는 id 를 건너뛰므로**(`ingest._add_batch`), 파일에서 좌표를 고친 뒤
`run_ingest` 를 다시 돌려도 "신규 0건"으로 끝나고 아무것도 갱신되지 않는다.
전량 재인제스트(`--reset`)는 807건을 다시 임베딩해 11분 + 임베딩 쿼터를 태운다.

메타데이터는 임베딩과 무관하다(벡터는 `ragText` 하나로만 계산된다). 그래서 벡터를 그대로 두고
값만 갱신하면 된다 — 이 스크립트가 그 일을 한다.

## 어떻게

  - "파일이 기대하는 상태"는 `ingest.build_docs()` 로 만든다. 인제스트와 **같은 코드**라
    id·메타 계산 규칙이 갈라지지 않는다.
  - 달라진 키만 `collection.update()` 로 보낸다. Chroma 의 update 는 **병합**이라 준 키만
    바뀌고 나머지는 살아 있으며, 값 `None` 은 그 키를 **삭제**한다(목적 태그가 빠진 경우).
  - `documents`/`embeddings` 는 **절대 넘기지 않는다.** 넘기면 컬렉션에 붙은 임베딩 함수가
    개입해 벡터가 깨질 수 있다. 그래서 Chroma 도 임베딩 함수 없이 raw 클라이언트로 연다
    (GOOGLE_API_KEY 없이도 돌고, 실수로 임베딩이 호출될 여지가 없다).
  - `ragText` 가 바뀐 문서는 벡터를 다시 계산해야 하므로 **손대지 않고 보고만** 한다.

## 반영 범위

인덱스는 도커 이미지에 구워 배포된다 — 패치한 뒤 **이미지를 다시 빌드해야** 프로덕션에 반영된다.
"""
from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.rag.ingest import build_docs

_CHUNK = 500          # update 한 번에 보낼 문서 수
_SHOW_DEFAULT = 30    # dry-run 에서 자세히 보여줄 장소 수


def _same(a, b) -> bool:
    """Chroma 에서 읽은 값 ↔ 파일에서 만든 값의 동등 비교.

    bool 은 sqlite 를 왕복하며 int 로 보일 수 있고, float 은 표현 오차가 날 수 있어
    타입에 관대하게 비교한다 (엄격 비교로 하면 매번 전 건이 '변경'으로 잡힌다).
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def _diff(stored: dict, expected: dict) -> dict:
    """변경할 키만 추린다. 파일에 없어진 키는 None(=삭제 지시)으로 담는다."""
    changes: dict = {}
    for key, want in expected.items():
        if not _same(stored.get(key), want):
            changes[key] = want
    for key in stored:
        if key not in expected:
            changes[key] = None
    return changes


def _open_collection():
    """임베딩 함수 없이 컬렉션을 연다 — 메타만 만지므로 벡터화 경로가 아예 없다."""
    import chromadb

    s = get_settings()
    client = chromadb.PersistentClient(path=s.chroma_dir)
    try:
        return client.get_collection(s.chroma_collection)
    except Exception as err:  # noqa: BLE001 — 인덱스가 없으면 패치할 대상도 없다
        sys.exit(f"컬렉션을 열 수 없습니다 ({s.chroma_dir} / {s.chroma_collection}): {err}\n"
                 f"먼저 `python -m scripts.run_ingest` 로 인덱스를 만드세요.")


def _fmt(v) -> str:
    return "(없음)" if v is None else repr(v) if isinstance(v, str) else str(v)


def main() -> None:
    args = _parse_args()
    s = get_settings()
    col = _open_collection()

    docs = build_docs()
    if not docs:
        sys.exit(f"장소 데이터가 비어 있습니다: {s.places_json}")
    expected = {d.id: d for d in docs}

    got = col.get(include=["metadatas", "documents"])
    stored_meta = dict(zip(got["ids"], got["metadatas"]))
    stored_text = dict(zip(got["ids"], got["documents"]))

    print(f"places.json {len(expected)}건 · Chroma {len(stored_meta)}건 "
          f"({s.chroma_dir} / {s.chroma_collection})\n")

    both = [i for i in expected if i in stored_meta]
    only_file = [i for i in expected if i not in stored_meta]
    only_index = [i for i in stored_meta if i not in expected]

    # 메타 변경분
    patches: dict[str, dict] = {}
    for doc_id in both:
        changes = _diff(stored_meta[doc_id] or {}, expected[doc_id].metadata)
        if changes:
            patches[doc_id] = changes

    # ragText 변경분 — 벡터를 다시 계산해야 하므로 여기서는 건드리지 않는다
    text_changed = [i for i in both
                    if (stored_text.get(i) or "") != expected[i].page_content]

    _report(patches, text_changed, only_file, only_index, expected, stored_meta,
            show=args.show)

    if not patches:
        print("메타데이터는 이미 일치합니다 — 패치할 것이 없습니다.")
    elif not args.apply:
        print(f"[dry-run] {len(patches)}건을 바꿀 수 있습니다. 실제로 적용하려면 --apply 를 붙이세요.")
    else:
        ids = list(patches)
        metas = [patches[i] for i in ids]
        for i in range(0, len(ids), _CHUNK):
            col.update(ids=ids[i:i + _CHUNK], metadatas=metas[i:i + _CHUNK])
        print(f"✅ 메타데이터 {len(ids)}건 갱신 완료 (임베딩 호출 0회)")

    if only_index:
        if args.prune and args.apply:
            col.delete(ids=only_index)
            print(f"🗑  고아 문서 {len(only_index)}건 삭제 완료")
        else:
            print(f"⚠️  인덱스에만 남은 고아 문서 {len(only_index)}건 — "
                  f"{'--apply --prune 으로 삭제' if args.prune else '--prune 을 붙이면 삭제'}합니다.")

    if only_file:
        print(f"➕ 파일에만 있는 신규 {len(only_file)}건 — "
              f"`python -m scripts.run_ingest` 로 이 건만 임베딩됩니다.")
    if text_changed:
        print(f"✏️  ragText 가 바뀐 {len(text_changed)}건은 벡터를 다시 계산해야 해 이 스크립트가 "
              f"건드리지 않았습니다. 해당 문서를 지운 뒤 재인제스트하거나(권장), 전량 "
              f"`run_ingest --reset` 을 돌리세요.")
    if patches or (only_index and args.prune and args.apply):
        print("\n⚠️  인덱스는 도커 이미지에 구워 배포된다 — 이미지를 다시 빌드해야 프로덕션에 반영된다.")


def _report(patches, text_changed, only_file, only_index, expected, stored_meta,
            *, show: int) -> None:
    """무엇이 어떻게 바뀌는지 사람이 검토할 수 있게 출력."""
    print(f"메타 변경 {len(patches)}건 · ragText 변경 {len(text_changed)}건 · "
          f"신규 {len(only_file)}건 · 고아 {len(only_index)}건\n")

    for n, (doc_id, changes) in enumerate(patches.items()):
        if n >= show:
            print(f"  … 그 외 {len(patches) - show}건 (--show 로 더 보기)\n")
            break
        name = (expected[doc_id].metadata.get("display_name") or doc_id)
        print(f"  {name}")
        for key, want in changes.items():
            before = (stored_meta[doc_id] or {}).get(key)
            arrow = "삭제" if want is None else _fmt(want)
            print(f"      {key:18} {_fmt(before)} → {arrow}")
    if patches:
        print()

    # id 에 리스트 인덱스가 들어가므로(place::aspect::i) 파일 중간에 삽입·정렬하면 뒤쪽 전체의
    # id 가 밀린다 → 같은 장소가 "신규 + 고아" 양쪽에 동시에 나타난다. 그 징후를 잡아 경고한다.
    if only_file and only_index:
        new_names = {i.split("::")[0] for i in only_file}
        old_names = {i.split("::")[0] for i in only_index}
        overlap = new_names & old_names
        if overlap:
            print(f"🚨 같은 장소가 신규·고아 양쪽에 {len(overlap)}건 걸쳐 있습니다 "
                  f"(예: {sorted(overlap)[:3]}).\n"
                  f"   places.json 중간에 삽입하거나 정렬해서 문서 id 의 인덱스가 밀린 것으로 "
                  f"보입니다. 이대로 인제스트하면 같은 장소가 중복 색인됩니다 — 장소 추가는 "
                  f"**파일 끝에 append** 하고, 이미 밀렸다면 `run_ingest --reset` 으로 "
                  f"다시 만드는 편이 안전합니다.\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="places.json 의 메타데이터를 Chroma 에 반영 (임베딩 없음)")
    p.add_argument("--apply", action="store_true",
                   help="실제로 갱신한다 (기본은 dry-run)")
    p.add_argument("--prune", action="store_true",
                   help="파일에서 사라진 고아 문서를 삭제한다 (--apply 와 함께)")
    p.add_argument("--show", type=int, default=_SHOW_DEFAULT,
                   help=f"변경 내역을 자세히 보여줄 장소 수 (기본 {_SHOW_DEFAULT})")
    return p.parse_args()


if __name__ == "__main__":
    main()
