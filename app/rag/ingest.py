"""장소 데이터(places.json)를 벡터 DB(Chroma)에 넣는 스크립트.

한 장소가 문서 하나가 된다. 검색에 쓸 본문을 만들고, 필터에 쓸 정보를 메타데이터로 붙인다.

    python -m app.rag.ingest --reset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.vectorstore import get_vectorstore
from app.features.course.schema import PURPOSE_SLUGS


def _embed_text(p: dict, hours_label: str) -> str:
    """장소 하나를 검색용 본문 텍스트로 만든다.

    이 텍스트가 검색에 걸리느냐 마느냐를 결정하고, 나중에 AI 가 장소를 고를 때 읽는 설명이
    되기도 한다. 그래서 아는 정보는 다 넣는다.

    `라벨: 값` 을 한 줄씩 쓴다. 한 줄에 `·` 로 이어 붙이면, 값 자체에 이미 중점이 들어
    있어서(`종로·중구`, `공원·자연`) 어디까지가 한 값인지 알아볼 수 없다.

    운영시간은 원문 문장을 그대로 넣는다. "매주 월요일 휴관" 같은 정보는 시작·종료
    시각 두 개로는 표현할 수 없는데, 월요일 코스에 휴관인 미술관을 넣지 않으려면 필요하다.
    """
    lines = [
        p["name"],
        f"종류: {p['category']}",
        f"권역: {p.get('district') or p['area']}",
        f"소개: {p['description']}",
    ]
    if highlights := p.get("highlights"):
        lines.append(f"있는 것: {', '.join(highlights)}")
    if purposes := p.get("purpose_tags"):
        lines.append(f"어울리는 목적: {', '.join(purposes)}")
    # 검색어는 "조용한 산책 자연 공원…" 처럼 상황을 말하는 문장으로 들어오는데,
    # 본문에 분류 명사만 있으면 안 걸린다. 그래서 태그를 사람이 쓸 법한 말로 풀어 넣는다.
    if situations := _situation_words(p):
        lines.append(f"찾는 상황: {', '.join(situations)}")
    lines.append(hours_label)
    if title := p.get("filming"):
        lines.append(f"K-콘텐츠 촬영지: {title}")
    return "\n".join(lines)


def _situation_words(p: dict) -> list[str]:
    """장소 태그를 사람이 검색할 법한 말로 바꾼다. 태그가 없는 장소는 빈 목록."""
    if "energy" not in p:
        return []
    words = ["비 오는 날 실내" if p.get("indoor") else "야외 활동"]
    if p.get("night_ok"):
        words.append("밤에 가기 좋은")
    if p["energy"] == "calm":
        words.append("조용한")
    else:
        words.append("활기찬")
    stay = p.get("stay_min", 60)
    words.append("잠깐 들르기" if stay <= 30 else "반나절 코스" if stay >= 120 else "한두 시간")
    return words


def _place_docs(places: list[dict]) -> list[Document]:
    """장소 목록을 벡터 DB 에 넣을 문서 목록으로 바꾼다. 장소 하나가 문서 하나다.

    파일의 필드 이름과 메타데이터 키가 항상 같지는 않다. 메타데이터 키는 코스 코드가
    읽는 이름이라 따로 정해져 있다(파일의 congestion_key → 메타의 area_name).

    꼭 있어야 하는 값은 `p["..."]` 로 꺼낸다. `.get()` 을 쓰면 파일 필드 이름이 바뀌었을 때
    조용히 빈 값이 들어가 버린다. 그런 경우엔 에러를 내고 죽는 게 낫다.
    """
    docs: list[Document] = []
    for i, p in enumerate(places):
        # 이름 앞뒤 공백은 여기서 지운다. 원본에 공백이 섞여 있는데, 그대로 넣으면
        # 나중에 AI 가 답한 이름과 안 맞아서 그 장소가 조용히 사라진다.
        display_name = p["name"].strip()
        lat, lng = float(p["lat"]), float(p["lng"])
        # 운영시간을 모르는 장소는 일단 0~24시로 채운다. 대신 아래 hours_known 으로 구분한다.
        hours = p.get("hours")
        op_start = int(hours["start"]) if hours else 0
        op_end = int(hours["end"]) if hours else 24
        aspect = "summary"
        meta = {
            "doc_type": "place",
            "place_id": display_name,
            "display_name": display_name,
            # 서울시 실시간 혼잡도 API 가 쓰는 지점 이름. 등록된 곳이 적어서 대개 비어 있다
            "area_name": p.get("congestion_key", ""),
            # 장소 종류. 하루에 비슷한 곳만 몰리지 않게 하는 기준이 된다
            "category": p["category"],
            "area": p["area"],
            "district": p.get("district", ""),
            "lat": lat,
            "lng": lng,
            "op_start": op_start,
            "op_end": op_end,
            # 위 0~24시가 진짜 24시간 영업인지, 그냥 모르는 건지 구분해 준다.
            # 이게 없으면 "모르는 곳"과 "항상 여는 곳"을 구별할 수 없다.
            "hours_known": hours is not None,
            # 좌표가 같은(사실상 같은 자리인) 장소들의 묶음 이름. 예: 광화문광장과 해치마당.
            # 한 코스에 둘 다 들어가지 않게 하는 데 쓴다. 묶음이 없으면 빈 문자열.
            "same_place_group": p.get("same_place_group", ""),
            "aspect": aspect,
            # 드라마·영화 촬영지인지
            "is_filming": bool(p.get("filming")),
            "content_title": p.get("filming", ""),
            "purpose_tags": ",".join(p.get("purpose_tags", [])),
            # 소개문. 본문에서 잘라 쓰지 않고 여기서 바로 읽을 수 있게 따로 둔다
            # (이러면 표현을 바꿔도 다시 인제스트할 필요가 없다)
            "description": p["description"],
            # 운영시간 원문. 휴관일이나 계절별 시간처럼 숫자 두 개로 못 담는 내용이 들어 있다
            "hours_note": p.get("hours_note", ""),
            # 그곳에 뭐가 있는지. 할 일을 쓸 때의 재료가 된다
            # (Chroma 메타데이터는 목록을 못 받아서 콤마로 이어 붙인다)
            "highlights": ",".join(p.get("highlights", [])),
        }
        # 아래 값들은 동반자·목적에 맞춰 순서를 조정할 때 쓴다.
        # 태그가 없는 장소는 키 자체를 안 넣는다. 그래야 필터에 안 걸린다.
        for axis in ("indoor", "night_ok"):
            if isinstance(p.get(axis), bool):
                meta[axis] = p[axis]
        if isinstance(p.get("stay_min"), int):
            meta["stay_min"] = p["stay_min"]
        if p.get("energy"):
            meta["energy"] = p["energy"]
        # 목적 태그는 목적마다 pt_<이름> 키를 따로 만든다.
        # Chroma 가 콤마로 이어 붙인 문자열 안을 뒤지지 못해서, 검색으로 거르려면 이렇게 해야 한다.
        for tag in p.get("purpose_tags", []):
            if slug := PURPOSE_SLUGS.get(tag):
                meta[f"pt_{slug}"] = True
        # 본문에 쓸 운영시간 문구. 위 메타데이터와 같은 자리에서 만들어야 둘이 어긋나지 않는다.
        # 원문이 있으면 원문을 쓴다. 휴관일 같은 건 숫자 두 개로 표현할 수 없기 때문이다.
        hours_label = (
            f"운영: {p['hours_note']}" if p.get("hours_note")
            else f"운영: {op_start}시~{op_end}시" if hours and (op_start, op_end) != (0, 24)
            else "운영: 상시(24시간)" if hours else "운영시간 미확인"
        )
        text = _embed_text(p, hours_label)
        docs.append(Document(page_content=text, metadata=meta,
                             id=f"{display_name}::{aspect}::{i}"))
    return docs


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  ! 없음, 건너뜀: {path}")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def build_docs() -> list[Document]:
    """넣을 문서 전체를 만든다. 파일 순서를 그대로 따라서, 몇 번 돌려도 배치가 똑같이 나뉜다.

    문서 id 와 메타데이터를 만드는 곳은 여기 하나뿐이다. 인덱스를 손대는 도구가
    생기면 반드시 이 함수를 불러 쓸 것 — 규칙이 두 군데로 갈라지면 엉뚱한 문서를 건드린다.
    """
    s = get_settings()
    return _place_docs(_load(s.places_json))


def _reset_store() -> None:
    """기존 Chroma 컬렉션을 비운다. 이후 get_vectorstore() 가 새 컬렉션을 만든다."""
    vs = get_vectorstore()
    try:
        vs.delete_collection()
        print("컬렉션 초기화 완료.")
    except Exception as err:  # noqa: BLE001 — 지울 컬렉션이 없어도 그냥 진행하면 된다
        print(f"초기화 건너뜀({err}).")
    get_vectorstore.cache_clear()


def _add_batch(vs, batch: list[Document], idx: int, total: int) -> int:
    """배치 하나를 넣는다. 이미 들어 있는 문서는 건너뛰고 새 것만 임베딩한다."""
    ids = [d.id for d in batch]
    existing = set(vs.get(ids=ids).get("ids", []))
    fresh = [d for d in batch if d.id not in existing]
    skipped = len(batch) - len(fresh)
    if fresh:
        vs.add_documents(fresh, ids=[d.id for d in fresh])
    print(f"  배치 {idx + 1}/{total}: 신규 {len(fresh)} · 스킵 {skipped}")
    return len(fresh)


def _default_sleep() -> float:
    """배치 사이에 얼마나 쉴지 정한다.

    제미나이는 무료 쿼터가 빡빡해서 안 쉬면 중간에 막힌다.
    ollama(로컬)와 솔라(2,000 RPM · 하루 한도 없음)는 쉴 필요가 없다.
    여기서 괜히 쉬면 대기 시간이 작업 시간보다 길어진다.
    """
    s = get_settings()
    return 0.0 if (s.use_ollama_embeddings or s.use_upstage_embeddings) else 60.0


def run(*, reset: bool = False, size: int = 70, sleep: float | None = None,
        batch: int | None = None) -> None:
    docs = build_docs()
    if not docs:
        print(f"인제스트할 문서가 없습니다. {get_settings().places_json} 를 먼저 준비하세요.")
        return

    if sleep is None:
        sleep = _default_sleep()

    # 모델과 컬렉션을 항상 같이 찍는다. 모델만 바꾸고 컬렉션 이름을 안 바꾸면 색인이 터지는데,
    # 그때 이 한 줄만 봐도 원인을 바로 알 수 있다.
    s = get_settings()
    print(f"임베딩: {s.embedding_provider} / {s.active_embedding_model} "
          f"→ 컬렉션 {s.chroma_collection}")

    if reset:
        _reset_store()

    batches = [docs[i:i + size] for i in range(0, len(docs), size)]
    vs = get_vectorstore()

    # 배치 하나만 골라 돌리기. 쿼터를 사람이 직접 조절하며 넣고 싶을 때 쓴다
    if batch is not None:
        if not 0 <= batch < len(batches):
            print(f"배치 인덱스 범위 밖: 0~{len(batches) - 1} 중 {batch}")
            return
        _add_batch(vs, batches[batch], batch, len(batches))
        return

    print(f"총 {len(docs)} 청크 → {len(batches)} 배치 (배치당 {size}, 사이 대기 {sleep:.0f}s)")
    added = 0
    for i, b in enumerate(batches):
        added += _add_batch(vs, b, i, len(batches))
        if i < len(batches) - 1:  # 마지막 배치 뒤에는 쉴 필요가 없다
            time.sleep(sleep)
    print(f"인제스트 완료: 신규 {added} · 전체 {len(docs)} 청크")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG 배치 인제스트")
    p.add_argument("--reset", action="store_true", help="시작 전 기존 컬렉션 초기화")
    p.add_argument("--size", type=int, default=70, help="배치당 청크 수 (기본 70)")
    p.add_argument("--sleep", type=float, default=None,
                   help="배치 사이 대기 초 (기본: gemini 60 · ollama 0)")
    p.add_argument("--batch", type=int, default=None, help="특정 배치(0-기반)만 실행")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(reset=args.reset, size=args.size, sleep=args.sleep, batch=args.batch)
