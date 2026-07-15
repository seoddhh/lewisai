from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.geo import region_of
from app.core.vectorstore import get_vectorstore


def _place_docs(places: list[dict]) -> list[Document]:
    """통합 장소 데이터 → 청크.

    한 레코드 = 한 청크. source 별 aspect 로 구분한다:
      - seoul_places → summary  (일반 장소 서사)
      - seoul_spots  → access   (요금·교통·운영시간 중심 명소)
      - kcontent     → scene    (드라마/영화/MV 촬영 장면 서사)
    임베딩 본문은 빌드 단계에서 만든 ragText 를 우선 사용한다.
    lat/lng 는 코스 검색의 앵커 반경 클러스터링에 쓰인다 (retrieve_node).
    region 메타는 표시/디버깅용으로 남겨둔다 (지리 필터는 반경 방식으로 대체됨).
    """
    docs: list[Document] = []
    for i, p in enumerate(places):
        place_id = p.get("areaName") or p["displayName"]
        lat, lng = float(p["lat"]), float(p["lng"])
        # 운영시간 — 프론트 라우팅의 시간창 제약·표시용. 기본 상시(0~24).
        hours = p.get("operatingHours") or {}
        op_start = int(hours.get("start", 0))
        op_end = int(hours.get("end", 24))
        aspect = p.get("aspect", "summary")
        meta = {
            "doc_type": "place",
            "source": p.get("source", "seoul_places"),
            "place_id": place_id,
            "display_name": p["displayName"],
            "area_name": p.get("areaName", ""),  # 혼잡도 매칭 (kcontent/spots 는 "")
            "category": p.get("category", ""),
            "region": region_of(lat, lng),
            "lat": lat,
            "lng": lng,
            "op_start": op_start,
            "op_end": op_end,
            "aspect": aspect,
            # K-콘텐츠 촬영지 필터 (search_kcontent_filming_spots 와 동일 키)
            "is_filming": bool(p.get("isFilming", False)),
            "content_title": p.get("contentTitle", ""),
            "tags": ",".join(p.get("tags", [])),
        }
        text = p.get("ragText") or (
            f"[{p['displayName']}] ({p.get('category','')}) {p.get('description','')}"
        )
        docs.append(Document(page_content=text, metadata=meta, id=f"{place_id}::{aspect}::{i}"))
    return docs


def _course_docs(courses: list[dict]) -> list[Document]:
    docs: list[Document] = []
    for c in courses:
        stops = "; ".join(
            f"{s['name']}({s.get('preview','')})" for s in c.get("stops", [])
        )
        text = (
            f"[코스: {c['title']}] {c.get('subtitle','')}. {c.get('description','')} "
            f"동선: {stops}"
        )
        tags = c.get("tags", [])
        meta = {
            "doc_type": "course",
            "place_id": c["id"],
            "display_name": c["title"],
            "category": c.get("category", ""),
            "aspect": "course",
            # Chroma 메타데이터는 스칼라만 허용 → 태그는 콤마 문자열로
            "tags": ",".join(tags),
            # K-컨텐츠 촬영지 여부 (search_kcontent_filming_spots 필터 키)
            "is_filming": c.get("category") == "서울배경 컨텐츠"
            or any("촬영지" in t or "성지순례" in t for t in tags),
        }
        docs.append(Document(page_content=text, metadata=meta, id=f"course::{c['id']}"))
    return docs


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  ! 없음, 건너뜀: {path}")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _build_docs() -> list[Document]:
    """항상 같은 순서(장소 → 코스)로 문서를 만든다 — 배치 인덱스가 실행마다 동일하도록."""
    s = get_settings()
    places = _load(s.places_json)
    courses = _load(s.courses_json)
    return _place_docs(places) + _course_docs(courses)


def _reset_store() -> None:
    """기존 Chroma 컬렉션을 비운다. 이후 get_vectorstore() 가 새 컬렉션을 만든다."""
    vs = get_vectorstore()
    try:
        vs.delete_collection()
        print("컬렉션 초기화 완료.")
    except Exception as err:  # noqa: BLE001 — 컬렉션이 없어도 진행
        print(f"초기화 건너뜀({err}).")
    get_vectorstore.cache_clear()


def _add_batch(vs, batch: list[Document], idx: int, total: int) -> int:
    """이미 존재하는 id 는 건너뛰고 새 문서만 임베딩. 실제로 넣은 개수를 반환."""
    ids = [d.id for d in batch]
    existing = set(vs.get(ids=ids).get("ids", []))
    fresh = [d for d in batch if d.id not in existing]
    skipped = len(batch) - len(fresh)
    if fresh:
        vs.add_documents(fresh, ids=[d.id for d in fresh])
    print(f"  배치 {idx + 1}/{total}: 신규 {len(fresh)} · 스킵 {skipped}")
    return len(fresh)


def run(*, reset: bool = False, size: int = 70, sleep: float = 60.0,
        batch: int | None = None) -> None:
    docs = _build_docs()
    if not docs:
        print("인제스트할 문서가 없습니다. data/raw/*.json 을 먼저 준비하세요.")
        return

    if reset:
        _reset_store()

    batches = [docs[i:i + size] for i in range(0, len(docs), size)]
    vs = get_vectorstore()

    # 특정 배치만 수동 실행 (분당 쿼터를 사람이 직접 통제하고 싶을 때)
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
        if i < len(batches) - 1:  # 마지막 배치 뒤엔 대기 불필요
            time.sleep(sleep)
    print(f"인제스트 완료: 신규 {added} · 전체 {len(docs)} 청크")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG 배치 인제스트")
    p.add_argument("--reset", action="store_true", help="시작 전 기존 컬렉션 초기화")
    p.add_argument("--size", type=int, default=70, help="배치당 청크 수 (기본 70)")
    p.add_argument("--sleep", type=float, default=60.0, help="배치 사이 대기 초 (기본 60)")
    p.add_argument("--batch", type=int, default=None, help="특정 배치(0-기반)만 실행")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(reset=args.reset, size=args.size, sleep=args.sleep, batch=args.batch)
