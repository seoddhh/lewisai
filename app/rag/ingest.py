from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.vectorstore import get_vectorstore
from app.features.course.schema import PURPOSE_SLUGS


def _place_docs(places: list[dict]) -> list[Document]:
    """통합 장소 데이터 → 청크.

    한 레코드 = 한 청크. source 별 aspect 로 구분한다:
      - seoul_places → summary  (일반 장소 서사)
      - seoul_spots  → access   (요금·교통·운영시간 중심 명소)
      - kcontent     → scene    (드라마/영화/MV 촬영 장면 서사)
    임베딩 본문은 빌드 단계에서 만든 ragText 를 우선 사용한다.
    lat/lng 는 코스 검색의 앵커 반경 클러스터링에 쓰인다 (retrieve_node).
    area(9권역 칩)는 주소 자치구 기준으로 빌드 단계에서 확정된다.
    """
    docs: list[Document] = []
    for i, p in enumerate(places):
        # 이름은 색인 진입점에서 정규화 — 원천에 앞뒤 공백(일반/탭/전각 U+3000)이 섞여 들어온다.
        # 오염된 이름이 색인되면 select 의 화이트리스트 매칭이 실패해 장소가 조용히 유실된다.
        display_name = p["displayName"].strip()
        place_id = (p.get("areaName") or display_name).strip()
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
            "display_name": display_name,
            "area_name": p.get("areaName", ""),  # 혼잡도 매칭 (없으면 "")
            "category": p.get("category", ""),
            # 코스 다양성·권역 분산 축 (embed 세트가 주소 자치구 기준으로 실어 보낸다)
            "coarse_category": p.get("coarse_category", ""),
            "area": p.get("area", ""),
            "lat": lat,
            "lng": lng,
            "op_start": op_start,
            "op_end": op_end,
            # 운영시간을 실제로 아는지. False 면 (0,24)로 채워져 있을 뿐 "상시개방"이 아니다 —
            # 시간창 필터가 (0,24)를 무조건 통과시키므로, 아는 시간이 안 맞는 곳만 걸러내려면
            # 이 플래그로 "미확인"과 "진짜 상시"를 구분해야 한다.
            "hours_known": bool(p.get("hours_known", False)),
            # 좌표가 동일한(=사실상 같은 자리) 장소 묶음. 한 코스에 하나만 넣기 위한 근거.
            # 예: 광화문광장 ⟷ 해치마당. 빈 문자열이면 그룹 없음.
            "same_place_group": p.get("same_place_group", ""),
            "aspect": aspect,
            # K-콘텐츠 촬영지 필터 (search_kcontent_filming_spots 와 동일 키)
            "is_filming": bool(p.get("isFilming", False)),
            "content_title": p.get("contentTitle", ""),
            "tags": ",".join(p.get("tags", [])),
            "purpose_tags": ",".join(p.get("purpose_tags", [])),
        }
        # 목적 축을 불리언 메타로 편다 — Chroma where 는 콤마 문자열 부분일치를 못 하므로
        # 목적당 pt_<slug> 필드를 둔다. 해당 없는 목적은 키 자체를 넣지 않아
        # {"pt_x": True} 필터에 걸리지 않는다 (retrieve_node 의 목적 필터).
        for tag in p.get("purpose_tags", []):
            if slug := PURPOSE_SLUGS.get(tag):
                meta[f"pt_{slug}"] = True
        text = p.get("ragText") or (
            f"[{p['displayName']}] ({p.get('category','')}) {p.get('description','')}"
        )
        docs.append(Document(page_content=text, metadata=meta, id=f"{place_id}::{aspect}::{i}"))
    return docs


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  ! 없음, 건너뜀: {path}")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _build_docs() -> list[Document]:
    """정제된 임베딩 장소 세트만 인제스트한다 (배치 인덱스가 실행마다 동일하도록 순서 고정).

    테마 코스(theme_courses)는 서울로(strangemap) 정적 데이터라 RAG 에 넣지 않는다 —
    코스 문서는 조회하는 곳도 없었다(retrieve_node 는 doc_type=place 만 쓴다).
    """
    s = get_settings()
    return _place_docs(_load(s.places_json))


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
