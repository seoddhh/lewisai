"""배치 인제스트 — data/raw/*.json(장소·코스) → 청킹 → 메타데이터 → 임베딩 → Chroma upsert.

실시간 데이터(혼잡도/행사)는 절대 여기서 인제스트하지 않는다. (app/tools/ 에서 호출)
멱등성: chunk_id 를 문서 id 로 써서 재실행 시 덮어쓴다.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.vectorstore import get_vectorstore


def _region(lat: float, lng: float) -> str:
    """strangemap getRegion() 로직과 일치 (좌표 → 권역)."""
    if lng >= 127.05:
        return "강동"
    if lng < 126.94:
        return "강서"
    if lat < 37.52:
        return "강남"
    return "강북"


def _place_docs(places: list[dict]) -> list[Document]:
    docs: list[Document] = []
    for p in places:
        place_id = p.get("areaName") or p["displayName"]
        lat, lng = float(p["lat"]), float(p["lng"])
        # 운영시간 — 동선 최적화(routing.plan_course)의 시간창 제약에 사용. 기본 상시(0~24).
        hours = p.get("operatingHours") or {}
        op_start = int(hours.get("start", 0))
        op_end = int(hours.get("end", 24))
        # 1차: description 1줄을 summary aspect 청크로. 컨텐츠 확장 시 aspect 별로 늘린다.
        meta = {
            "doc_type": "place",
            "place_id": place_id,
            "display_name": p["displayName"],
            "area_name": p.get("areaName", ""),
            "category": p.get("category", ""),
            "region": _region(lat, lng),
            "lat": lat,
            "lng": lng,
            "op_start": op_start,
            "op_end": op_end,
            "aspect": "summary",
        }
        text = f"[{p['displayName']}] ({p.get('category','')}) {p.get('description','')}"
        docs.append(Document(page_content=text, metadata=meta, id=f"{place_id}::summary::0"))
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
        meta = {
            "doc_type": "course",
            "place_id": c["id"],
            "display_name": c["title"],
            "category": c.get("category", ""),
            "aspect": "course",
        }
        docs.append(Document(page_content=text, metadata=meta, id=f"course::{c['id']}"))
    return docs


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  ! 없음, 건너뜀: {path}")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def run() -> None:
    s = get_settings()
    places = _load(s.places_json)
    courses = _load(s.courses_json)

    docs = _place_docs(places) + _course_docs(courses)
    if not docs:
        print("인제스트할 문서가 없습니다. data/raw/*.json 을 먼저 준비하세요.")
        return

    vs = get_vectorstore()
    vs.add_documents(docs, ids=[d.id for d in docs])
    print(f"upsert 완료: 장소 {len(places)} · 코스 {len(courses)} → 청크 {len(docs)}")


if __name__ == "__main__":
    run()
