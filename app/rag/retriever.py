"""Chroma 메타데이터 필터 리트리버 — 권역·목적 필터를 벡터검색 단계에서 적용한다.

진입점은 `search_with_score` 하나다. 점수 없는 `search()`·MMR `search_diverse()` 도
있었지만 코스 파이프라인이 지리 재랭킹을 하려면 의미 거리가 필요해 쓰이지 않았다.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.core.vectorstore import get_vectorstore


def _where(filters: dict | None) -> dict | None:
    """{key: value} 또는 {key: [v1, v2]} → Chroma where 절 ($and 결합).

    "$or"/"$and" 로 시작하는 키는 이미 Chroma 문법인 절로 보고 그대로 통과시킨다
    (목적 필터처럼 "선택한 목적 중 하나라도" 조건을 걸 때 쓴다).
    """
    if not filters:
        return None
    clauses = []
    for key, val in filters.items():
        if val is None:
            continue
        if key.startswith("$"):
            clauses.append({key: val})
        elif isinstance(val, (list, tuple)):
            clauses.append({key: {"$in": list(val)}})
        else:
            clauses.append({key: {"$eq": val}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search_with_score(
    query: str, k: int = 40, filters: dict | None = None
) -> list[tuple[Document, float]]:
    """유사도 검색 + 점수. (doc, distance) 리스트, 거리가 작을수록 더 유사.

    코스 후처리 재랭킹(지리 근접 가중)에서 의미 유사도 점수가 필요할 때 쓴다.
    """
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k, filter=_where(filters))
