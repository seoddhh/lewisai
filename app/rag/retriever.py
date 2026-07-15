"""Chroma 메타데이터 필터 리트리버. 권역/카테고리/place_id 필터를 벡터검색 단계에서 적용."""
from __future__ import annotations

from langchain_core.documents import Document

from app.core.vectorstore import get_vectorstore


def _where(filters: dict | None) -> dict | None:
    """{key: value} 또는 {key: [v1, v2]} → Chroma where 절."""
    if not filters:
        return None
    clauses = []
    for key, val in filters.items():
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            clauses.append({key: {"$in": list(val)}})
        else:
            clauses.append({key: {"$eq": val}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(query: str, k: int = 6, filters: dict | None = None) -> list[Document]:
    """유사도 검색 + 메타필터."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k, filter=_where(filters))


def search_diverse(query: str, k: int = 6, filters: dict | None = None) -> list[Document]:
    """MMR 검색 — 추천처럼 다양성이 필요한 의도용."""
    vs = get_vectorstore()
    return vs.max_marginal_relevance_search(
        query, k=k, fetch_k=max(k * 4, 20), filter=_where(filters)
    )


def search_with_score(
    query: str, k: int = 40, filters: dict | None = None
) -> list[tuple[Document, float]]:
    """유사도 검색 + 점수. (doc, distance) 리스트, 거리가 작을수록 더 유사.

    코스 후처리 재랭킹(지리 근접 가중)에서 의미 유사도 점수가 필요할 때 쓴다.
    """
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k, filter=_where(filters))
