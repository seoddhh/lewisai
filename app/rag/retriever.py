"""벡터 DB 에서 장소를 검색한다. 동네·목적 같은 조건도 검색 단계에서 같이 건다.

쓰는 함수는 `search_with_score` 하나뿐이다. 코스는 검색 결과를 거리까지 따져
다시 줄 세우기 때문에, 점수 없이 목록만 받는 함수로는 부족하다.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.core.vectorstore import get_vectorstore


def _where(filters: dict | None) -> dict | None:
    """조건 딕셔너리를 Chroma 가 알아듣는 형태로 바꾼다. 조건이 여럿이면 전부 만족해야 한다.

    값이 목록이면 "그중 하나면 됨"이 되고, `$or` 같은 키는 이미 Chroma 문법이라 그대로 둔다
    ("고른 목적 중 하나라도 해당" 같은 조건을 걸 때 쓴다).
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
    """검색어와 비슷한 장소를 점수와 함께 찾는다. 점수는 작을수록 비슷하다는 뜻이다.

    이 점수는 나중에 거리와 섞어 후보 순서를 다시 정하는 데 쓴다.
    """
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k, filter=_where(filters))
