"""벡터 DB(Chroma) 연결. 데이터는 설정된 디렉터리에 sqlite 파일로 저장된다."""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

from app.config import get_settings
from app.core.embeddings import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    """벡터 DB 를 열어 준다. 한 번 열면 계속 재사용한다."""
    s = get_settings()
    return Chroma(
        collection_name=s.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=s.chroma_dir,
    )
