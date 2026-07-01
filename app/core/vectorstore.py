"""Chroma persistent 벡터스토어 persist 디렉터리 안에 sqlite 저장"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

from app.config import get_settings
from app.core.embeddings import get_embeddings


@lru_cache
def get_vectorstore() -> Chroma:
    s = get_settings()
    return Chroma(
        collection_name=s.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=s.chroma_dir,
    )
