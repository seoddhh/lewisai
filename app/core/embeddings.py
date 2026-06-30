"""임베딩 팩토리 — 로컬 HuggingFace 모델. 인제스트·쿼리에서 동일 인스턴스 사용."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    s = get_settings()
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
