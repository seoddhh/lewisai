"""임베딩 팩토리 — 제미나이 네이티브 임베딩. 인제스트·쿼리에서 동일 인스턴스 사용."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    s = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=s.embedding_model,
        google_api_key=s.google_api_key,
    )
