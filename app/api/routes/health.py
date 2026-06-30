from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model if s.llm_provider != "gemini" else s.gemini_model,
        "embedding_model": s.embedding_model,
        "chroma_dir": s.chroma_dir,
    }
