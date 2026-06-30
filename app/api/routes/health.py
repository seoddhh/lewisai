from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "llm_provider": "gemini",
        "llm_model": s.gemini_model,
        "embedding_model": s.embedding_model,
        "chroma_dir": s.chroma_dir,
    }
