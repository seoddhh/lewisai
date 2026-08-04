from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """서버가 살아 있는지와, 지금 어떤 모델·컬렉션에 붙어 있는지 알려준다.

    배포 후 설정이 의도대로 들어갔는지(특히 임베딩 모델과 컬렉션 짝이 맞는지)
    눈으로 확인하는 용도다.
    """
    s = get_settings()
    gemini = s.llm_provider.lower() == "gemini"
    return {
        "status": "ok",
        "llm_provider": "gemini" if gemini else "upstage",
        "llm_model": s.gemini_model if gemini else s.upstage_model,
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.active_embedding_model,
        "chroma_dir": s.chroma_dir,
        "chroma_collection": s.chroma_collection,
    }
