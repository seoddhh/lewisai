from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    gemini = s.llm_provider.lower() == "gemini"
    return {
        "status": "ok",
        # 실제로 어느 프로바이더가 붙어 있는지 그대로 보고한다 (LLM_PROVIDER 기준)
        "llm_provider": "gemini" if gemini else "upstage",
        "llm_model": s.gemini_model if gemini else s.upstage_model,
        "embedding_model": s.embedding_model,
        "chroma_dir": s.chroma_dir,
    }
