from fastapi import APIRouter

from app.features.recommend.chain import run
from app.features.recommend.schema import RecommendRequest, RecommendResponse

router = APIRouter(tags=["recommend"])


@router.post("/agent/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    return await run(req)
