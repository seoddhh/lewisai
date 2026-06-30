from fastapi import APIRouter

from app.features.chitchat.chain import ChitchatRequest, ChitchatResponse, run

router = APIRouter(tags=["chitchat"])


@router.post("/agent/chitchat", response_model=ChitchatResponse)
async def chitchat(req: ChitchatRequest) -> ChitchatResponse:
    return await run(req)
