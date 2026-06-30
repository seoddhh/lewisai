from fastapi import APIRouter

from app.features.place_intro.chain import run
from app.features.place_intro.schema import PlaceIntroRequest, PlaceIntroResponse

router = APIRouter(tags=["place_intro"])


@router.post("/agent/place_intro", response_model=PlaceIntroResponse)
async def place_intro(req: PlaceIntroRequest) -> PlaceIntroResponse:
    return await run(req)
