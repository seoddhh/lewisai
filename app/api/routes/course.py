from fastapi import APIRouter

from app.features.course.chain import run
from app.features.course.schema import CourseRequest, CourseResponse

router = APIRouter(tags=["course"])


@router.post("/agent/course", response_model=CourseResponse)
async def course(req: CourseRequest) -> CourseResponse:
    return await run(req)
