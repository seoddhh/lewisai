"""코스만 만드는 단순 API. 챗 형태가 아니라 코스 결과만 필요할 때 쓴다."""
from fastapi import APIRouter

from app.features.course.chain import run
from app.features.course.schema import CourseRequest, CourseResponse

router = APIRouter(tags=["course"])


@router.post("/agent/course", response_model=CourseResponse)
async def course(req: CourseRequest) -> CourseResponse:
    return await run(req)
