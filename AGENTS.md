# AGENTS.md — lewisai 작업 규약

이 저장소는 서울로(strangemap)의 AI 기능을 **파이썬 LangChain RAG 에이전트**로 옮기는 프로젝트다.
원본 설계 청사진: strangemap 레포 `docs/python-rag-agent-plan.md` (vLLM/Qdrant/LangGraph 버전).
이 저장소는 그것을 **Chroma+SQLite / LangChain** 으로 구현한 1차 코드이다.

## 단계
- **1차(완료)**: RAG + LangChain, FastAPI REST. 기능별 결정론적 LCEL 체인.
- **2차(진행 중)**: LangGraph StateGraph 통합 에이전트(`app/graph/`). 자연어 진입 `POST /agent/chat`
  (router → parse_intent → 의도별 파이프라인). **동선 순서는 LLM 이 아니라 결정론적 오픈-패스
  TSP(`app/core/routing.py`, strangemap `courseRouting.ts` 이식)가 정한다** — "어떤 장소"는 AI,
  "어떤 순서"는 routing 모듈. 실시간 혼잡도를 동선에 반영(붐빔 시 대체·경고).
  기존 타입 엔드포인트(place_intro/recommend/course/chitchat)는 계약 유지(course 는 그래프에 위임).

## 핵심 규칙
1. **기능마다 폴더 분리.** 새 AI 기능 = `app/features/<name>/{schema,prompt,chain}.py` + `app/api/routes/<name>.py` 한 쌍.
2. **실시간 데이터는 `app/tools/` 에만.** 혼잡도(citydata_ppltn)·행사(culturalEventInfo)는 절대 벡터DB(`app/rag/ingest.py`)에 인제스트하지 말 것. 초단위로 변한다.
3. **벡터DB에는 잘 안 변하는 컨텐츠만** (장소 서사, 코스 노하우, 큐레이션 글).
4. **응답 JSON 계약 동결.** `AIPlaceInfo`, `Suggestion[]` 등 서울로 프론트 스키마를 깨지 말 것 (UI 재작업 0이 목표).
5. **API 키는 `.env` 만.** 코드 하드코딩 금지.
6. **화이트리스트 강제.** recommend/course 는 LLM 응답 장소가 RAG 후보 목록에 있을 때만 채택(환각 차단). 실패 시 mock 폴백.

## 메타데이터 규약 (Chroma payload)
`place_id`, `display_name`, `area_name`(실시간 API 매칭 키), `region`(강북/강남/강서/강동), `lat`, `lng`, `op_start`/`op_end`(운영시간 — routing 시간창 제약용, 기본 0/24), `category`, `aspect`(summary|history|photo|tip|access|course), `doc_type`(place|course).
`region` 은 strangemap `getRegion()` 좌표 로직과 일치시킨다 (`app/rag/ingest.py:_region`).

## 데이터 출처
strangemap `src/lib/seoulPlaces.ts`(71개), `src/data/themeCourses.ts`. `node scripts/export_data.mjs` 로 `data/raw/*.json` 생성 후 `uv run python -m scripts.run_ingest`.

## 의존성 관리
uv 로 관리한다. 설치·동기화는 `uv sync`, 추가는 `uv add <pkg>`, 삭제는 `uv remove <pkg>`.
의존성은 `pyproject.toml` 에 선언하고 `uv.lock` 이 버전을 고정한다(둘 다 커밋). pip/requirements.txt 는 쓰지 않는다.
