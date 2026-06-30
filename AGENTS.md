# AGENTS.md — lewisai 작업 규약

이 저장소는 서울로(strangemap)의 AI 기능을 **파이썬 LangChain RAG 에이전트**로 옮기는 프로젝트다.
원본 설계 청사진: strangemap 레포 `docs/python-rag-agent-plan.md` (vLLM/Qdrant/LangGraph 버전).
이 저장소는 그것을 **Chroma+SQLite / LangChain** 으로 구현한 1차 코드이다.

## 단계
- **1차(현재 작업 대상)**: RAG + LangChain, FastAPI REST. LangGraph 쓰지 않음.
- **2차(아직)**: 동일 기능을 LangGraph StateGraph 로. 지금은 손대지 말 것.

## 핵심 규칙
1. **기능마다 폴더 분리.** 새 AI 기능 = `app/features/<name>/{schema,prompt,chain}.py` + `app/api/routes/<name>.py` 한 쌍.
2. **실시간 데이터는 `app/tools/` 에만.** 혼잡도(citydata_ppltn)·행사(culturalEventInfo)는 절대 벡터DB(`app/rag/ingest.py`)에 인제스트하지 말 것. 초단위로 변한다.
3. **벡터DB에는 잘 안 변하는 컨텐츠만** (장소 서사, 코스 노하우, 큐레이션 글).
4. **응답 JSON 계약 동결.** `AIPlaceInfo`, `Suggestion[]` 등 서울로 프론트 스키마를 깨지 말 것 (UI 재작업 0이 목표).
5. **API 키는 `.env` 만.** 코드 하드코딩 금지.
6. **화이트리스트 강제.** recommend/course 는 LLM 응답 장소가 RAG 후보 목록에 있을 때만 채택(환각 차단). 실패 시 mock 폴백.

## 메타데이터 규약 (Chroma payload)
`place_id`, `display_name`, `area_name`(실시간 API 매칭 키), `region`(강북/강남/강서/강동), `lat`, `lng`, `category`, `aspect`(summary|history|photo|tip|access|course), `doc_type`(place|course).
`region` 은 strangemap `getRegion()` 좌표 로직과 일치시킨다 (`app/rag/ingest.py:_region`).

## 데이터 출처
strangemap `src/lib/seoulPlaces.ts`(71개), `src/data/themeCourses.ts`. `node scripts/export_data.mjs` 로 `data/raw/*.json` 생성 후 `python -m scripts.run_ingest`.
