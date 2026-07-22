# AGENTS.md — lewisai 작업 규약

이 저장소는 서울로(strangemap)의 AI 기능을 **파이썬 LangChain RAG 에이전트**로 옮기는 프로젝트다.
원본 설계 청사진: strangemap 레포 `docs/python-rag-agent-plan.md` (vLLM/Qdrant/LangGraph 버전).
이 저장소는 그것을 **Chroma+SQLite / LangChain** 으로 구현한 1차 코드이다.

## 단계
- **1차(완료)**: RAG + LangChain, FastAPI REST. 기능별 결정론적 LCEL 체인.
- **2차(완료)**: LangGraph StateGraph 통합 에이전트(`app/graph/`). 자연어 진입 `POST /agent/chat`,
  복합 요청은 Plan-and-Execute(`app/graph/plan_execute/`) + `POST /agent/chat/v2`.
- **3차(진행 중)**: 칩 진입 + 역할 재분리. `POST /agent/chat/v2` 가 `chips`(경로 생성 칩)를 받으면
  코스 그래프로 직행한다. 기존 타입 엔드포인트는 계약 유지(필드는 추가만).

## 역할 분리 (가장 중요)
- **AI 서버(이 레포)**: "어떤 장소를, 왜" — 장소 선정 + `reason`(선정 이유) + `activities`
  (그곳에서 할 수 있는 일). 지금의 서울(오늘 날짜·시간대·실시간 혼잡도)에 맞춰 쓴다.
- **개인화 축 분리**: 목적(`PURPOSE_RULES`)은 **검색(장소 선정)**을, 동반자(`COMPANION_RULES`)는
  검색을 건드리지 않고 **행동(`activities`)·서사만** 움직인다 — 임베딩 장소가 지역구 단위라
  같은 장소라도 누구와 왔는지로 할 일이 갈리게 한다 (친구와 잠실→야구, 연인과 잠실→호수 산책).
- **strangemap 프론트(`src/lib/courseRouting.ts`)**: "어떤 순서로, 어떤 선으로" — 방문 순서,
  지도 오버레이 폴리라인, 경로 거리. **라우팅 로직을 서버에 다시 만들지 말 것.**
  서버는 stop 마다 `lat`/`lng` 만 실어 보낸다 (`distance_km` 은 항상 null — deprecated).
  `app/core/geo.py` 의 haversine 은 라우팅용이 아니라 반경 필터·핫스팟 매칭 전용이다.
- **주변 정보는 3분할** (데이터 시간성에 따라 소스를 나눈다, 한 소스 = 한 역할):
  - **Visit Seoul(`app/tools/visitseoul.py`)** = **식당 전용.** 느리게 변하므로 권역(9개)별로
    미리 구워 캐시한다 (`scripts/build_meal_cache.py` → `data/meal_cache/`, 런타임은 `app/tools/meal_cache.py`).
    캐시가 없거나 얇으면 라이브 조회로 폴백. 코스에 넣을 장소를 Visit Seoul 로 고르지 않는다(그건 RAG 담당).
  - **서울시 citydata** = 혼잡도(`congestion.py`) + **실시간 문화행사**(`events.py` EVENT_STTS).
    행사는 실시간이라 캐시하지 않는다 (Visit Seoul 축제 목록은 지난 행사가 섞여 안 쓴다).
  - **문화·역사·자연 관광지** = 정적이라 임베딩 세트(seoul_places)에 큐레이션. 런타임 fetch 하지 않는다.

## 경로 생성 칩 (프론트와 어휘 1:1 — 트리플식 단계별 위저드)
`app/features/course/schema.py:CourseChips` — audience(local 서울 시민 / tourist 서울 여행자),
companion(혼자/친구와/연인과/배우자와/아이와/부모님과),
time_window(로컬: 시작~종료 시각) 또는 time(오전/오후/밤), days(여행자: 1~6 — 당일치기~5박6일, 일자별 코스 생성),
purpose(로컬: 힐링/놀거리/데이트/관광/문화생활 · 여행자: 체험·액티비티/핫플레이스/자연 힐링/유명 관광지/
문화·예술·역사/쇼핑/맛집 탐방 — 목적별 검색어·분위기 태그·선정 조건은 `PURPOSE_RULES`),
location(종로·중구/강북·성북/홍대·마포/용산·이태원/여의도·영등포/강남·서초/성수·건대/잠실·송파/관악·사당/상관없음),
congestion(여유/보통/상관없음 — 프론트에서 필수 선택, 상관없음일 때만 혼잡도 미반영·API 미호출),
pace(packed 하루 5곳 / relaxed 하루 3곳 — 미선택 시 place_count 3~5 사용).
위치 칩 → 좌표·자치구 매핑은 `app/core/geo.py:LOCATION_CHIPS`.
시간 범위에 09/13/19시가 포함되면 아침/점심/저녁 식사 슬롯이 생기고, 앵커 장소 3km 이내
Visit Seoul 실데이터 식당을 코스 전체 중복 없이 `meal_options`(최대 4곳)로 싣는다 — 없으면 지어내지 않는다.

## 핵심 규칙
1. **기능마다 폴더 분리.** 새 AI 기능 = `app/features/<name>/{schema,prompt,chain}.py` + `app/api/routes/<name>.py` 한 쌍.
2. **실시간 데이터는 `app/tools/` 에만.** 혼잡도(citydata_ppltn)·행사(culturalEventInfo)는 절대 벡터DB(`app/rag/ingest.py`)에 인제스트하지 말 것. 초단위로 변한다.
3. **벡터DB에는 잘 안 변하는 컨텐츠만** (장소 서사, 코스 노하우, 큐레이션 글).
4. **응답 JSON 계약 동결.** `AIPlaceInfo`, `Suggestion[]`, `CourseStop` 등 서울로 프론트 스키마를
   깨지 말 것 — **필드 추가는 되고, 기존 필드 제거·의미 변경은 안 된다** (UI 재작업 0이 목표).
5. **API 키는 `.env` 만.** 코드 하드코딩 금지.
6. **화이트리스트 강제.** recommend/course 는 LLM 응답 장소가 RAG 후보 목록에 있을 때만 채택(환각 차단). 실패 시 mock 폴백.
7. **라우팅 금지.** 방문 순서·폴리라인·경로 거리는 프론트 몫이다 (위 "역할 분리" 참고).

## 메타데이터 규약 (Chroma payload)
`place_id`, `display_name`, `area_name`(실시간 API 매칭 키), `region`(강북/강남/강서/강동), `lat`, `lng`, `op_start`/`op_end`(운영시간 — 프론트 라우팅 시간창·표시용, 기본 0/24), `category`, `aspect`(summary|history|photo|tip|access|course), `doc_type`(place|course). 코스 문서 한정: `tags`(콤마 문자열), `is_filming`(bool — K-컨텐츠 촬영지, `search_kcontent_filming_spots` 필터 키).
`region` 은 strangemap `getRegion()` 좌표 로직과 일치시킨다 (`app/core/geo.py:region_of`).

## 데이터 출처
strangemap `src/lib/seoulPlaces.ts`(71개), `src/data/themeCourses.ts`. `node scripts/export_data.mjs` 로 `data/raw/*.json` 생성 후 `uv run python -m scripts.run_ingest`.

## 의존성 관리
uv 로 관리한다. 설치·동기화는 `uv sync`, 추가는 `uv add <pkg>`, 삭제는 `uv remove <pkg>`.
의존성은 `pyproject.toml` 에 선언하고 `uv.lock` 이 버전을 고정한다(둘 다 커밋). pip/requirements.txt 는 쓰지 않는다.
