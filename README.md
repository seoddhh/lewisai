# lewisai — 서울로 AI (Python · LangChain · RAG)

서울로(strangemap)의 AI 기능을 **파이썬 LangGraph RAG 에이전트**로 재구현하는 프로젝트.
추후 서울로를 3-tier(프론트 Next.js / **AI 티어** / 데이터베이스)로 분리하기 위한 AI 백엔드이자 LangChain·LangGraph·RAG를 구현하기 위한 레포지토리

- **1차(완료)**: RAG + LangChain → FastAPI REST.
- **2차(진행 중)**: LangGraph StateGraph 통합 에이전트 + 자연어 진입 `POST /agent/chat`.
  동선 순서는 결정론적 오픈-패스 TSP(`app/core/routing.py`)로 지도에 최적화 동선 오버레이, 실시간 혼잡도 반영

## 기능 (각 기능 = `app/features/<name>/`)

| 엔드포인트 | 기능 | 서울로 TS 대응 | 출력 계약 |
|---|---|---|---|
| `POST /agent/place_intro` | 장소소개 | `api/ai-info` | `AIPlaceInfo` |
| `POST /agent/recommend` | 상황추천 | `api/ai-recommend` | `Suggestion[]` |
| `POST /agent/course` | 테마코스(동선 최적화) | `data/themeCourses` | `Course` |
| `POST /agent/chitchat` | 기본 응답 | — | `{reply}` |
| `POST /agent/chat` | 자연어 통합(의도 자동 분기) | — | `{intent, result, source, distance_km}` |

- **RAG 대상(임베딩 O)**: 장소 서사·코스 노하우 → Chroma.
- **실시간(임베딩 X)**: 혼잡도/행사는 `app/tools/` 에서 매 호출 fetch 후 프롬프트 주입 (실시간으로 바뀌는 정보라 RAG로 저장하지 않음)

## 폴더 구조

```
app/
  main.py            FastAPI 엔트리
  config.py          .env 설정(제미나이 키/경로)
  core/              llm·embeddings·vectorstore·json_parse·routing(동선 TSP) 팩토리
  rag/               retriever(메타필터) · ingest(배치)
  tools/             congestion · events  (실시간, RAG 아님)
  graph/             LangGraph 통합 에이전트 (state · build · nodes/)
  features/          place_intro · recommend · course · chitchat
                     └ 각 폴더: schema.py / prompt.py / chain.py
  api/routes/        기능별 FastAPI 라우터 (+ chat)
data/raw/            seoul_places.json · theme_courses.json (기존 서울로 json포맷 데이터)
data/chroma/         Chroma 영속(sqlite 포함) — gitignore
scripts/             export_data.mjs · run_ingest.py
```

## LangGraph 에이전트 & 도구

자연어(`POST /agent/chat`)가 들어오면 그래프가 의도를 분기하고, 각 노드가 아래 **도구**를 결정론적으로
호출한다. LLM function-calling이 아니라 **그래프 노드**가 도구를 호출하는 방식

역할 분리 원칙: 장소는 LLM이 반환, 그에 맞는 경로는 알고리즘 로직으로 반환하여 분리 한다.

```
START → router → parse_intent → (의도 분기)
  course → retrieve → select_places → enrich → optimize → compose → END
  recommend / place_intro / chitchat → 기존 feature 체인 재사용 → END
```

### 노드가 호출하는 도구 목록

| 도구 | 위치 | 역할 | 입력 → 출력 |
|---|---|---|---|
| `retriever.search` | `app/rag/retriever.py` | 유사도 검색 + 메타필터 | query, filters → Document[] |
| `retriever.search_diverse` | `app/rag/retriever.py` | MMR 다양성 검색(후보 풀) | query, filters → Document[] |
| `tools.get_congestion` | `app/tools/congestion.py` | 실시간 혼잡도(5분 캐시) | area_name → "여유/보통/붐빔: …" |
| `tools.get_events` | `app/tools/events.py` | 반경 3km 문화행사 | lat, lng → event[] |
| `routing.plan_course` | `app/core/routing.py` | 오픈-패스 TSP 순서 최적화 + 5곳·10km 상한 | stops → 정렬된 stops, 총거리 |
| `routing.select_cluster` | `app/core/routing.py` | 3km 군집 선택(거리 사전 반영) | cands → 한 군집 |
| `routing.haversine_km` | `app/core/routing.py` | 좌표 간 거리 | 2좌표 → km |

### 노드별로 쓰는 도구

| 노드 | 하는 일 | 호출 도구 |
|---|---|---|
| `router` | 자연어 → 의도(course/recommend/place_intro/chitchat) 분류 | LLM |
| `parse_intent` | 자연어 → 구조화 요청(note/region/time 등) 추출 | LLM |
| `retrieve` | 권역·doc_type 필터로 후보 장소 검색 | `search_diverse` |
| `select_places` | 후보 중 요청에 맞는 장소 선택(화이트리스트 강제) | LLM |
| `enrich` | 선택 장소 혼잡도 조회, **'붐빔'이면 후보 대체·경고** | `get_congestion` |
| `optimize` | **방문 순서 결정론적 확정**(LLM 아님) | `plan_course` |
| `compose` | 확정 순서에 제목·서사만 입힘(순서 불변) | LLM |

> `optimize` 는 좌표 기반 오픈-패스 TSP(`app/core/routing.py`, strangemap `courseRouting.ts` 이식)로
> 지그재그 동선을 구조적으로 제거한다. 스톱 수에 따라 전수순열(n≤8)·Held-Karp(n≤13)·2-opt(n>13)를
> 자동 선택하고, 시간대가 있으면 운영시간 시간창 위반을 거리보다 우선 최소화한다.

## 빠른 시작

```bash
cd /Users/seodonghwi/Desktop/lewisai
uv sync                            # pyproject.toml + uv.lock 기준으로 .venv 생성·동기화

cp .env.example .env          # ← GOOGLE_API_KEY / SEOUL_API_KEY 입력

# 1) 데이터 export (strangemap TS → data/raw/*.json). 샘플 4개가 이미 있어 생략 가능.
node scripts/export_data.mjs

# 2) 인제스트 (청킹 → 임베딩 → Chroma)
uv run python -m scripts.run_ingest

# 3) 서버 실행
uv run uvicorn app.main:app --reload --port 8800
```

> 의존성은 `pyproject.toml` 에 선언하고 정확한 버전은 `uv.lock` 이 고정한다(둘 다 커밋).
> 패키지 추가/삭제는 `uv add <pkg>` / `uv remove <pkg>` (lock 자동 갱신).
> `uv run` 은 활성화된 venv 없이도 `.venv` 를 자동 사용하며, `source .venv/bin/activate` 후엔
> `python -m ...` / `uvicorn ...` 를 그대로 써도 된다.
## 테스트를 위한 json 포맷 구조
테스트:
```bash
curl -s localhost:8800/health | jq
curl -s localhost:8800/agent/place_intro -H 'content-type: application/json' \
  -d '{"place":"남산공원","lat":37.5512,"lng":126.9882}' | jq
```

### 기능별 요청/응답 JSON 포맷

#### `GET /health`
응답:
```json
{
  "status": "ok",
  "llm_provider": "gemini",
  "llm_model": "gemini-3.1-flash-lite",
  "embedding_model": "models/gemini-embedding-001",
  "chroma_dir": "./data/chroma"
}
```

#### `POST /agent/place_intro`
요청:
```json
{
  "place": "남산공원",
  "lat": 37.5512,
  "lng": 126.9882,
  "type": "culture"
}
```
응답:
```json
{
  "info": {
    "placeName": "남산공원",
    "summary": "서울 도심 속 대표 자연 명소",
    "highlights": ["N서울타워 야경", "산책로", "자물쇠 전망대"],
    "tip": "케이블카보다 걷는 코스가 여유롭습니다.",
    "best_time": "일몰 직후",
    "crowd_tip": "주말 오후는 혼잡, 평일 저녁 추천",
    "right_now": "현재 혼잡도 보통",
    "viewpoint_guide": "N서울타워 전망대에서 도심 전경 감상",
    "nearby": ["명동", "회현동"],
    "vibe": ["로맨틱", "힐링"],
    "tags": ["야경", "자연", "데이트"],
    "events": [
      {
        "title": "가을 빛초롱 축제",
        "desc": "야간 조명 전시",
        "period": "2026-10-01 ~ 2026-10-31",
        "fee": "무료",
        "dist_km": 0.3
      }
    ]
  },
  "source": "ai"
}
```

#### `POST /agent/recommend`
요청:
```json
{
  "companion": "연인",
  "ageGroup": "20대",
  "time": "저녁",
  "purpose": "데이트",
  "region": "강북",
  "congestion": "여유"
}
```
응답:
```json
{
  "suggestions": [
    {
      "title": "남산 야경 데이트",
      "place": "남산공원",
      "duration": "2시간",
      "description": "케이블카 없이 걸어서 즐기는 야경 코스",
      "reason": "연인과 함께 걷기 좋은 저녁 코스",
      "tags": ["야경", "산책", "데이트"]
    }
  ],
  "source": "ai"
}
```

#### `POST /agent/course`
요청:
```json
{
  "note": "야경 보면서 데이트",
  "region": "상관없음",
  "time": "저녁"
}
```
응답:
```json
{
  "course": {
    "title": "서울 야경 데이트 코스",
    "subtitle": "도심 속 로맨틱 산책",
    "description": "해질 무렵부터 야경까지 즐기는 코스",
    "stops": [
      {
        "name": "남산공원",
        "preview": "도심 속 자연",
        "description": "산책로를 따라 N서울타워까지",
        "duration": "1시간 30분",
        "tip": "편한 신발 추천"
      }
    ],
    "tags": ["야경", "데이트", "산책"]
  },
  "source": "ai"
}
```

#### `POST /agent/chitchat`
요청:
```json
{
  "message": "서울에서 데이트하기 좋은 곳 있어?"
}
```
응답:
```json
{
  "reply": "남산공원 어때요? 야경 보며 산책하기 좋아요!"
}
```

#### `POST /agent/chat` (자연어 통합 · LangGraph)
자연어 한 문장으로 의도를 자동 분기한다. `course` 의도면 좌표 기반 동선 최적화 + 실시간 혼잡도가 반영된다.
요청:
```json
{
  "message": "강북에서 저녁에 야경 보면서 데이트하기 좋은 코스 짜줘"
}
```
응답 (course 의도):
```json
{
  "intent": "course",
  "result": {
    "course": {
      "title": "서울의 밤을 걷는 시간, 강북 야경 데이트",
      "subtitle": "...",
      "description": "...",
      "stops": [
        { "name": "북촌한옥마을", "preview": "...", "description": "...", "duration": "1시간", "tip": null }
      ],
      "tags": ["야경", "데이트"]
    },
    "source": "ai"
  },
  "source": "ai",
  "distance_km": 5.3
}
```
> `intent` 는 `course|recommend|place_intro|chitchat` 중 하나이며, `result` 는 해당 기능의 응답 스키마와 동일하다.
> 방문 순서는 LLM 이 아니라 `app/core/routing.py` 의 오픈-패스 TSP로직으로 결정.

## LLM / 임베딩
- 1차 구현은 챗 모델·임베딩 모두 제미나이(`langchain-google-genai`) 고정. `.env`의 `GOOGLE_API_KEY`·`GEMINI_MODEL` 설정.

