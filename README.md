# lewisai — 서울로 AI (Python · LangGraph · RAG)

서울로(strangemap)의 AI 기능을 **파이썬 LangGraph RAG 에이전트**로 재구현한 프로젝트.
서울로를 3-tier(프론트 Next.js / **AI 서버** / DB)로 분리하기 위한 AI 백엔드다.

**역할 분리 원칙**
- **AI 서버**: 어떤 장소를, **왜** 골랐는지(`reason`/`activities`)와 실시간 정보(혼잡도·주변 식당·문화행사).
- **strangemap 프론트(`courseRouting.ts`)**: 방문 순서·지도 폴리라인·경로 거리. 서버는 stop 좌표(`lat`/`lng`)만 실어 보낸다.

## 엔드포인트

| 메서드 · 경로 | 기능 | 요청 | 응답 |
|---|---|---|---|
| `GET /` | 검증용 챗봇 UI(정적 페이지) | — | HTML |
| `GET /health` | 헬스체크 · 모델 정보 | — | `{status, llm_model, embedding_model, …}` |
| `POST /agent/course` | 테마 코스 생성(칩 기반) | `CourseRequest` | `CourseResponse` |
| `POST /agent/chitchat` | 일반 대화 | `{message}` | `{reply}` |
| `POST /agent/chat` | **통합 진입**(칩 또는 자연어 → 코스/잡담) | `{message, chips?}` | `{kind, course/text, steps, source}` |
| `POST /agent/chat/stream` | `/agent/chat` 의 SSE 스트리밍 | `{message, chips?}` | `text/event-stream` |

`/agent/chat` 이 프론트가 실제로 붙는 통합 진입점이다:
- **칩 진입**(`chips` 있음) → 코스 파이프라인으로 직행(결정적).
- **자연어 진입**(`chips` 없음) → 라우터가 코스/잡담을 판별.

## 데이터: RAG vs 실시간

- **RAG 대상(임베딩 O)**: 장소 서사·코스 노하우 → ChromaDB에 저장.
- **실시간(임베딩 X)**: 혼잡도·문화행사는 매 호출 후 프롬프트에 주입(실시간 api라 저장 불가).

## 도구(tools)

그래프 노드가 아래 도구를 **결정론적으로** 호출한다(LLM function-calling 아님).

### RAG 검색 — `app/rag/retriever.py`

| 함수 | 용도 | 시그니처 |
|---|---|---|
| `search` | 유사도 검색 + 메타필터 | `search(query, k=6, filters) → Document[]` |
| `search_diverse` | MMR 다양성 검색(후보 풀 확보) | `search_diverse(query, k=6, filters) → Document[]` |
| `search_with_score` | 점수 포함 검색(지리 재랭킹용) | `search_with_score(query, k=40, filters) → (Document, distance)[]` |

```python
from app.rag import retriever
scored = retriever.search_with_score("강북 야경 데이트", k=40, filters={"doc_type": "place"})
# → [(Document(page_content="북촌한옥마을 …", metadata={"lat":…, "area_name":…}), 0.31), …]
```

### 실시간 혼잡도 — `app/tools/congestion.py`

| 함수 | 용도 | 시그니처 |
|---|---|---|
| `get_congestion` | 서울시 citydata_ppltn 실시간 혼잡도(5분 캐시) | `get_congestion(area_name) → "여유: …" \| None` |

```python
from app.tools.congestion import get_congestion
await get_congestion("남산공원")   # → "보통: 사람이 몰려있을 가능성이 있어요."
```

### 주변 정보(Visit Seoul) — `app/tools/visitseoul.py`

| 함수 · 상수 | 용도 |
|---|---|
| `search_nearby` | 좌표 주변(반경 내) 식당/행사/관광을 가까운 순으로 |
| `place_keyword` | 장소명 → Visit Seoul 검색 키워드 정규화(`"광화문·덕수궁"→"광화문"`) |
| `classify` | `cate_depth` → `restaurant`/`event`/`attraction` 분류 |
| `KIND_RESTAURANT` · `KIND_EVENT` · `KIND_ATTRACTION` | 종류 상수 |

```python
from app.tools.visitseoul import search_nearby, KIND_RESTAURANT, KIND_EVENT
items = await search_nearby(
    lat=37.5826, lng=126.9838,
    keywords=("북촌한옥마을", "삼청동"),
    kinds=(KIND_RESTAURANT, KIND_EVENT),
    radius_km=1.5,
)   # → NearbyItem[] (title/dist_km/kind/period …)
```

### 좌표 · 위치 칩 — `app/core/geo.py`

라우팅 로직은 **없다**. 거리 계산은 Visit Seoul 반경 필터·핫스팟 매칭 전용.

| 함수 | 용도 | 시그니처 |
|---|---|---|
| `haversine_km` | 두 좌표 직선거리(반경 필터용) | `haversine_km(a_lat, a_lng, b_lat, b_lng) → km` |
| `region_of` | 좌표 → 권역(강북/강남/강서/강동) | `region_of(lat, lng) → str` |
| `chip_of` | 위치 칩 → 중심좌표·주소어(검색 앵커) | `chip_of("홍대·마포") → LocationChip \| None` |
| `chip_region` | 위치 칩 → RAG 필터용 권역 | `chip_region("강남·서초") → "강남"` |
| `address_terms` | 위치 칩 → 주소 매칭 자치구·법정동 이름 | `address_terms("홍대·마포") → ("마포","서교",…)` |

## LangGraph 파이프라인

```
START → router → parse_intent → (의도 분기)
  course   → retrieve → select_places → enrich → nearby → compose → END
  chitchat → chitchat → END
```

에이전트의 임무는 **코스 생성**이다. 코스와 무관한 입력은 `chitchat` 로 폴백한다.

| 노드 | 하는 일 | 호출 도구 |
|---|---|---|
| `router` | 자연어 → 의도(`course`/`chitchat`) 분류 | LLM |
| `parse_intent` | 자연어 → 구조화 요청(note/region/time) | LLM |
| `retrieve` | 후보 장소 검색 후 앵커 반경 클러스터링·재랭킹 | `search_with_score` · `chip_of` · `haversine_km` |
| `select_places` | 장소 선정 + **선정 이유·활동 생성**(화이트리스트 강제) | LLM |
| `enrich` | 혼잡도 조회, 선호 칩이 있으면 '붐빔' 장소 대체 | `get_congestion` |
| `nearby` | 확정 장소의 주변 식당 / 문화행사·관광 | `search_nearby` · `place_keyword` · `address_terms` |
| `compose` | 확정 장소에 제목·서사 입힘(장소·개수 불변) | LLM |

## 폴더 구조

```
app/
  main.py            FastAPI 엔트리 (+ GET / 정적 챗봇 UI)
  config.py          .env 설정(제미나이 키 / 서울시 API 키 / 경로)
  core/              llm · embeddings · vectorstore · json_parse · geo(좌표·위치 칩)
  rag/               retriever(메타필터·재랭킹) · ingest(배치)
  tools/             congestion · visitseoul  (실시간, RAG 아님)
  graph/             LangGraph 에이전트 (state · build · nodes/)
  features/          course · chitchat  (└ schema.py / prompt.py / chain.py)
  api/routes/        health · course · chitchat · chat
  static/            index.html (검증용 챗봇 UI)
data/raw/            seoul_places.json · theme_courses.json
data/chroma/         Chroma 영속(sqlite) — gitignore
scripts/             build_dataset.mjs · run_ingest.py
```

## 빠른 시작

```bash
cd /Users/seodonghwi/Desktop/lewisai
uv sync                             # pyproject.toml + uv.lock 기준 .venv 동기화
cp .env.example .env                # ← GOOGLE_API_KEY / SEOUL_API_KEY 입력

node scripts/build_dataset.mjs      # (선택) strangemap TS → data/raw/*.json
uv run python -m scripts.run_ingest # 청킹 → 임베딩 → Chroma
uv run uvicorn app.main:app --reload --port 8800
```

> 의존성은 `pyproject.toml` 에 선언, 정확한 버전은 `uv.lock` 고정(둘 다 커밋).
> 추가/삭제는 `uv add <pkg>` / `uv remove <pkg>` (lock 자동 갱신).

## API 예시

```bash
curl -s localhost:8800/health | jq
```

### `POST /agent/chat` — 자연어 진입

```bash
curl -s localhost:8800/agent/chat -H 'content-type: application/json' \
  -d '{"message":"강북에서 저녁에 야경 보면서 데이트하기 좋은 코스 짜줘"}' | jq
```

### `POST /agent/chat` — 칩 진입

```json
{
  "message": "",
  "chips": {
    "companion": "커플", "time": "밤", "purpose": "데이트",
    "location": "종로·중구", "congestion": "여유", "place_count": 4
  }
}
```

응답(`kind: "course"`):

```json
{
  "kind": "course",
  "course": {
    "title": "서울의 밤을 걷는 시간, 강북 야경 데이트",
    "subtitle": "…", "description": "…",
    "stops": [
      {
        "name": "북촌한옥마을",
        "preview": "…", "description": "…", "duration": "1시간", "tip": null,
        "lat": 37.5826, "lng": 126.9838,
        "reason": "커플이 걷기 좋은 한옥 골목이라 골랐어요.",
        "activities": ["골목 사진", "전통 공방 체험"],
        "congestion": "보통",
        "nearby": {
          "restaurants": [{ "title": "삼청동 수제비", "dist_km": 0.48, "kind": "restaurant" }],
          "attractions": [{ "title": "북촌 공예주간", "kind": "event", "period": "2026-06-20~2026-07-20" }]
        }
      }
    ],
    "tags": ["야경", "데이트"]
  },
  "steps": [
    { "id": "s1", "tool": "retrieve",       "label": "후보 장소 검색 (RAG)",           "ok": true, "detail": "후보 10곳" },
    { "id": "s2", "tool": "select_places",  "label": "AI 장소 선정",                   "ok": true, "picks": [{ "name": "북촌한옥마을", "reason": "…", "activities": ["…"] }] },
    { "id": "s3", "tool": "enrich",         "label": "실시간 혼잡도 확인",              "ok": true, "detail": "북촌한옥마을 보통" },
    { "id": "s4", "tool": "nearby",         "label": "주변 식당·문화행사 조회 (Visit Seoul)", "ok": true, "detail": "주변 정보 6건" },
    { "id": "s5", "tool": "compose",        "label": "코스 서사 작성",                 "ok": true, "detail": "서울의 밤을 걷는 시간…" }
  ],
  "source": "ai"
}
```

- `kind` 는 `course` 또는 `text`(잡담 폴백).
- `steps` 는 "이 장소들이 어떻게 나왔는지" 생성 과정 트레이스 — 챗봇이 그대로 노출한다.
- **방문 순서·폴리라인·거리는 서버가 만들지 않는다** — 프론트가 `lat`/`lng` 로 계산한다.

### `POST /agent/course` — 칩 기반 코스(단일 기능)

요청 `CourseRequest`:

```json
{
  "note": "야경 보면서 데이트",
  "chips": { "companion": "커플", "time": "밤", "purpose": "데이트", "place_count": 4 }
}
```

응답은 `CourseResponse`(`{course, source}`) — `stops[]` 구조는 위 `chat` 응답의 `course.stops` 와 동일.

### `POST /agent/chitchat`

```json
// 요청
{ "message": "서울에서 데이트하기 좋은 곳 있어?" }
// 응답
{ "reply": "남산공원 어때요? 야경 보며 산책하기 좋아요!" }
```

### `POST /agent/chat/stream` — SSE

`/agent/chat` 과 입력이 같고, 진행 상황을 `text/event-stream` 으로 흘린다.

| `event` | 페이로드 | 설명 |
|---|---|---|
| `progress` | `{stage, message}` | 코스 생성 단계 안내(retrieve→…→compose) |
| `token` | `{text}` | 잡담 답변 조각(실시간 타이핑). 코스는 JSON 이라 미노출 |
| `final` | `{payload}` | 최종 구조화 응답(코스/steps) |

## LLM / 임베딩

챗 모델·임베딩 모두 제미나이(`langchain-google-genai`) 고정.
`.env` 의 `GOOGLE_API_KEY`·`GEMINI_MODEL`, 혼잡도·주변 정보용 `SEOUL_API_KEY` 설정.
