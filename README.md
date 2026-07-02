# lewisai — 서울로 AI (Python · LangChain · RAG)

서울로(strangemap)의 AI 기능을 **파이썬 LangGraph RAG 에이전트**로 재구현하는 프로젝트.
추후 서울로를 3-tier(프론트 Next.js / **AI 티어** / 데이터베이스)로 분리하기 위한 AI 백엔드이자 LangChain·LangGraph·RAG를 구현하기 위한 레포지토리

- **1차(현재)**: RAG + LangChain → FastAPI REST. 구현 완료
- **2차(예정)**: 동일 기능을 LangGraph StateGraph + mcp 연결까지 구상중

## 기능 (각 기능 = `app/features/<name>/`)

| 엔드포인트 | 기능 | 서울로 TS 대응 | 출력 계약 |
|---|---|---|---|
| `POST /agent/place_intro` | 장소소개 | `api/ai-info` | `AIPlaceInfo` |
| `POST /agent/recommend` | 상황추천 | `api/ai-recommend` | `Suggestion[]` |
| `POST /agent/course` | 테마코스(기본) | `data/themeCourses` | `Course` |
| `POST /agent/chitchat` | 기본 응답 | — | `{reply}` |

- **RAG 대상(임베딩 O)**: 장소 서사·코스 노하우 → Chroma.
- **실시간(임베딩 X)**: 혼잡도/행사는 `app/tools/` 에서 매 호출 fetch 후 프롬프트 주입.

## 폴더 구조

```
app/
  main.py            FastAPI 엔트리
  config.py          .env 설정(제미나이 키/경로)
  core/              llm·embeddings·vectorstore·json_parse 팩토리
  rag/               retriever(메타필터) · ingest(배치)
  tools/             congestion · events  (실시간, RAG 아님)
  features/          place_intro · recommend · course · chitchat
                     └ 각 폴더: schema.py / prompt.py / chain.py
  api/routes/        기능별 FastAPI 라우터
data/raw/            seoul_places.json · theme_courses.json (strangemap export)
data/chroma/         Chroma 영속(sqlite 포함) — gitignore
scripts/             export_data.mjs · run_ingest.py
```

## 빠른 시작

```bash
cd /Users/seodonghwi/Desktop/lewisai
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # ← GOOGLE_API_KEY / SEOUL_API_KEY 입력

# 1) 데이터 export (strangemap TS → data/raw/*.json). 샘플 4개가 이미 있어 생략 가능.
node scripts/export_data.mjs

# 2) 인제스트 (청킹 → 임베딩 → Chroma)
python -m scripts.run_ingest

# 3) 서버 실행
uvicorn app.main:app --reload --port 8800
```
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
  "llm_model": "gemini-2.0-flash",
  "embedding_model": "models/text-embedding-004",
  "chroma_dir": "data/chroma"
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

## LLM / 임베딩
- 챗 모델·임베딩 모두 제미나이(`langchain-google-genai`) 고정. `.env`의 `GOOGLE_API_KEY`·`GEMINI_MODEL` 설정.

> 임베딩 모델을 바꾸면(`EMBEDDING_MODEL`) 전체 재인제스트 필요.
