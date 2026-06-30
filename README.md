# lewisai — 서울로 AI (Python · LangChain · RAG)

서울로(strangemap)의 AI 기능을 **파이썬 LangChain RAG 에이전트**로 재구현하는 프로젝트.
추후 서울로를 3-tier(프론트 Next.js / **AI 티어 = 이 프로젝트** / 데이터)로 분리하기 위한 AI 백엔드이자 LangChain·LangGraph·RAG 학습용.

- **1차(현재)**: RAG + LangChain → FastAPI REST. *(이 단계 구현됨)*
- **2차(예정)**: 동일 기능을 LangGraph StateGraph 로. *(아직 안 함)*

## 기능 (각 기능 = `app/features/<name>/`)

| 엔드포인트 | 기능 | 서울로 TS 대응 | 출력 계약 |
|---|---|---|---|
| `POST /agent/place_intro` | 장소소개 | `api/ai-info` | `AIPlaceInfo` |
| `POST /agent/recommend` | 상황추천 | `api/ai-recommend` | `Suggestion[]` |
| `POST /agent/course` | 테마코스(기본) | `data/themeCourses` | `Course` |
| `POST /agent/chitchat` | 잡담 폴백 | — | `{reply}` |

- **RAG 대상(임베딩 O)**: 장소 서사·코스 노하우 → Chroma.
- **실시간(임베딩 X)**: 혼잡도/행사는 `app/tools/` 에서 매 호출 fetch 후 프롬프트 주입.

## 폴더 구조

```
app/
  main.py            FastAPI 엔트리
  config.py          .env 설정(LLM provider/키/경로)
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

cp .env.example .env          # ← LLM_API_KEY / SEOUL_API_KEY 입력

# 1) 데이터 export (strangemap TS → data/raw/*.json). 샘플 4개가 이미 있어 생략 가능.
node scripts/export_data.mjs

# 2) 인제스트 (청킹 → 임베딩 → Chroma)
python -m scripts.run_ingest

# 3) 서버 실행
uvicorn app.main:app --reload --port 8800
```

테스트:
```bash
curl -s localhost:8800/health | jq
curl -s localhost:8800/agent/place_intro -H 'content-type: application/json' \
  -d '{"place":"남산공원","lat":37.5512,"lng":126.9882}' | jq
```

## LLM 공급자 전환 (`.env` 의 `LLM_PROVIDER`)
- `nim` / `local` / `openai`: OpenAI 호환 → `LLM_BASE_URL`·`LLM_API_KEY`·`LLM_MODEL`.
- `gemini`: 네이티브 → `GOOGLE_API_KEY`·`GEMINI_MODEL`.

> 임베딩 모델을 바꾸면(`EMBEDDING_MODEL`) 전체 재인제스트 필요.
