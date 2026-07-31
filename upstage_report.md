# Solar Agent Partner Stage 1 — 결과물 제출

> **프로젝트** 서울로(Seoulro) AI 코스 생성 에이전트 — `lewisai`

> **참가자** 서동휘

> **모델** `solar-open2` (Solar Open 2, Private Beta)

> **기간** 2026-07-17 ~ 2026-07-31
> **서비스** https://seoulro.site/

> **client 저장소** https://github.com/seoulbidata/strangemap
> **AI 서버(에이전트기능)저장소** https://github.com/seoddhh/lewisai



---
우선 

## 1. 프로젝트 소개 — 서울로는 어떤 서비스인가

서울로는 서울시 빅데이터 활용 경진대회 수상작으로 서울시 관광객과 .
사용자가 동반자·목적·권역·시간대를 고르면 **갈 곳 3~6곳을 순서와 시각까지 확정해서** 돌려준다.

기존 서울로는 Next.js 단일 구조(Vercel)였고, 이번에 코스 생성 기능만 떼어내
**Python · LangGraph · RAG 기반 AI 서버**로 재구현한 것이 이 저장소다.

| 담당 | 만드는 것 |
|---|---|
| **AI 서버 (이 저장소)** | 어떤 장소를 **왜** 골랐는지, 방문 **순서·시각**, 식사 슬롯, 예상 혼잡도, 주변 식당·행사 |
| 프론트(strangemap) | 서버가 확정한 순서 위에 지도 폴리라인·실거리만 렌더링 |


### 데이터 규모

- 임베딩 장소 **1,651곳** (서울시 열린데이터 + Visit Seoul + 문화공간 정보를 정제·통합)
- 식당 권역 캐시 9권역
- 실시간: 서울시 citydata(혼잡도·행사), Visit Seoul(식당)

---

## 2. 왜 solar-open2로 에이전트를 만들게 되었나

독파모 1차 발표회에 참가하며 upstage의 솔라를 알게되 었고 

**사실 기록 (참고용)**
- 2026-07-16 Stage 1 선정
- **2026-07-22 챗 LLM을 solar-open2로 전환** (커밋: `feat:챗 LLM을 업스테이지 솔라(solar-open2)로 전환`)
- 전환 전 구성: Gemini 우선 + Claude 폴백(`FallbackLLM`)
- Stage 1 기간 중 총 **28개 커밋**

---

## 3. 에이전트 동작 방식

### 3-1. 전체 구조

```
브라우저 → Vercel BFF(X-Internal-Token) → FastAPI AI 서버
                                              ↓
                                  LangGraph StateGraph (9노드)
                                              ↓
              Chroma(1,651청크) · meal_cache · 서울시 citydata · solar-open2
```

### 3-2. 파이프라인 — 9개 노드

```
START → parse_intent → plan → retrieve → select_places
      → fit_schedule → meals → enrich → nearby → compose → END
```

| 노드 | 하는 일 | LLM 사용 |
|---|---|:---:|
| `parse_intent` | 칩/자연어 입력을 공통 구조로 정규화 | |
| `plan` | 시간 골격(식사 앵커 + 장소 구간) 계산 | |
| `retrieve` | RAG 후보 검색 + 지리 근접 재랭킹 | |
| **`select_places`** | **후보 중 장소 선정 + 선정 이유·행동 생성** | ✅ **solar-open2** |
| `fit_schedule` | 순열 탐색으로 방문 순서·시각 확정 | |
| `meals` | 끼니 슬롯에 실제 식당 후보 부착 | |
| `enrich` | 방문 시각의 예상 혼잡도 부착 | |
| `nearby` | 스톱 주변 식당·행사 카드 | |
| **`compose`** | **코스 제목·서사·팁 작성** | ✅ **solar-open2** |

### 3-3. 설계 원칙 — "에이전트"와 "워크플로우"를 구분했다

코스 생성은 **툴콜링 에이전트가 아니라 결정적 워크플로우**로 만들었다.

- 방문 순서는 LLM이 아니라 **서버가 순열 전수 탐색**으로 정한다 (`app/core/scheduler.py`)
- LLM은 **화이트리스트 안에서만** 장소를 고를 수 있다 — 후보에 없는 이름은 버린다 (환각 차단)
- 즉 **solar-open2가 맡은 일은 "고르는 이유"와 "서사"** 이고, 순서·시각 같은 검증 가능한 계산은 코드가 한다

> 여행 코스라는 특성상 

---

## 4. solar-open2를 쓰면서 좋았던 점

> ✍️ **직접 작성이 필요한 부분** — 아래는 코드에 남아 있는 근거만 정리해둔 것. 살을 붙일 것.

### 4-1. OpenAI 호환이라 전환 비용이 거의 없었다
`ChatUpstage`가 OpenAI 호환 엔드포인트를 감싸는 구조라, 기존 Gemini/Claude 체인을
`PROMPT | get_llm()` 형태 그대로 두고 **프로바이더 함수 하나만 추가**해서 전환했다.
(`app/core/llm.py`)

### 4-2. 한국어 출력 품질
> ✍️ 실제 생성 결과를 붙여서 서술. 아래는 실측 출력 예시:
> ```
> 성수·건대 오후 산책과 야경 코스
>   살곶이길  — 조선 시대 물길 유적을 따라 조용히 산책하기 좋아서 골랐어요
>   서울숲    — 오후의 숲길을 천천히 걸으며 자연 속에서 쉬어 가기 좋은 시작점이라서
>   응봉산    — 탁 트인 도심 전경과 야경을 함께 감상할 수 있는 명소라서 골랐어요
> ```
> (입력: 성수·건대 / 연인과 / 자연·힐링 / 자유입력 "조용히 걷고 야경 볼 수 있는 곳")

### 4-3. 넉넉한 Rate Limit
400 RPM / 150,000 TPM. 배치 평가 스크립트(페르소나 10개 × arm 비교)를 돌리는 데 제약이 없었다.

### 4-4. tool_calls 규격 정상 동작
> ✍️ 검증 내용 서술

---

## 5. 아쉬웠던 점 · 개선 요청



### 추론 토큰이 max_tokens 예산을 먹고 답이 조용히 잘린다

**증상**
`solar-open2`는 추론 모델이라 `reasoning_content`가 `max_tokens` 예산을 **먼저** 소진한다.
그 결과 실제 답(`content`)이 잘리거나 **빈 문자열**로 오고 `finish_reason=length`가 된다.

**왜 위험한가 — 실패가 조용하다**
에러가 나지 않는다. 빈 `content` → JSON 파싱 실패 → 선정/서사가 **mock으로 폴백**된다.
사용자에게는 "품질이 좀 떨어지는 응답"으로 보일 뿐이라, 로그를 파기 전까지 원인을 알 수 없었다.

**실측**
| `reasoning_effort` | 응답 시간 | 결과 |
|---|---:|---|
| 기본(추론 켜짐) | **29s** | 잘림 → mock 폴백 발생 |
| `low` / `high` | ≈29s | 추론 토큰이 거의 안 줄어 효과 없음 |
| **`none`** | **3.7s** | 정상 JSON 생성 |

**→ 추론을 끄니 8배 빨라지고 품질에도 큰 차이가 없었습니다**


---

## 6. 모델 비교 — solar-open2 / Gemini / Claude(예정)

> ✍️ **직접 작성** — 동일 프롬프트로 세 모델의 출력·레이턴시를 측정해 채울 것.
> 프로바이더 전환이 `.env` 한 줄(`LLM_PROVIDER`)이라 같은 파이프라인에서 바로 비교 가능하다.

| 항목 | solar-open2 | Gemini 3.1 Flash Lite | Claude Haiku 4.5 |
|---|---|---|---|
| 코스 생성 총 소요 | | | |
| `select_places` 단계 | | | |
| `compose` 단계 | | | |
| 한국어 자연스러움 | | | |
| JSON 형식 준수율 | | | |
| 화이트리스트 준수(환각) | | | |
| 비용 | | | |

**측정 방법**
```bash
# .env 의 LLM_PROVIDER 만 바꿔가며 동일 페르소나 10개 실행
LLM_PROVIDER=upstage .venv/bin/python ab_artifacts/run_arm.py arm_solar.json
LLM_PROVIDER=gemini  .venv/bin/python ab_artifacts/run_arm.py arm_gemini.json
```

> ✍️ 정성 평가(어느 모델 서사가 더 나았는지)도 실제 출력 붙여서 서술

---

## 7. 업스테이지에 대한 관심

> ✍️ **직접 작성** — 가장 개인적인 장이고, 심사에서 사람이 읽는 부분.
> 아래는 구성 제안일 뿐이니 본인 언어로 다시 쓸 것.
>
> - 왜 업스테이지에 관심을 갖게 되었는지
> - Solar를 직접 써보고 나서 생각이 어떻게 바뀌었는지
> - Stage 2에서 해보고 싶은 것 (Solar Pro 4로 무엇을 시도할지 구체적으로)
> - 앞으로 어떤 방식으로 기여하고 싶은지

---

## 8. Stage 2에서 이어가고 싶은 것

> ✍️ 직접 작성. 후보:
> - 자유질문 경로를 진짜 툴콜링 에이전트로 구현 (현재 코스는 워크플로우, 자유질문만 에이전트로 분리 예정)
> - `reasoning_effort`를 켠 상태에서 품질이 실제로 좋아지는 구간 탐색
> - 멀티 프로바이더 폴백 복구 (`FallbackLLM`은 코드에 남겨둠)

---

## 부록 A. 실행 방법

```bash
uv sync
cp .env.example .env        # UPSTAGE_API_KEY 등 입력
python -m scripts.run_ingest        # 벡터 DB 구축
uvicorn app.main:app --reload --port 8800
# http://localhost:8800 — 검증용 챗봇 UI 제공
```

## 부록 B. 주요 파일

| 파일 | 역할 |
|---|---|
| `app/graph/build.py` | LangGraph 파이프라인 조립 |
| `app/graph/nodes/course.py` | 검색·선정·서사 — Solar 호출 지점 |
| `app/core/llm.py` | 프로바이더 레이어 (upstage \| gemini) |
| `app/core/json_parse.py` | 잘린 JSON 복구 |
| `app/core/scheduler.py` | 방문 순서 순열 탐색 |
| `docs/course-redesign.md` | 코스 기능 재설계 기록 |
| `docs/embed-quality-ab.md` | 임베딩·프롬프트 A/B 측정 |
