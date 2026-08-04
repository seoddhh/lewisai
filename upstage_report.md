# Solar Agent Partner Stage 1 — 결과물 제출

> **프로젝트** 서울로(Seoulro) AI 코스 생성 에이전트 — `lewisai`

> **참가자** 서동휘

> **모델** `solar-open2` (Solar Open 2, Private Beta)

> **기간** 2026-07-17 ~ 2026-07-31

> **사이트 주소** https://seoulro.site/

> **client 백엔드 저장소** https://github.com/seoulbidata/strangemap
> **AI 서버(에이전트기능)저장소** https://github.com/seoddhh/lewisai



---

## 1. 프로젝트 소개 — 서울로는 어떤 서비스인가

서울로는 서울시 빅데이터 활용 경진대회 최우수상 수상작으로 서울시 관광객과 서울에서 무엇을할지 사용자가 동반자,목적, 위치,시간대등을 고르면 **갈 곳을 순서와 시각까지 확정해서** 코스를 생성하는 기능을 이번 기간동안 설계했습니다 원래 기존 서울로는 Next.js 단일 구조(Vercel배포)였고, 나만의코스 기능을 에이전트로  **Python · LangGraph · RAG 기반 AI 서버**로 고도화 하기 위해 만든것이 이 레포입니다.

| 담당 | 만드는 것 |
|---|---|
| **AI 서버 (이 저장소)** | 어떤 장소를 **왜** 골랐는지, 방문 **순서·시각**, 식사 슬롯, 예상 혼잡도, 주변 식당·행사 |
| 프론트(strangemap) | 서버가 확정한 순서 위에 지도 폴리라인·실거리만 렌더링 |


### 데이터 규모

- RAG구축을 위해 장소 **1,651곳 확보**하여 임베딩 하였습니다 (서울시 열린데이터 + Visit Seoul + 문화공간 정보를 직접 정제하고 통합함)
- 식당 권역 캐시 9권역
- 실시간으로 받아오는 api: 서울시 citydata(혼잡도·행사정보 데이터)

---

## 2. 왜 solar-open2로 에이전트를 만들게 되었나

독파모 1차 발표회에 참가하며 upstage의 solar모델을 알게 되었고 많은 관심을 가지던 중에 solar를 이용해 agent를 구축할 수있는 기회로 

**사실 기록을 위해 이전에 사용하던 모델을 작성하겠습니다**
- 2026-07-16 Stage 1 선정
- **2026-07-22 챗 LLM을 solar-open2로 전환** (커밋: `feat:챗 LLM을 업스테이지 솔라(solar-open2)로 전환`)
- 전환 전 구성: Gemini 우선 + Claude 폴백(`FallbackLLM`)
- Stage 1 기간 중 기능을 업그레이드 하여 총 **28개 커밋**

---

## 3. 에이전트 동작 방식은 [README.md](https://github.com/seoddhh/lewisai/blob/main/README.md)를 참고해주세요!


### 3-1. 전체 구조

```
브라우저 → Vercel BFF(X-Internal-Token) → FastAPI AI 서버
                                              ↓
                                  LangGraph StateGraph (9노드)
                                              ↓
              Chroma(1,651청크) · meal_cache · 서울시 citydata · solar-open2
```

하지만 현재(07/31)기준 Ai 서버 호스팅 문제와 사용자 출력에 제한을 두기 위해 로그인 기능을 구현하는 중이라 실제 [서울로](https://seoulro.site/)프로덕션에는 agent기능이 적용되지 않습니다
첨부한 데모영상을 기준으로 봐주시면 감사하겠습니다.

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

> 여행 코스라는 특성상 고정된 워크플로우 방식을 사용하여 사용자 입력 단계에 맞게 출력하고 마지막 선택 자유 입력을 쿼리로 함께 보내는 방식입니다.

---

## 4. solar-open2를 쓰면서 좋았던 점


---

## 5. 아쉬웠던 점 · 개선 요청

추후작성

---

## 6. 모델 비교 — solar-open2 / solar-pro4 / Gemini / Claude /     

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

**측정 방법** — 설계는 [`docs/LLM-eval.md`](docs/LLM-eval.md) 에 정리했다.

`.env` 의 `LLM_PROVIDER`(`upstage | gemini | claude`) 한 줄만 바꿔가며 동일 페르소나 세트를 돌린다.
세 프로바이더 모두 같은 파이프라인·같은 벡터 컬렉션·같은 프롬프트를 쓴다.

- **연결 확인** — `scripts/check_upstage.py` / `scripts/check_claude.py` 로 arm 별 사전 점검.
  클로드 경로는 2026-08-04 에 복구해 코스 1건 E2E(13.5s, 폴백 없이 `source=ai`)까지 확인했다
- **제미나이 arm** — 15 RPM 계정이라 `compose` 의 동시 호출(`asyncio.gather`)이 즉시 429 를 낸다.
  평가 하네스에서 직렬화·스로틀이 필요하고, 지연 지표는 별도로 측정한다
- **지표 정의** — 위 표의 각 칸이 어떤 규칙으로 채워지는지는 평가 문서 3절 참조

---

## 7. Stage 2에 선정 된다면 develop하고 싶은점

추후작성

---

### 실행 방법과 주요 파일 디렉토리 구조와 실행 방법에 대한 내용은 README.md에 작성해 놓았습니다 추후 도커 이미지를 빌드할 계획입니다 (현재는 서버 비용 문제로 실제 사이트에선 나만의 코스 기능을 사용할수 없습니다.)
