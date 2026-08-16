# 서울로 AI (lewisai)

> 선택 몇 번으로 **시간표까지 완성된 코스**를 만들어 주는 LangGraph RAG 에이전트 서버.
> 실제 서울속 장소 1,648곳을 임베딩해 두고, 검색 → 선정 → 시간표 → 실시간 정보 → 추천이유등 9개 노드로 나눠 처리한다.

**서비스 주소**: https://seoulro.site/ 

**프론트 레포**: [seoulbidata/strangemap](https://github.com/seoulbidata/strangemap)

---

## 데모영상


[![서울로 AI 시연영상](https://img.youtube.com/vi/ltkDEYeeAxc/maxresdefault.jpg)](https://youtu.be/ltkDEYeeAxc?si=XYFXiKIZ46MjOPhk)



---

## 프로젝트 개요

기존 서울로 프로젝트는 Next.js기반의 풀스택 프레임워크를 사용하기 때문에 랭그래프와 RAG 생태계의 에이전트를 만들기 위해선 Python 언어를 기반으로 만들어야 했다. 

lewisai는 나만의 코스 기능을 에이전트로 고도화 하기 위한 프로젝트이며 실제 서울속 장소의 다양성과 더 정교한 사용자 맞춤 선택구조 + 자연어 입력 기반으로 코스가 생성되도록 하였다.

---

## 주요 기능

- **사용자 선택에 따라 완성되는 코스** — 동반·목적·위치·시간·식사수·인원·일수·혼잡도 여부를 고르면 장소 선정부터 방문 시각까지 서버가 확정한다. 자연어 한 줄(`"저녁에 연인이랑 홍대에서 야경"`)로도 같은 결과를 낸다.
- **"왜 이 장소인가"를 함께 추천한다.** — 각 스톱에 선정 이유(`reason`)와 그곳에서 할 일(`activities`)이 붙는다. 실제 시설·볼거리(`highlights`)에 따라 공원이라 산책하기 좋아요 같은 일반적인 대답을 막았다.
- **실행 가능한 시간표** — 방문 순서를 순열 전수 탐색으로 최적화하고, 남은 시간 예산을 체류시간 비율대로 나눠 방문 시각·이동수단까지 확정한다. 시간 산술은 LLM 에게 맡기지 않는다.
- **예상 혼잡도 반영** — 방문 시각 기준 서울시 **예상 혼잡도**(미래에 방문할 시간대를 기반으로 하기 때문), 진행 중인 문화행사, 주변 식당 정보를 붙인다.
- **같은 조건이면 같은 코스, "다시 만들기"는 다른 코스** — 시드를 칩+자유입력 해시로 잡아 재현 가능하게 두고, `seed` 를 실어 보내면 달라지도록 한다.(다양한 코스 생성을 위함)
- **생성 과정이 보이는 스트리밍** — SSE 로 노드별 진행 상황(`progress`)을 흘려 "지금 뭘 하고 있는지"를 UI 가 그대로 보여준다.

---

## 기술 스택 & 아키텍처

### 선택 이유

| 기술 | 왜 선택했나 |
|---|---|
| **LangGraph** (StateGraph) | 나만의 코스 기능은 사용자 선택기반으로 동작하기 때문에 확실한 워크플로우가 있고 분기에 맞게 처리해야해서 LangGraph를 선택했다. 코스 생성은 무엇을 할지가 요청 시점에 이미 정해져 있어서 모델이 다음 단계를 고를 필요가 없다. 여기에 Toolcalling 루프를 씌우면 레이턴시만 길어지게 된다 |
| **Chroma** (로컬 persistent) | 장소 1,648개는 벡터 DB를 따로 띄울 규모가 아니다(데이터 적음). sqlite 파일이라 배포 이미지에 같이 구워서 읽기 전용으로 올릴 수 있다 |
| **FastAPI + SSE** | 코스 하나 만드는 데 5~6초 정도 걸려서 진행 상황 표시가 꼭 필요했다. 서버가 보내주기만 하면 되고 양방향이 필요 없어서 WebSocket 대신 SSE를 썼다. BFF는 그대로 흘려보내기만 하면 되고 재연결도 브라우저가 알아서 한다 |
| **챗 LLM `gemini-3.1-flash-lite`** | 한국어 장소명·지명을 잘 다루고 JSON 형식으로 안정적으로 뱉는다. 다만 호출제한이 있어 프로바이더 스위치(`upstage` \| `gemini` \| `claude`)를 둬서 `LLM_PROVIDER` 값만 바꾸면 코드 수정 없이 fallback 처리가 가능하도록 했다 |
| **임베딩 Upstage `embedding-2`** | 아래 "임베딩을 로컬 GPU에서 API로 바꾼 이유" 참고. 챗 LLM과 완전히 따로 놀아서 챗 모델을 바꿔도 인덱스를 다시 구울 필요가 없다 |
| **계산 노드 분리** (`plan`/`fit_schedule`) | 시간 계산과 동선 최적화는 항상 같은 답이 나와야 한다. LLM에 맡기면 검증할 방법이 없어서 로직으로 처리한다 |
| **uv** | `pyproject.toml` + `uv.lock` 을 커밋해서 팀·배포 환경의 버전을 고정하였다. |

### 임베딩을 로컬 GPU에서 API로 바꾼 이유

개발 단계에서는 로컬 Ollama의 qwen3-embedding:8b(4096차원)로 인덱스를 구축했다. 로컬 환경이라 쿼터 제한 없이 데이터 테스트를 빠르게 반복할 수 있었다.

하지만 이 모델을 그대로 배포에 쓰기엔 비용 부담이 컸다. AI 서버에서는 벡터 임베딩 자체가 아니라, 검색어가 들어올 때마다 이를 벡터로 변환하는 임베딩 추론 요청을 처리해야 한다. 즉 qwen3-embedding:8b를 서빙하려면 GPU 인스턴스를 상시 띄워둬야 하고, 트래픽이 없는 시간에도 비용이 그대로 발생하는 구조였다.
그래서 모델을 직접 서빙하는 대신 API 임베딩으로 전환했다. 서버는 트래픽이 없으면 0으로 스케일 다운되는 Cloud Run CPU 컨테이너를 사용하고, 임베딩 비용은 실제 호출량만큼만 지불하도록 구조를 바꿨다.

API는 **Upstage `embedding-2`** 를 사용한다. 여러 임베딩 모델을 테스트 해본 결과 한국어 검색 품질이 가장 좋았으며 코스 생성 품질 또한 뛰어나다. 아래 표 참조.

| 항목 | 판단 |
|---|---|
| **한국어 검색 품질** | 검색어와 문서를 서로 다른 모델로 임베딩하는 비대칭 구조(`embedding-2-query` / `-passage`)다. `"비 오는 날 실내에서 조용히 볼 만한 전시"` 같은 문장이 실제 전시·실내 장소로 잘 걸리는 걸 확인했고, 코스 품질도 qwen3 때와 차이가 없었다 |
| **쿼터** | 2,000 RPM · 700,000 TPM에 하루 한도가 없다. 1,648청크를 통째로 다시 넣어도 기다림 없이 한 번에 끝난다 |
| **인덱스 크기** | 1024차원이라 4096차원 대비 인덱스가 1/4로 줄었다(`data/chroma` 62M → 20M). 배포 이미지도 그만큼 가벼워진다 |


## 클라이언트와 AI 서버의 연결 아키텍쳐

```mermaid
flowchart LR
    subgraph client["클라이언트 (strangemap)"]
        BR["브라우저<br/>위저드 · 지도 렌더링"]
        BFF["Vercel Next.js BFF<br/>AI_SERVER_URL 로 프록시<br/>X-Internal-Token 부착"]
    end

    subgraph server["AI 서버 — FastAPI (lewisai)"]
        MW["verify_internal_token<br/>미들웨어 · /agent/* 만"]
        R1["POST /agent/chat<br/>POST /agent/chat/stream (SSE)"]
        R2["POST /agent/course"]
        R3["GET /health"]
        G["LangGraph StateGraph<br/>선형 9노드 파이프라인"]
    end

    subgraph data["데이터 계층"]
        CH[("Chroma seoulro_v3<br/>places 1,648청크 · 1024차원")]
        MC[("meal_cache<br/>9권역 1,234곳")]
        LLM["챗 LLM<br/>gemini-3.1-flash-lite"]
        EMB["임베딩 API<br/>Upstage embedding-2"]
        SEO["서울시 열린데이터<br/>혼잡도 · 문화행사"]
    end

    VS["Visit Seoul API<br/>배포 전 캐시 구울 때만<br/>scripts/build_meal_cache.py"]

    BR -->|same-origin| BFF
    BFF -->|HTTPS| MW
    MW --> R1 --> G
    MW --> R2 --> G
    R3 -.-> EMB
    G -. 의미검색 .-> CH
    CH -. 쿼리 임베딩 .-> EMB
    G -. 식당 풀 .-> MC
    G -. 예보·행사 .-> SEO
    VS -. 배포 전 오프라인 .-> MC
    G -. select · compose .-> LLM
    G ==>|"stops[] 순서 확정"| BFF
    BFF ==>|"폴리라인 · 실거리 렌더링"| BR
```

**역할 나누기**: 어디를, 왜, 어떤 순서로 갈지는 AI 서버가 다 정한다. 프론트는 그 순서 위에 지도 SDK로 실제 도로 폴리라인을 그린다.

### LangGraph 파이프라인

노드는 순서대로 이어지는 고정 워크플로우다. 관광객/현지인, 시간대 선택 유무, 식사 유무 같은 갈래는 엣지로 나누지 않고 노드 안에서 처리한다.



```mermaid
flowchart TD
    S(["START"]) --> PI

    PI["parse_intent<br/>자연어 → 칩 구조화<br/>LLM 1콜 · 칩 진입이면 0콜"]
    PL["plan<br/>시간 골격 · 식사 앵커<br/>순수 계산"]
    RT["retrieve<br/>Chroma 의미검색 + 권역 필터<br/>개인화 재랭킹 · 종류 쿼터"]
    SP["select_places<br/>장소 선정 · reason · activities<br/>LLM 1콜 · 멀티데이는 일차별 병렬"]
    FS["fit_schedule<br/>방문 순서 · 시각 · 이동수단<br/>순수 계산"]
    ML["meals<br/>끼니별 실제 식당 3곳"]
    EN["enrich<br/>방문 시각의 혼잡도 예보"]
    NB["nearby<br/>주변 식당 · 진행중 행사 카드"]
    CP["compose<br/>제목 · 선정 근거 · 스톱 카드<br/>LLM 병렬 · 전역 1 + 일차 N + 스톱 M"]

    PI --> PL --> RT --> SP --> FS --> ML --> EN --> NB --> CP --> E(["END"])

    subgraph ext["노드 바깥"]
        direction TB
        subgraph aiModels["AI 모델"]
            direction LR
            LLM["챗 LLM<br/>gemini-3.1-flash-lite"]
            EMB["Upstage embedding-2-query<br/>1024차원 · 검색어 1개당 1콜"]
        end
        subgraph localStore["AI 서버"]
            direction LR
            CHR[("Chroma seoulro_v3<br/>places 1,648청크")]
            MCF[("data/meal_cache 권역.json<br/>9권역 1,234곳")]
        end
        subgraph seoulApi["서울시 Open API"]
            direction LR
            PPL["서울시 citydata_ppltn<br/>AREA_CONGEST_LVL · FCST_PPLTN"]
            CTY["서울시 citydata<br/>EVENT_STTS"]
        end
    end

    PI -. "칩 JSON · 1콜" .-> LLM
    RT -. "검색어 임베딩" .-> EMB
    RT -. "similarity_search_with_score k=40<br/>area · slug 필터" .-> CHR
    SP -. "장소 선정 JSON · 1콜 또는 일차별 N콜" .-> LLM
    ML -. "권역별 식당 풀 로드 (lru_cache)" .-> MCF
    EN -. "스톱의 area_name 별 · 5분 캐시" .-> PPL
    NB -. "meals 가 만든 meal_pool 재사용" .-> MCF
    NB -. "권역별 진행중 행사 · 5분 캐시" .-> CTY
    CP -. "제목 · 근거 · 스톱 카드 문구<br/>1 + N + M콜" .-> LLM

    classDef pure fill:#eaf6ea,stroke:#4a8f4a,color:#1d3d1d
    classDef ai fill:#e9eefb,stroke:#4a67b0,color:#1b2748
    classDef io fill:#fbf5e6,stroke:#b09344,color:#4a3c14
    classDef term fill:#f2f2f2,stroke:#999,color:#333
    classDef branch stroke-width:3px
    classDef net fill:#fff,stroke:#8a5a8a,color:#3a1f3a,stroke-dasharray:4 3
    classDef local fill:#ececec,stroke:#777,color:#333
    class PL,FS pure
    class PI,SP,CP ai
    class RT,ML,EN,NB io
    class S,E term
    class PL,RT,SP,FS branch
    class LLM,EMB,PPL,CTY net
    class CHR,MCF local
```

초록은 LLM 없이 계산만 하는 노드, 파랑은 LLM을 부르는 노드, 노랑은 외부 데이터를 가져오는 노드다. 테두리가 굵은 4개는 노드 안에서 갈래가 생기는 지점이다.

점선으로 빠지는 게 노드 바깥 의존이다. 테두리가 점선인 넷은 **요청 중에 네트워크를 타는 것**이고, 회색 실린더 둘은 **배포 이미지에 같이 구워 들어가는 로컬 데이터**라 네트워크를 타지 않는다.

| 노드 | 가져오는 것 | 어디서 |
|---|---|---|
| `parse_intent` | 자연어 → 칩 JSON (칩 진입이면 호출 안 함) | 챗 LLM |
| `retrieve` | 검색어 임베딩 1개 → 유사 장소 상위 40개 + 점수 | Upstage `embedding-2-query` → Chroma `seoulro_v3` |
| `select_places` | 후보 중 어디를 넣을지 + `reason` · `activities` | 챗 LLM (멀티데이는 일차별 병렬 N콜) |
| `meals` | 스톱이 속한 권역의 식당 · 카페 · 주점 풀 | `data/meal_cache/권역.json` (`lru_cache`) |
| `enrich` | 스톱 `area_name` 의 현재 혼잡도 + 12시간 예보에서 방문 시각 값 | 서울시 `citydata_ppltn` (5분 캐시) |
| `nearby` | 스톱 반경 안 식당 카드 / 권역별 진행중 행사 카드 | `meal_pool` 재사용 · 서울시 `citydata` `EVENT_STTS` (5분 캐시) |
| `compose` | 코스 제목 · 선정 근거 · 스톱 카드 문구 | 챗 LLM (전역 1 + 일차 N + 스톱 M) |

`plan` 과 `fit_schedule` 은 바깥을 전혀 보지 않는다. 9개 노드 중 LLM이 판단하는 건 3개고, 나머지는 전부 로직으로 계산한다.

### 관광객 · 현지인 분기

사람 구분은 `audience`(`local` | `tourist`) 하나뿐이고, 코드에서 실제로 갈라지는 곳은 딱 세 군데다. 나머지는 전부 같은 함수를 사용한다.

| 갈라지는 곳 | `tourist` | `local` |
|---|---|---|
| `_base_window()` — 시간 칩이 없을 때 | 기본 여행 시간을 09~21시로 설정한다(여행자는 하루 단위로 코스이기 때문) | 시간창을 만들지 않는다 → 시각 없는 자유 방문 코스 |
| `_day_areas()` — 여러 날 요청일 때 | 일차별로 권역을 나눠 배정 | 항상 한 권역만 선택하도록 |
| `summary()` 프롬프트 라벨 | `"서울 여행자"` | `"시민"` |

이 중 `_day_areas()` 결과가 있느냐 없느냐가 아래 세 노드의 동작까지 바꾼다.

```mermaid
flowchart TD
    IN["칩 확정<br/>audience · days · locations · time"] --> W

    W{"시간 칩이<br/>있나"}
    W -->|있음| WS["그 시간창을 쓴다"]
    W -->|"없음 · tourist"| WT["기본 창 09~21시"]
    W -->|"없음 · local"| WN["창 없음<br/>시각 없는 자유 방문"]

    WS --> D
    WT --> D
    WN --> D

    D{"tourist 이면서<br/>days > 1 인가"}
    D -->|"아니오 — local 또는 1일"| ONE
    D -->|예| L{"고른 위치 칩<br/>개수"}

    L -->|2개 이상| L2["고른 순서대로 하루씩"]
    L -->|1개| ONE["day_areas = None<br/>단일 권역"]
    L -->|0개| L0["목적 기반 시작 권역<br/>+ 6권역 순회"]

    L2 --> MULTI["day_areas =<br/>1일차 종로·중구 · 2일차 홍대·마포 …"]
    L0 --> MULTI

    MULTI --> RM["retrieve<br/>일차별 검색 루프 · day_hint 부착<br/>앞날에 쓴 장소는 제외"]
    ONE --> RS["retrieve<br/>단일 검색 · 앵커 1개"]

    RM --> SM["select_places<br/>일차별 LLM 병렬 N콜"]
    RS --> SS["select_places<br/>LLM 단일 1콜"]

    SM --> FM["fit_schedule<br/>앵커 = 그 일차의 권역"]
    SS --> FSS["fit_schedule<br/>앵커 = 위치 칩"]

    FM --> MG
    FSS --> MG(["합류 — 이후는 완전히 같은 코드"])
    MG --> REST["meals → enrich → nearby → compose"]

    classDef q fill:#fff6e0,stroke:#c99a3a,color:#4a3512
    classDef tour fill:#e9eefb,stroke:#4a67b0,color:#1b2748
    classDef loc fill:#eaf6ea,stroke:#4a8f4a,color:#1d3d1d
    classDef both fill:#f2f2f2,stroke:#999,color:#333
    class W,D,L q
    class WT,L2,L0,MULTI,RM,SM,FM tour
    class WN,ONE,RS,SS,FSS loc
    class IN,WS,MG,REST both
```

**일부러 합쳐 둔 것** — 갈래를 늘리지 않으려고 공유한 지점들이다.

- `locations` 를 하나만 고른 여행자는 현지인과 똑같이 단일 권역으로 간다. "그 동네에서만 놀고 싶다"는 뜻이라 나눌 이유가 없다.
- `retrieve` 의 차이는 루프를 도느냐 마느냐뿐이다. 안에서 쓰는 `_search_pool` → `_geo_rerank` → `_dedupe_same_place` → `_quota_pick` 는 완전히 같은 함수다. 일차별 호출은 시드에 일차 번호만 더한다(`seed + d`).
- `select_places` 의 병렬 N콜도 프롬프트가 다른 게 아니라 후보 목록만 일차별로 잘라 넣은 같은 콜이다. 후보에 `day_hint` 가 박혀 있어서 서로 겹칠 수가 없다.
- `fit_schedule` 은 앵커 좌표만 다르고 순열 탐색·예산 배분 로직은 하나다 (`day_areas` 가 없으면 위치 칩으로 폴백).
- `meals` 부터 `compose` 까지 4개 노드는 분기가 아예 없다. 여행자든 현지인이든 그 시점엔 "확정된 스톱 목록 + 시간표"라는 같은 모양이 되어 있기 때문이다.

### 엔드포인트

| 메서드 · 경로 | 기능 |
|---|---|
| `GET /` | 검증용 챗봇 UI (정적 페이지) |
| `GET /health` | 헬스체크 · LLM/임베딩 프로바이더 · Chroma 컬렉션 |
| `POST /agent/chat` | 통합 진입점 — `chips` 가 있으면 칩 진입, 없으면 `message` 자연어 진입 |
| `POST /agent/chat/stream` | 입력은 위와 같고 응답이 SSE. `progress`(노드 완료) → `final`(payload). 프론트가 실제로 쓰는 경로 |
| `POST /agent/course` | 코스 생성만 하는 단일 기능 (`CourseResponse`) |

---

## 폴더 구조

```
app/
  main.py              FastAPI 엔트리 · X-Internal-Token 미들웨어 · 정적 UI
  config.py            .env 단일 출처 (프로바이더·API 키·데이터 경로)
  api/routes/          health · course · chat(+SSE)
  core/                LLM·임베딩·벡터스토어 팩토리 + 순수 계산 로직
    llm.py             프로바이더 스위치 (upstage | gemini | claude)
    embeddings.py      upstage(embedding-2 비대칭) | ollama(Qwen3 쿼리 프리픽스 래퍼)
    vectorstore.py     Chroma persistent 싱글턴
    json_parse.py      LLM JSON 파싱 + 잘린 응답 부분 복구(salvage)
    geo.py             haversine · 위치 칩 9종 · 자치구→권역 매핑
    plan.py            시간 골격 (세그먼트·식사 앵커) — 순수 계산
    scheduler.py       방문 순서 순열 탐색 · 예산 배분 — 순수 계산
  graph/
    state.py           AgentState (TypedDict)
    build.py           StateGraph 조립 · run/stream 진입점 · steps 트레이스
    nodes/             common(parse_intent) · planning(plan) · schedule · meals
                       · course(retrieve/select/enrich/nearby/compose)
  rag/
    retriever.py       Chroma 메타필터 검색(+score)
    ingest.py          places.json → 임베딩 본문 조립 → 배치 인제스트
  tools/               congestion · events (서울시 citydata) · meal_cache (캐시 읽기, RAG 아님)
  features/course/     schema.py (칩·가중치·응답 계약) · chain.py (어댑터)
  static/              index.html (검증용 챗봇 UI)

data/
  embed/places.json    임베딩 세트 1,648곳 — 인제스트 소스이자 소스오브트루스
  meal_cache/          권역별 식당 캐시 9파일 1,234곳 (런타임이 직접 읽음)
  chroma/              Chroma 영속 sqlite — 컬렉션 seoulro_v3 (gitignore)
  mock/                키 없을 때 쓰는 Mock 응답

scripts/               인제스트·캐시 빌드 (run_ingest · build_meal_cache · start_ollama.sh)
  lib/visitseoul.py    Visit Seoul 클라이언트 — 캐시 구울 때만 쓴다(런타임·이미지에 없음)
                       스키마 검증 (normalize_places)
                       계측 (profile_course · profile_diversity)
                       프로바이더 점검 (check_upstage · check_claude)
tests/                 pytest 9종 — 계약 회귀 · 다양성 · 스케줄러 · 개인화 · 외부 API 어댑터
docs/                  배포 아키텍처 · LLM 프로바이더 평가 · 로드맵 · 주차별 회고
```

---

## 빠른 시작

```bash
uv sync                             # pyproject.toml + uv.lock 기준 .venv 동기화
cp .env.example .env                # UPSTAGE_API_KEY / GOOGLE_API_KEY / SEOUL_API_KEY / VISITSEOUL_API_KEY

# data/chroma 는 gitignore 라 clone 직후에는 인덱스가 없다. 한 번은 구워야 한다
uv run python -m scripts.run_ingest            # places.json → 임베딩 → Chroma (약 1분)
uv run uvicorn app.main:app --reload --port 8800
```

지금 어떤 모델과 컬렉션에 붙어 있는지는 `GET /health` 로 볼 수 있다. 임베딩 모델과 컬렉션 짝이 어긋나면 에러 없이 검색 결과만 이상해지기 때문에, 배포하고 나면 이 줄부터 확인한다.

```jsonc
{ "llm_provider": "gemini", "embedding_provider": "upstage",
  "embedding_model": "embedding-2", "chroma_collection": "seoulro_v3" }
```

```bash
uv run pytest                                                  # 단위·계약 테스트
uv run python -m scripts.profile_course                        # 노드별 레이턴시
uv run python -m scripts.profile_diversity --stage retrieve    # 다양성·개인화 (LLM 콜 0)
uv run python -m scripts.normalize_places --check              # 장소 데이터 스키마 검증
```

### 컨테이너로 띄우기

```bash
docker build -t lewisai:seoulro_v3 .
docker run --rm -p 8800:8800 --env-file .env lewisai:seoulro_v3
```

---

## 회고 / 배운 점

추후작성
