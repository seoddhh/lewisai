"""환경설정 — .env 에서 로드. 모든 키/경로/공급자 선택의 단일 출처."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 챗 LLM 프로바이더 선택 — upstage | gemini.
    # 솔라 Private Beta 가 끝나거나 크레딧이 소진되면 코드를 고치지 않고 이 값만 gemini 로
    # 바꿔 제미나이 챗 경로(_gemini)로 넘어간다. 임베딩은 이 값과 무관하게 항상 제미나이다.
    llm_provider: str = "upstage"

    # 업스테이지 솔라 (챗 LLM — 현행 프로바이더)
    upstage_api_key: str = ""
    # Solar Open 2 (Private Beta, Solar Agent Partner Stage 1 — 2026-07-17~07-31)
    # Rate limit: 400 RPM / 150,000 TPM 호출제한 없음
    upstage_model: str = "solar-open2"
    upstage_base_url: str = "https://api.upstage.ai/v1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4000  # 3일치 select(장소 9곳 × 이유+행동) JSON 이 안 잘리도록 여유
    # solar-open2 는 추론(reasoning) 모델 — 사고 과정(reasoning_content)이 max_tokens
    # 예산을 먼저 먹어 실제 답(content)이 잘린다(finish_reason=length → content 빈 문자열
    # → JSON 파싱 실패 → 선정/서사가 조용히 mock 으로 폴백). "none" 이면 추론을 끄고
    # 즉시 JSON 을 뱉어 정상 생성 + 레이턴시 8배 단축(실측: 29s→3.7s). low/high 는
    # 추론 토큰이 거의 안 줄어 효과 없음(실측). 값: none|low|high (flat, OpenAI 스타일).
    llm_reasoning_effort: str = "none"

    # gemini 3.1flash-lite
    # Rate limit: 25 RPM / 250k TPM 하루 호출제한 500
    google_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"

    # 클로드 폴백 (제미나이 쿼터 소진/429 시 자동 전환) — 솔라 전환으로 비활성
    # anthropic_api_key: str = ""
    # claude_model: str = "claude-haiku-4-5"
    # llm_fallback_enabled: bool = True

    # 임베딩 (제미나이 네이티브 — 기존 Chroma 인덱스와 호환 유지 위해 그대로 둔다)
    embedding_model: str = "models/gemini-embedding-001"

    # 벡터 DB
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "seoullo"

    # 내부 인증 — BFF(strangemap)만 /agent/* 를 호출하도록 하는 공유 시크릿.
    # 비어 있으면(로컬 개발 등) 검증을 건너뛴다. 프로덕션에서만 채운다.
    # 코드 하드코딩 금지 — .env 에만 두고 BFF 와 같은 값을 공유한다.
    ai_server_token: str = ""

    # 서울 실시간 API
    seoul_api_key: str = ""

    # Visit Seoul API (키 없으면 Mock 클라이언트로 동작) — 이제 "식당(음식)" 전용.
    # 문화/역사/자연 관광은 임베딩 세트(seoul_places)로, 행사는 서울시 citydata 로 간다.
    visitseoul_api_key: str = ""
    visitseoul_base_url: str = "https://api-call.visitseoul.net"
    visitseoul_timeout: float = 5.0
    visitseoul_detail_limit: int = 6          # 상세 조회 상한 (rate limit·지연 균형)
    visitseoul_min_interval: float = 0.7      # 요청 시작 최소 간격(초) — 키당 rate limit 회피

    # 식당 권역 캐시 — scripts/build_meal_cache.py 가 권역별로 미리 구워둔 식당 풀.
    # 런타임 nearby 는 이 캐시를 먼저 읽어 Visit Seoul 상세 fan-out(=지연)을 0으로 만든다.
    meal_cache_dir: str = "./data/meal_cache"
    meal_pool_min: int = 8                    # 캐시 풀이 이보다 얇으면 라이브 조회로 폴백

    # 데이터 — 코스 RAG 의 유일한 원천. 원천 3종(seoul_places·culture·poi)을 정제·통합해
    # 방문가치 없는 곳을 걸러낸 결과물이며, 빌드 파이프라인은 폐기했다(2026-07-28).
    # 이제 이 파일 자체가 소스오브트루스다 — 장소 수정은 여기를 직접 고치고 재인제스트한다.
    places_json: str = "./data/embed/places.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
