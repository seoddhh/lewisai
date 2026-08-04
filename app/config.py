"""환경설정. .env 를 읽어 들이며, API 키·경로·모델 선택은 전부 여기서만 나온다."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 챗에 쓸 모델 선택 — upstage | gemini | claude. 한쪽이 막히면 이 값만 바꾸면 된다.
    # 임베딩과는 무관하다(아래 embedding_provider 가 따로 있다).
    # docs/llm-provider-eval.md 의 3-arm 비교가 이 값 하나로 갈린다.
    llm_provider: str = "upstage"

    # 업스테이지 솔라
    upstage_api_key: str = ""
    upstage_model: str = "solar-open2"  # rate limit 400 RPM / 150,000 TPM
    upstage_base_url: str = "https://api.upstage.ai/v1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4000  # 3일치 코스 JSON 이 잘리지 않을 만큼

    # 솔라는 답하기 전에 먼저 "생각"하는 모델인데, 그 생각이 max_tokens 를 다 먹으면
    # 정작 본문이 빈 채로 잘려서 JSON 파싱이 실패한다. "none" 이면 생각을 건너뛰고
    # 바로 답해서 훨씬 빠르다(29초 → 3.7초). low/high 는 별 차이가 없었다.
    llm_reasoning_effort: str = "none"

    # 제미나이 — 챗과 임베딩 양쪽에서 같은 키를 쓴다
    google_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"  # rate limit 25 RPM / 하루 500회

    # 클로드 — 평가 arm C 용. 200K 컨텍스트 · 입력 $1 / 출력 $5 per MTok.
    # solar-open2·gemini-flash-lite 와 같은 빠른/저가 티어라 arm 비교가 공정하다.
    # 이 모델은 temperature 를 그대로 받는다(Opus 4.7 계열은 400 을 낸다).
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5"

    # 임베딩에 쓸 모델 선택 — gemini | ollama. 위 llm_provider 와는 별개다.
    #
    # 이 값을 바꾸면 CHROMA_COLLECTION 도 새 이름으로 바꾸고 전부 다시 인제스트해야 한다.
    # 모델이 다르면 벡터 차원부터 다르고(제미나이 3072 ↔ qwen8b 4096), 같은 컬렉션에
    # 섞으면 Chroma 가 에러를 낸다.
    embedding_provider: str = "gemini"

    # 제미나이 임베딩 — 무료 쿼터가 빡빡해서(25 RPM / 하루 500회) 전체 인제스트에 24분 넘게 걸린다
    embedding_model: str = "models/gemini-embedding-001"

    # ollama 임베딩 — 로컬이라 쿼터가 없고 더 빠르다(전체 인제스트 약 9분).
    # 미리 `ollama pull` 로 모델을 받아둬야 한다.
    ollama_embedding_model: str = "qwen3-embedding:8b"
    ollama_base_url: str = "http://localhost:11434"

    # 벡터 DB
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "seoullo"

    # BFF 만 이 서버를 부르도록 막는 공유 비밀값. 비어 있으면 검사를 건너뛴다(로컬 개발용).
    # 프로덕션에서만 채우고, .env 밖으로 나가면 안 된다.
    ai_server_token: str = ""

    # 서울 실시간 API — 혼잡도·행사 조회용
    seoul_api_key: str = ""

    # Visit Seoul API — 식당 찾기에만 쓴다. 키가 없으면 Mock 으로 동작한다.
    # (관광지는 아래 places_json, 행사는 서울시 API 담당)
    visitseoul_api_key: str = ""
    visitseoul_base_url: str = "https://api-call.visitseoul.net"
    visitseoul_timeout: float = 5.0
    visitseoul_detail_limit: int = 6          # 한 번에 상세 조회할 최대 개수
    visitseoul_min_interval: float = 0.7      # 요청 간 최소 간격(초). rate limit 에 걸리지 않으려고 둔다

    # 권역별로 식당을 미리 구워둔 캐시(scripts/build_meal_cache.py 가 만든다).
    # 실시간으로 Visit Seoul 을 여러 번 부르면 느려서, 먼저 이 캐시를 본다.
    meal_cache_dir: str = "./data/meal_cache"
    meal_pool_min: int = 8                    # 캐시에 이보다 적게 있으면 실시간 조회로 넘어간다

    # 코스에 쓰이는 장소 데이터. 이 파일이 원본이다 —
    # 장소를 고치려면 여기를 직접 고치고 다시 인제스트하면 된다.
    places_json: str = "./data/embed/places.json"

    @property
    def use_ollama_embeddings(self) -> bool:
        return self.embedding_provider.lower() == "ollama"

    @property
    def active_embedding_model(self) -> str:
        """지금 실제로 쓰는 임베딩 모델 이름. 헬스체크와 로그가 이 값만 보면 되게 한다."""
        return self.ollama_embedding_model if self.use_ollama_embeddings else self.embedding_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
