"""환경설정 — .env 에서 로드. 모든 키/경로/공급자 선택의 단일 출처."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 제미나이
    google_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2500

    # 클로드 폴백 (제미나이 쿼터 소진/429 시 자동 전환)
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5"
    llm_fallback_enabled: bool = True

    # 임베딩 (제미나이 네이티브)
    embedding_model: str = "models/gemini-embedding-001"

    # 벡터 DB
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "seoullo"

    # 서울 실시간 API
    seoul_api_key: str = ""

    # Visit Seoul API (키 없으면 Mock 클라이언트로 동작)
    visitseoul_api_key: str = ""
    visitseoul_base_url: str = "https://api-call.visitseoul.net"
    visitseoul_timeout: float = 5.0
    visitseoul_detail_limit: int = 6          # 상세 조회 상한 (rate limit·지연 균형)
    visitseoul_min_interval: float = 0.7      # 요청 시작 최소 간격(초) — 키당 rate limit 회피

    # 데이터
    places_json: str = "./data/raw/seoul_places.json"
    courses_json: str = "./data/raw/theme_courses.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
