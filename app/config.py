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

    # 임베딩 (제미나이 네이티브)
    embedding_model: str = "models/gemini-embedding-001"

    # 벡터 DB
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "seoullo"

    # 서울 실시간 API
    seoul_api_key: str = ""

    # 데이터
    places_json: str = "./data/raw/seoul_places.json"
    courses_json: str = "./data/raw/theme_courses.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
