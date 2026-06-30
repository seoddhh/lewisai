"""환경설정 — .env 에서 로드. 모든 키/경로/공급자 선택의 단일 출처."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["nim", "openai", "gemini", "local"] = "nim"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str = ""
    llm_model: str = "z-ai/glm-5.1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2500

    # Gemini 네이티브
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # 임베딩
    embedding_model: str = "BAAI/bge-m3"

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
