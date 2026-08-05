"""텍스트를 벡터로 바꿔주는 임베딩 모델을 만들어 준다.

`EMBEDDING_PROVIDER` 로 upstage 나 ollama 중 하나를 고른다. 챗 모델과는 별개라
챗은 솔라를 쓰면서 임베딩만 로컬(ollama)로 돌릴 수도 있다.

배포(Cloud Run)에서는 upstage 만 쓸 수 있다. ollama 는 `localhost:11434` 를 보는데
컨테이너 안엔 그런 게 없고, 8B 가중치를 이미지에 굽는 건 크기·속도 양쪽에서 안 맞는다.
인덱스는 로컬에서 미리 구워 이미지에 넣지만, 검색어를 벡터로 바꾸는 일은 요청마다
서버에서 일어나므로 서버도 같은 모델에 닿을 수 있어야 한다.

주의: 데이터를 넣을 때(인제스트)와 검색할 때가 같은 모델이어야 한다. 모델이 다르면
벡터의 생김새가 달라 거리 비교가 의미 없어진다. 모델을 바꿨다면 컬렉션 이름을 바꾸고
전부 다시 인제스트해야 한다.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings

# Qwen3-Embedding 은 검색어 앞에 이런 안내문을 붙이도록 학습된 모델이다.
# 안 붙이면 검색 성능이 조금 떨어진다. 형식은 모델 문서 그대로 옮긴 것이다.
_QWEN_QUERY_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


@lru_cache
def get_embeddings() -> Embeddings:
    """설정된 임베딩 모델을 돌려준다. 인제스트와 검색이 같은 인스턴스를 공유한다."""
    s = get_settings()
    if s.use_upstage_embeddings:
        return _upstage_embeddings()
    if s.use_ollama_embeddings:
        return _ollama_embeddings()

    # 제미나이 경로 — 무료 쿼터(25 RPM / 하루 500회)에 인제스트가 막혀서 꺼둔다.
    # 되살리려면 아래 3줄과 pyproject 의 langchain-google-genai 를 함께 살리면 된다.
    # from langchain_google_genai import GoogleGenerativeAIEmbeddings
    #
    # return GoogleGenerativeAIEmbeddings(model=s.embedding_model,
    #                                     google_api_key=s.google_api_key)
    raise ValueError(
        f"EMBEDDING_PROVIDER={s.embedding_provider!r} 는 쓸 수 없다. upstage 나 ollama 로 두면 된다."
    )


def _upstage_embeddings() -> Embeddings:
    """솔라 임베딩. 검색어와 문서를 서로 다른 모델로 부르는 비대칭 임베딩이다.

    `UpstageEmbeddings` 가 문서에는 `-passage`, 검색어에는 `-query` 접미사를 알아서
    붙여준다. 아래 Qwen 쪽에서 손으로 안내문을 붙이는 것과 같은 목적이라 여긴 손댈 게 없다.
    """
    from langchain_upstage import UpstageEmbeddings

    s = get_settings()
    return UpstageEmbeddings(
        model=s.upstage_embedding_model,
        api_key=s.upstage_api_key,
        base_url=s.upstage_base_url,
    )


def _ollama_embeddings() -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    class _QwenEmbeddings(OllamaEmbeddings):
        """검색어에만 안내문을 붙인다. 저장되는 문서 쪽은 그대로 둔다.

        문서에까지 같은 문구가 붙으면 다 비슷해져서 오히려 검색이 부정확해진다.
        """

        def embed_query(self, text: str) -> list[float]:
            return super().embed_query(_as_query(text))

        async def aembed_query(self, text: str) -> list[float]:
            return await super().aembed_query(_as_query(text))

    s = get_settings()
    return _QwenEmbeddings(
        model=s.ollama_embedding_model,
        base_url=s.ollama_base_url,
        # 모델을 안 받아둔 채 시작하면 인제스트 도중에 터진다. 시작할 때 바로 알려주는 게 낫다
        # (에러가 나면 `ollama pull <모델>` 을 먼저 하면 된다).
        validate_model_on_init=True,
    )


def _as_query(text: str) -> str:
    return f"Instruct: {_QWEN_QUERY_TASK}\nQuery:{text}"
