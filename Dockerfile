# 서울로 AI 서버(lewisai) 컨테이너 이미지 — Cloud Run 배포용.
#
# 벡터 인덱스(data/chroma)를 이미지에 함께 굽는다. 컨테이너 안에서는 읽기만 하고
# 인제스트는 하지 않는다 — Cloud Run 파일시스템은 인스턴스가 죽으면 사라지고,
# 인스턴스가 여러 개로 늘어나면 각자 다른 인덱스를 갖게 되기 때문이다.
# 장소를 고쳤다면 로컬에서 다시 인제스트한 뒤 이미지를 새로 구우면 된다.
#
#   docker build -t lewisai:latest .
#   docker run --rm -p 8800:8800 --env-file .env lewisai:latest

# ── 1단계: 의존성만 설치한다 ──────────────────────────────────────────────
# 앱 코드와 분리해야 코드만 고쳤을 때 이 무거운 층을 캐시에서 그대로 재사용한다.
FROM python:3.13-slim AS builder

# 로컬에서 uv.lock 을 만든 버전과 맞춰 둔다(로컬 uv 0.8.4).
COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# 락 파일만 먼저 넣고 설치한다. 코드가 바뀌어도 이 층은 다시 안 돈다.
COPY pyproject.toml uv.lock ./
# --frozen: 락 파일을 그대로 따른다(빌드 중에 조용히 버전이 바뀌지 않게).
# --no-dev: pytest 같은 개발 도구는 이미지에 넣지 않는다.
RUN uv sync --frozen --no-dev --no-install-project

# ── 2단계: 실행 이미지 ────────────────────────────────────────────────────
FROM python:3.13-slim

# 루트로 돌리지 않는다. 인덱스도 읽기만 하므로 쓰기 권한이 필요 없다.
RUN useradd --create-home --uid 1000 app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# 앱 코드와, 서버가 읽는 데이터만 넣는다.
#   chroma    — 벡터 인덱스(seoulro_v3 · embedding-2 · 1024차원)
#   embed     — 장소 원본. 코스 응답이 여기 필드를 그대로 읽는다
#   meal_cache— 권역별 식당 캐시. 없으면 매 요청마다 Visit Seoul 을 실시간으로 부른다
COPY --chown=app:app app/ ./app/
COPY --chown=app:app data/chroma/ ./data/chroma/
COPY --chown=app:app data/embed/ ./data/embed/
COPY --chown=app:app data/meal_cache/ ./data/meal_cache/

USER app

# Cloud Run 은 $PORT 로 포트를 알려준다. 로컬에서 그냥 띄우면 8800 을 쓴다.
ENV PORT=8800
EXPOSE 8800

# 컨테이너에는 ollama 가 없다. 임베딩은 반드시 솔라(upstage)여야 하고,
# 컬렉션은 위에서 구운 인덱스와 같은 이름이어야 한다. 배포 env 가 빠져도
# 엉뚱한 컬렉션을 보지 않도록 여기서 한 번 더 못 박는다.
ENV EMBEDDING_PROVIDER=upstage \
    CHROMA_DIR=/app/data/chroma \
    CHROMA_COLLECTION=seoulro_v3 \
    PLACES_JSON=/app/data/embed/places.json \
    MEAL_CACHE_DIR=/app/data/meal_cache

# 워커는 1개다. Cloud Run 은 인스턴스를 늘려서 부하를 감당하므로
# 컨테이너 안에서 프로세스를 늘리면 메모리만 배로 먹는다.
# JSON 형식으로 둬야 컨테이너가 받은 종료 신호가 uvicorn 까지 그대로 간다.
# $PORT 를 펼쳐야 해서 sh -c 를 거치되, exec 로 넘겨 uvicorn 이 PID 1 이 되게 한다.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
