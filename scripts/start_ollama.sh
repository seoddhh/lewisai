#!/usr/bin/env bash
# 로컬 임베딩 서버(ollama) 기동 — 서버 + 모델 pull + 메모리 로드까지 한 번에.
#
# EMBEDDING_PROVIDER=ollama 일 때만 필요하다. 인제스트뿐 아니라 런타임에도 필요하다 —
# 쿼리 임베딩은 자유입력 note 가 섞여 미리 캐시할 수 없어 요청마다 호출된다.
# 이미 떠 있으면 아무것도 하지 않고 통과한다 (여러 번 실행해도 안전).
set -euo pipefail

MODEL="${OLLAMA_EMBEDDING_MODEL:-qwen3-embedding:8b}"
BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"   # 유휴 시 모델을 메모리에 유지할 시간 (기본값은 5분)

command -v ollama >/dev/null || {
  echo "✗ ollama 가 설치되어 있지 않습니다 → https://ollama.com/download" >&2
  exit 1
}

up() { curl -sf -m 2 "$BASE_URL/api/tags" >/dev/null 2>&1; }

# 1) 서버 — 이미 떠 있으면 그대로 쓴다.
if up; then
  echo "· 서버 실행 중 ($BASE_URL)"
else
  echo "· 서버 기동 중..."
  # nohup: 이 스크립트가 끝나거나 터미널을 닫아도 서버가 살아 있도록 분리한다.
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 1 30); do
    up && break
    sleep 1
  done
  up || { echo "✗ 서버 기동 실패 — /tmp/ollama-serve.log 확인" >&2; exit 1; }
  echo "· 서버 기동 완료 (로그: /tmp/ollama-serve.log)"
fi

# 2) 모델 — 없으면 받는다 (8b 기준 4.7GB, 최초 1회).
if ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
  echo "· 모델 있음: $MODEL"
else
  echo "· 모델 내려받는 중: $MODEL (최초 1회, 수 GB)"
  ollama pull "$MODEL"
fi

# 3) 워밍업 — 첫 요청에서 로드 지연이 나지 않도록 미리 메모리에 올린다.
echo "· 모델 로드 중 (keep_alive=$KEEP_ALIVE)..."
curl -sf "$BASE_URL/api/embed" \
  -d "{\"model\":\"$MODEL\",\"input\":\"warmup\",\"keep_alive\":\"$KEEP_ALIVE\"}" \
  >/dev/null || { echo "✗ 워밍업 실패 — 모델명($MODEL)을 확인하세요" >&2; exit 1; }

echo "✓ 준비 완료 — $MODEL @ $BASE_URL"
curl -s "$BASE_URL/api/ps"
echo
