"""식당 권역 캐시 빌드 — Visit Seoul "음식" 카테고리 전체를 훑어 9개 권역으로 나눠 저장한다.

    PYTHONPATH=. uv run python scripts/build_meal_cache.py
    PYTHONPATH=. uv run python scripts/build_meal_cache.py --limit 30   # 빠른 확인용

결과: data/meal_cache/<권역>.json (좌표 있는 식당/카페/주점 카드 목록).
런타임(app/tools/meal_cache.py)이 이 파일을 읽어 API 호출 없이 식당 풀을 만든다.

방식: 권역 좌표 중심 "키워드 검색 + 반경 필터"가 아니라 "음식" 카테고리 **전체**를
페이지네이션으로 끝까지 훑는다. 키워드 검색은 제목·요약 문자열 매칭이라 다수를
놓친다(권역 앵커+반경 방식 실측: 서울 전체 96곳뿐 — 실제로는 1200여 건 존재).
전체를 훑고 좌표로 가장 가까운 권역 칩에 배정하면 커버리지가 최대가 된다.

Visit Seoul 상세는 키당 rate limit(~1.4 req/s, 클라이언트가 자동 스로틀)이라 1200여
건을 다 받으면 15분 안팎 걸린다 — 오프라인 빌드라 요청 지연과 무관하니 감수한다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.core.geo import LOCATION_CHIPS, nearest_chip
from app.tools.meal_cache import meal_card
from app.tools.visitseoul import CATEGORY, MEAL_KINDS, VsContent, classify, get_visitseoul_client

OK, WARN = "✅", "⚠️"


async def _list_all_food(client) -> list[VsContent]:
    """"음식" 카테고리 전체를 끝까지 페이지네이션."""
    rows: list[VsContent] = []
    page = 1
    while True:
        batch = await client.list_contents(category=CATEGORY["음식"], page_no=page)
        if not batch:
            break
        rows.extend(batch)
        page += 1
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="상세 조회 건수 상한 (테스트용)")
    args = ap.parse_args()

    s = get_settings()
    client = get_visitseoul_client()
    if client.source != "visitseoul":
        print(f"{WARN} VISITSEOUL_API_KEY 가 없어 Mock 데이터로 빌드됩니다 (.env 확인).")

    print("① 음식 카테고리 전체 목록 수집 중…")
    rows = await _list_all_food(client)
    targets = [r for r in rows if classify(r.cate_depth) in MEAL_KINDS]
    if args.limit:
        targets = targets[: args.limit]
    print(f"   목록 {len(rows)}건 중 식당/카페/주점 {len(targets)}건 → 상세 조회 시작"
          f" (예상 {len(targets) * 0.7 / 60:.1f}분)")

    buckets: dict[str, list[dict]] = {area: [] for area in LOCATION_CHIPS}
    seen_titles: set[str] = set()
    start = time.monotonic()
    for i, row in enumerate(targets, 1):
        detail = await client.get_content(row.cid)
        if detail is None or detail.lat is None or detail.lng is None or detail.title in seen_titles:
            continue
        seen_titles.add(detail.title)
        area = nearest_chip(detail.lat, detail.lng)
        buckets[area].append(meal_card(detail, classify(row.cate_depth)))
        if i % 100 == 0 or i == len(targets):
            print(f"   {i}/{len(targets)}건 처리 ({time.monotonic() - start:.0f}초 경과)")

    out_dir = Path(s.meal_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for area, cards in buckets.items():
        (out_dir / f"{area}.json").write_text(
            json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        total += len(cards)
        print(f"{OK} {area}: 식당 {len(cards)}곳")

    print(f"\n🍚 완료 — 좌표 있는 식당/카페/주점 {total}곳 (원본 {len(targets)}건 중 좌표 없음·중복 제외).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
