"""Visit Seoul API 어댑터 (표준 콘텐츠 조회). RAG 아님 — 직접 fetch 후 주입.

**역할 한정**: Visit Seoul 은 "코스에 이미 확정된 장소의 **주변** 정보"만 담당한다.
 - 주변 식당 (음식 카테고리)
 - 주변 문화행사·관광정보 (축제공연행사 + 문화/역사/자연/체험관광)
코스에 들어갈 장소 자체는 RAG(seoul_places) 가 고르고, AI 가 선정 이유를 붙인다.

- 실제 API: https://api-call.visitseoul.net (헤더 VISITSEOUL-API-KEY 인증)
    POST /api/v1/contents/list  (com_ctgry_sn/lang_code_id/keyword/sort_type/page_no)
    POST /api/v1/contents/info  (cid)  — 문서엔 GET 이라 적혀 있으나 실제로는 POST
- 키가 없으면 MockVisitSeoulClient 가 data/mock/visitseoul_sample.json 으로 응답
  → 레포 단독 실행/테스트 가능.
- 목록 API에 지리 필터가 없다 → 지역명을 keyword 로 좁혀 받고, 상세의 좌표/주소로
  반경·지역 후처리 필터링한다 (search_nearby 참고).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.config import get_settings
from app.core.geo import haversine_km

logger = logging.getLogger("lewisai.visitseoul")

# 표준 콘텐츠 카테고리 일련번호 (상위 카테고리 필터용).
# 주의: 응답의 com_ctgry_sn 은 우리가 보낸 상위 코드가 아니라 **하위 코드**(예: 음식 → Cx0t8m5
# "카페/찻집")로 돌아온다. 따라서 종류 판정은 코드가 아니라 cate_depth 문자열로 한다.
CATEGORY = {
    "문화관광": "Ca0o2d4",
    "쇼핑": "Cu8e6t5",
    "숙박": "Ch4v8z7",
    "역사관광": "Ca1z6p7",
    "음식": "Cl9s3y9",
    "자연관광": "Co6c2n2",
    "체험관광": "Cc9i5o2",
    "축제공연행사": "Cv7s8m5",
}
CAT_EVENTS = CATEGORY["축제공연행사"]
CAT_FOOD = CATEGORY["음식"]

# cate_depth(" 음식 > 카페/찻집") 앞부분 → 우리가 쓰는 종류
KIND_RESTAURANT = "restaurant"
KIND_CAFE = "cafe"
KIND_BAR = "bar"
KIND_EVENT = "event"
KIND_ATTRACTION = "attraction"

# 식사 슬롯이 쓰는 종류 — 끼니마다 구성이 다르다 (점심 식당2+카페1 / 저녁 식당2+주점1)
MEAL_KINDS = (KIND_RESTAURANT, KIND_CAFE, KIND_BAR)

_DEPTH_TO_KIND: dict[str, str] = {
    "축제/공연/행사": KIND_EVENT,
    "문화관광": KIND_ATTRACTION,
    "역사관광": KIND_ATTRACTION,
    "자연관광": KIND_ATTRACTION,
    "체험관광": KIND_ATTRACTION,
    # 쇼핑·숙박은 주변 정보 대상이 아니다 → "" (제외)
}

# "음식" 은 한 종류로 뭉치지 않는다 — 카페·주점을 점심 식당으로 추천하면 안 되고,
# 끼니별 구성도 다르기 때문. cate_depth 2번째 마디로 가른다.
#   " 음식 > 한식" / " 음식 > 외국식 > 서양식" → 식당
#   " 음식 > 카페/찻집" → 카페,  " 음식 > 주점" → 주점
_FOOD_TO_KIND: dict[str, str] = {
    "한식": KIND_RESTAURANT,
    "외국식": KIND_RESTAURANT,
    "카페/찻집": KIND_CAFE,
    "주점": KIND_BAR,
}

_LIST_TTL = 10 * 60  # 목록 10분
_DETAIL_TTL = 24 * 60 * 60  # 상세 24시간
_MOCK_JSON = Path(__file__).resolve().parents[2] / "data" / "mock" / "visitseoul_sample.json"


def classify(cate_depth: str) -> str:
    """" 음식 > 카페/찻집" → "cafe". 대상이 아니면 빈 문자열.

    " 음식"(하위분류 없음, 실측 전체의 약 14%)은 카페·주점이 아닌 일반 식당으로 본다.
    """
    parts = [p.strip() for p in (cate_depth or "").split(">")]
    if parts and parts[0] == "음식":
        if len(parts) == 1:
            return KIND_RESTAURANT
        return _FOOD_TO_KIND.get(parts[1], "")
    return _DEPTH_TO_KIND.get(parts[0] if parts else "", "")


# 우리 장소명(seoul_places)을 Visit Seoul 검색어로 바꾸는 규칙.
# 목록 API 는 제목·요약 문자열 검색이라 이름을 그대로 넣으면 헛친다:
#   "광화문·덕수궁" 0건 → "광화문" 22건 / "남산공원" 2건 → "남산" 50건
_KW_SPLIT = re.compile(r"[·/,()\[\]]|\s+")
_KW_SUFFIXES = ("한옥마을", "한강공원", "카페거리", "마을", "공원", "거리", "일대", "지구", "동")


def place_keyword(name: str) -> str:
    """장소명 → 검색어 (첫 토큰 + 접미어 제거). 남는 글자가 2자 미만이면 안 깎는다.

    "창덕궁·종묘"→"창덕궁", "북촌한옥마을"→"북촌", "성수동"→"성수", "명동"→"명동"(유지).
    """
    core = next((p for p in _KW_SPLIT.split(name) if p), name)
    for suffix in _KW_SUFFIXES:
        if core.endswith(suffix) and len(core) - len(suffix) >= 2:
            return core[: -len(suffix)]
    return core


def _norm_date(value: str) -> str:
    """"2026.07.07" → "2026-07-07" (API 가 점 구분 날짜를 준다). 빈 값은 그대로."""
    return (value or "").strip().replace(".", "-")


class VisitSeoulError(Exception):
    """429/5xx/네트워크 등 Visit Seoul 호출 실패."""


@dataclass
class VsContent:
    cid: str
    title: str
    summary: str = ""
    image: str = ""
    category: str = ""
    cate_depth: str = ""  # " 음식 > 카페/찻집" — 종류 판정의 근거

    @property
    def kind(self) -> str:
        return classify(self.cate_depth)


@dataclass
class VsDetail(VsContent):
    description: str = ""
    address: str = ""
    new_address: str = ""
    lat: float | None = None
    lng: float | None = None
    subway: str = ""
    begin_date: str = ""
    end_date: str = ""
    use_time: str = ""
    closed_days: str = ""
    tel: str = ""
    tags: list[str] = field(default_factory=list)


class BaseVisitSeoulClient(Protocol):
    source: str

    async def list_contents(
        self,
        *,
        category: str | None = None,
        keyword: str | None = None,
        lang: str = "ko",
        page_no: int = 1,
        sort_type: str = "latest",
    ) -> list[VsContent]: ...

    async def get_content(self, cid: str) -> VsDetail | None: ...


def _strip_html(html: str, max_len: int = 500) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _to_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f != 0 else None
    except (TypeError, ValueError):
        return None


class HttpVisitSeoulClient:
    source = "visitseoul"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 5.0,
        min_interval: float = 0.7,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # 키당 rate limit(~1.4 req/s) — 요청 시작 간 최소 간격을 강제.
        # 이걸 안 두면 동시/연속 상세 조회 시 서버가 500 을 뱉는다.
        self._min_interval = min_interval
        self._throttle_lock = asyncio.Lock()
        self._last_ts = 0.0
        self._transport = transport
        self._list_cache: dict[str, tuple[list[VsContent], float]] = {}
        self._detail_cache: dict[str, tuple[VsDetail | None, float]] = {}

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"VISITSEOUL-API-KEY": self._api_key},
            transport=self._transport,
        )

    async def _throttle(self) -> None:
        """요청 시작을 min_interval 간격으로 직렬화 (rate limit 회피)."""
        if self._min_interval <= 0:
            return
        async with self._throttle_lock:
            wait = self._min_interval - (time.monotonic() - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.monotonic()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """스로틀 → 요청. 429/5xx 는 1회 재시도(백오프) 후 VisitSeoulError."""
        last: Exception | None = None
        for attempt in range(2):
            await self._throttle()
            try:
                async with self._client() as client:
                    res = await client.request(method, path, **kwargs)
                if res.status_code == 200:
                    return res.json()
                last = VisitSeoulError(f"{path} → HTTP {res.status_code}")
                if res.status_code not in (429, 500, 502, 503, 504):
                    break
            except httpx.HTTPError as err:
                last = err
            if attempt == 0:
                await asyncio.sleep(0.5)
        raise VisitSeoulError(str(last))

    async def list_contents(
        self,
        *,
        category: str | None = None,
        keyword: str | None = None,
        lang: str = "ko",
        page_no: int = 1,
        sort_type: str = "latest",
    ) -> list[VsContent]:
        cache_key = f"{category}|{keyword}|{lang}|{page_no}|{sort_type}"
        cached = self._list_cache.get(cache_key)
        if cached and time.time() - cached[1] < _LIST_TTL:
            return cached[0]

        body: dict[str, Any] = {
            "lang_code_id": lang,
            "sort_type": sort_type,
            "page_no": page_no,
        }
        if category:
            body["com_ctgry_sn"] = category
        if keyword:
            body["keyword"] = keyword

        try:
            data = await self._request("POST", "/api/v1/contents/list", json=body)
        except VisitSeoulError:
            if cached:  # 만료됐어도 있으면 stale 반환
                return cached[0]
            raise

        items = [
            VsContent(
                cid=str(row.get("cid", "")),
                title=row.get("post_sj", ""),
                summary=row.get("sumry", "") or "",
                image=row.get("main_img", "") or "",
                category=row.get("com_ctgry_sn", "") or "",
                cate_depth=row.get("cate_depth", "") or "",
            )
            for row in data.get("data", []) or []
            if row.get("cid")
        ]
        self._list_cache[cache_key] = (items, time.time())
        return items

    async def get_content(self, cid: str) -> VsDetail | None:
        cached = self._detail_cache.get(cid)
        if cached and time.time() - cached[1] < _DETAIL_TTL:
            return cached[0]

        try:
            # 상세는 GET 이 아니라 POST + JSON body (GET 은 405 Method Not Allowed)
            data = await self._request("POST", "/api/v1/contents/info", json={"cid": cid})
        except VisitSeoulError:
            if cached:
                return cached[0]
            logger.warning("visitseoul detail 실패 cid=%s", cid)
            return None

        row = data.get("data") or data  # 문서상 최상위/data 래핑 모두 방어
        traffic = row.get("traffic") or {}
        extra = row.get("extra") or {}
        detail = VsDetail(
            cid=str(row.get("cid", cid)),
            title=row.get("post_sj", ""),
            summary=row.get("sumry", "") or "",
            image=row.get("main_img", "") or "",
            category=row.get("com_ctgry_sn", "") or "",
            cate_depth=row.get("cate_depth", "") or "",
            description=_strip_html(row.get("post_desc", "")),
            address=traffic.get("adres", "") or "",
            new_address=traffic.get("new_adres", "") or "",
            lat=_to_float(traffic.get("map_position_y")),
            lng=_to_float(traffic.get("map_position_x")),
            subway=traffic.get("subway_info", "") or "",
            begin_date=_norm_date(row.get("schdul_info_bgnde", "")),
            end_date=_norm_date(row.get("schdul_info_endde", "")),
            use_time=extra.get("cmmn_use_time", "") or "",
            closed_days=extra.get("closed_days", "") or "",
            tel=extra.get("cmmn_telno", "") or "",
            tags=list(row.get("tag") or []),
        )
        self._detail_cache[cid] = (detail, time.time())
        return detail


class MockVisitSeoulClient:
    """data/mock/visitseoul_sample.json 기반 오프라인 클라이언트."""

    source = "mock"

    # 픽스처는 상위 카테고리 코드를 쓰므로 실제 API 의 cate_depth 문자열로 흉내 낸다
    _CODE_TO_DEPTH = {
        CATEGORY["음식"]: " 음식 > 한식",
        CATEGORY["축제공연행사"]: " 축제/공연/행사 > 축제",
        CATEGORY["문화관광"]: " 문화관광 > 전시/미술관",
        CATEGORY["역사관광"]: " 역사관광 > 역사유적지",
        CATEGORY["자연관광"]: " 자연관광 > 공원",
        CATEGORY["체험관광"]: " 체험관광 > 체험",
        CATEGORY["쇼핑"]: " 쇼핑 > 전문매장/상가",
        CATEGORY["숙박"]: " 숙박 > 호텔",
    }

    def __init__(self, fixture_path: Path = _MOCK_JSON):
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        self._items: list[dict] = raw["contents"]

    def _depth(self, item: dict) -> str:
        return item.get("cate_depth") or self._CODE_TO_DEPTH.get(item.get("category", ""), "")

    @staticmethod
    def _haystack(item: dict) -> str:
        return " ".join(
            [item.get("title", ""), item.get("summary", ""), item.get("address", ""),
             item.get("new_address", ""), " ".join(item.get("tags", []))]
        )

    async def list_contents(
        self,
        *,
        category: str | None = None,
        keyword: str | None = None,
        lang: str = "ko",
        page_no: int = 1,
        sort_type: str = "latest",
    ) -> list[VsContent]:
        rows = self._items
        if category:
            rows = [r for r in rows if r.get("category") == category]
        if keyword:
            rows = [r for r in rows if keyword in self._haystack(r)]
        return [
            VsContent(
                cid=r["cid"],
                title=r.get("title", ""),
                summary=r.get("summary", ""),
                image=r.get("image", ""),
                category=r.get("category", ""),
                cate_depth=self._depth(r),
            )
            for r in rows
        ]

    async def get_content(self, cid: str) -> VsDetail | None:
        for r in self._items:
            if r["cid"] == cid:
                return VsDetail(
                    cid=r["cid"],
                    title=r.get("title", ""),
                    summary=r.get("summary", ""),
                    image=r.get("image", ""),
                    category=r.get("category", ""),
                    cate_depth=self._depth(r),
                    description=r.get("description", ""),
                    address=r.get("address", ""),
                    new_address=r.get("new_address", ""),
                    lat=r.get("lat"),
                    lng=r.get("lng"),
                    subway=r.get("subway", ""),
                    begin_date=r.get("begin_date", ""),
                    end_date=r.get("end_date", ""),
                    use_time=r.get("use_time", ""),
                    closed_days=r.get("closed_days", ""),
                    tel=r.get("tel", ""),
                    tags=list(r.get("tags", [])),
                )
        return None


@lru_cache
def get_visitseoul_client() -> BaseVisitSeoulClient:
    s = get_settings()
    if s.visitseoul_api_key:
        return HttpVisitSeoulClient(
            s.visitseoul_api_key,
            s.visitseoul_base_url,
            s.visitseoul_timeout,
            min_interval=s.visitseoul_min_interval,
        )
    logger.info("VISITSEOUL_API_KEY 없음 → Mock 클라이언트 사용")
    return MockVisitSeoulClient()


# ── 주변 검색 (키워드 → 상세 → 반경) ────────────────────────────────────────
#
# 목록 API 에는 지리 필터도 좌표도 없다. 대신 **카테고리 없이 키워드만** 주면 그 장소와
# 실제로 엮인 소수 정예 결과가 돌아온다 (예: keyword="경복궁" → 별빛야행/집옥재/돌담길/
# 서촌 카페, 14건). 카테고리별로 훑을 때보다 정확하고 호출도 훨씬 적다.
# 그래서: (1) 키워드(장소명·자치구)별 목록 → (2) cate_depth 로 종류 분류 →
# (3) 상세 조회(budget 만큼) → (4) 좌표 반경 필터.
# 상세는 키당 rate limit(~1.4 req/s)이라 budget 으로 건수를 제한한다.


@dataclass
class NearbyItem:
    detail: VsDetail
    dist_km: float | None  # 좌표가 없는 항목은 None
    kind: str = ""         # restaurant | event | attraction


def _matches_terms(d: VsDetail, terms: tuple[str, ...]) -> bool:
    addr = f"{d.address} {d.new_address}"
    return any(t in addr for t in terms)


async def _list_by_keywords(
    client: BaseVisitSeoulClient,
    keywords: tuple[str, ...],
    kinds: tuple[str, ...],
    lang: str,
) -> list[VsContent]:
    """키워드별 목록을 라운드로빈으로 섞는다 — 한 키워드가 상세 예산을 독식하지 않도록."""
    per_kw: list[list[VsContent]] = []
    for kw in keywords:
        try:
            rows = await client.list_contents(keyword=kw, lang=lang)
        except VisitSeoulError as err:
            logger.warning("visitseoul list 실패 keyword=%s: %s", kw, err)
            continue
        per_kw.append([r for r in rows if r.kind in kinds])

    merged: list[VsContent] = []
    seen: set[str] = set()
    for row in (c for group in zip_longest(*per_kw) for c in group):
        if row is not None and row.cid not in seen:
            seen.add(row.cid)
            merged.append(row)
    return merged


async def search_nearby(
    *,
    lat: float,
    lng: float,
    keywords: tuple[str, ...],
    kinds: tuple[str, ...],
    radius_km: float = 2.0,
    region_terms: tuple[str, ...] = (),
    limit: int = 5,
    budget: int | None = None,
    lang: str = "ko",
    client: BaseVisitSeoulClient | None = None,
) -> list[NearbyItem]:
    """좌표 주변(radius_km 이내)의 Visit Seoul 콘텐츠를 가까운 순으로 반환.

    keywords: 검색어들 — 보통 (장소명, …, 자치구). 앞쪽 키워드가 우선 예산을 받는다.
    kinds:    restaurant | event | attraction 중 원하는 종류.
    region_terms: 좌표가 없는 항목을 주소로 구제할 때 쓰는 자치구·동 이름.
    budget:   상세 조회 건수 상한 (rate limit 예산).
    """
    client = client or get_visitseoul_client()
    budget = budget or get_settings().visitseoul_detail_limit
    keywords = tuple(k for k in keywords if k)

    contents = await _list_by_keywords(client, keywords, kinds, lang)
    details = await asyncio.gather(*(client.get_content(c.cid) for c in contents[:budget]))

    items: list[NearbyItem] = []
    for content, d in zip(contents[:budget], details):
        if d is None:
            continue
        if d.lat is not None and d.lng is not None:
            dist = haversine_km(lat, lng, d.lat, d.lng)
            if dist <= radius_km:
                items.append(NearbyItem(d, dist, content.kind))
        elif region_terms and _matches_terms(d, region_terms):
            items.append(NearbyItem(d, None, content.kind))  # 좌표 없음 → 같은 지역이면 통과

    items.sort(key=lambda it: it.dist_km if it.dist_km is not None else float("inf"))
    return items[:limit]
