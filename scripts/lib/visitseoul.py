"""Visit Seoul API 를 부르는 코드.

여기서 가져오는 건 이미 확정된 장소의 "주변" 정보뿐이다(식당, 문화행사, 관광정보).
코스에 들어갈 장소 자체는 벡터 검색으로 고른다.

몇 가지 알아둘 점:
- 상세 조회는 문서에 GET 이라고 적혀 있지만 실제로는 POST 여야 동작한다.
- API 키가 없으면 Mock 클라이언트가 샘플 파일로 응답한다. 키 없이도 돌려볼 수 있다.
- 목록 API 에 "이 좌표 근처" 같은 조건이 없다. 그래서 지역명을 검색어로 넣어 넉넉히
  받아온 다음, 상세에 실린 좌표로 직접 걸러낸다(search_nearby 참고).
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

# 카테고리 코드. 요청할 때 "이 분류만 달라"고 넣는 값이다.
# 주의: 응답에는 우리가 보낸 코드가 아니라 하위 분류 코드가 돌아온다.
# 그래서 받은 결과의 종류를 판정할 때는 코드가 아니라 분류 이름 문자열(cate_depth)을 본다.
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
    """분류 문자열을 우리가 쓰는 종류로 바꾼다. 예: "음식 > 카페/찻집" → "cafe".

    하위 분류 없이 "음식"이라고만 온 것은 일반 식당으로 본다. 관심 없는 분류면 빈 문자열.
    """
    parts = [p.strip() for p in (cate_depth or "").split(">")]
    if parts and parts[0] == "음식":
        if len(parts) == 1:
            return KIND_RESTAURANT
        return _FOOD_TO_KIND.get(parts[1], "")
    return _DEPTH_TO_KIND.get(parts[0] if parts else "", "")


# 우리 장소 이름을 Visit Seoul 검색어로 바꾸는 규칙.
# 이 API 는 단순 문자열 검색이라 이름을 통째로 넣으면 거의 안 걸린다.
#   "광화문·덕수궁"은 0건인데 "광화문"은 22건, "남산공원"은 2건인데 "남산"은 50건.
_KW_SPLIT = re.compile(r"[·/,()\[\]]|\s+")
_KW_SUFFIXES = ("한옥마을", "한강공원", "카페거리", "마을", "공원", "거리", "일대", "지구", "동")


def place_keyword(name: str) -> str:
    """장소 이름을 검색어로 다듬는다. 첫 덩어리만 남기고 흔한 접미사를 뗀다.

    "창덕궁·종묘" → "창덕궁", "북촌한옥마을" → "북촌", "성수동" → "성수".
    떼고 나서 두 글자가 안 되면 그냥 둔다("명동"은 "명동" 그대로).
    """
    core = next((p for p in _KW_SPLIT.split(name) if p), name)
    for suffix in _KW_SUFFIXES:
        if core.endswith(suffix) and len(core) - len(suffix) >= 2:
            return core[: -len(suffix)]
    return core


def _norm_date(value: str) -> str:
    """API 가 점으로 구분해 주는 날짜를 하이픈 형식으로 바꾼다. 빈 값은 그대로 둔다."""
    return (value or "").strip().replace(".", "-")


class VisitSeoulError(Exception):
    """Visit Seoul 호출이 실패했을 때 나는 에러(호출 제한, 서버 오류, 네트워크 문제 등)."""


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


# 네이버 SmartEditor 본문(post_desc)은 <style>…</style> 블록을 품고 온다. 태그만 지우면
# 그 안의 CSS 규칙(.se-contents{…})이 텍스트로 남아 RAG 본문을 오염시킨다 → 블록째 제거.
_STYLE_SCRIPT = re.compile(r"<(style|script)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _strip_html(html: str, max_len: int = 500) -> str:
    text = _STYLE_SCRIPT.sub(" ", html or "")
    text = re.sub(r"<[^>]+>", " ", text)
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
        """요청 사이에 최소 간격을 둔다. 몰아서 부르면 호출 제한에 걸린다."""
        if self._min_interval <= 0:
            return
        async with self._throttle_lock:
            wait = self._min_interval - (time.monotonic() - self._last_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_ts = time.monotonic()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """간격을 지켜 요청을 보낸다. 서버 쪽 문제로 실패하면 한 번만 다시 시도한다."""
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
    """API 키가 없을 때 쓰는 가짜 클라이언트. 샘플 파일로 응답한다."""

    source = "mock"

    # 샘플 파일에는 분류 이름이 없고 코드만 있어서, 실제 API 가 주는 형태로 흉내 낸다
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
    """검색어별 결과를 번갈아 가며 섞는다. 한 검색어가 결과를 다 차지하지 않게."""
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
    """좌표 주변에 있는 것들을 가까운 순으로 찾는다.

    keywords: 검색어들. 보통 장소 이름과 자치구를 넣는다. 앞에 둔 것이 우선한다.
    kinds:    찾을 종류 — restaurant | event | attraction
    region_terms: 좌표가 없는 항목을 주소로라도 건지고 싶을 때 쓰는 동네 이름들
    budget:   상세 조회를 몇 번까지 할지. 호출 제한 때문에 무한정 부를 수 없다.
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
