"""좌표 유틸 + 서울로 위치 칩 ↔ 지리 정보 매핑.

**라우팅 로직은 여기에 없다.** 방문 순서·지도 폴리라인·실경로 거리는 전부
strangemap 프론트(`src/lib/courseRouting.ts`)가 계산해 지도에 오버레이한다.
AI 서버는 "어떤 장소를, 왜" 만 정하고 좌표를 실어 보낸다.

거리 계산(haversine)이 남아 있는 이유는 하나뿐이다:
 - Visit Seoul 주변 검색의 반경 필터 (목록 API 에 지리 필터가 없다)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """두 좌표 사이 직선거리(km). 반경 필터·최근접 매칭 전용 (경로 거리가 아니다)."""
    R = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lng = math.radians(b_lng - a_lng)
    x = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(d_lng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def region_of(lat: float, lng: float) -> str:
    """좌표 → 권역(강북/강남/강서/강동). strangemap getRegion() 과 동일 로직."""
    if lng >= 127.05:
        return "강동"
    if lng < 126.94:
        return "강서"
    if lat < 37.52:
        return "강남"
    return "강북"


@dataclass(frozen=True)
class LocationChip:
    """위치 칩 하나 — 중심 좌표(검색 앵커)와 주소 매칭어.

    terms 는 Visit Seoul 상세의 주소 문자열에 실제로 등장하는 자치구·법정동 이름이다.
    (칩 라벨 "홍대·마포" 로는 주소가 매칭되지 않는다 — 공식 주소는 "마포구 어울마당로…")
    """

    lat: float
    lng: float
    terms: tuple[str, ...]


# 서울로 경로 생성 "위치" 칩 9종 (+ 상관없음은 칩 미선택으로 취급)
LOCATION_CHIPS: dict[str, LocationChip] = {
    "종로·중구": LocationChip(37.5720, 126.9860, ("종로", "중구", "명동", "인사동", "삼청", "을지로")),
    "강북·성북": LocationChip(37.5940, 127.0170, ("강북", "성북", "수유", "미아", "정릉")),
    "홍대·마포": LocationChip(37.5563, 126.9236, ("마포", "서교", "합정", "연남", "망원")),
    "용산·이태원": LocationChip(37.5347, 126.9947, ("용산", "이태원", "한남", "후암")),
    "여의도·영등포": LocationChip(37.5216, 126.9243, ("영등포", "여의도", "문래")),
    "강남·서초": LocationChip(37.4979, 127.0276, ("강남", "서초", "역삼", "신사", "압구정", "청담")),
    "성수·건대": LocationChip(37.5445, 127.0557, ("성동", "성수", "광진", "자양", "화양")),
    "잠실·송파": LocationChip(37.5133, 127.1000, ("송파", "잠실", "방이", "석촌")),
    "관악·사당": LocationChip(37.4767, 126.9816, ("관악", "동작", "사당", "신림", "서울대")),
}

ANY = "상관없음"


def chip_of(location: str | None) -> LocationChip | None:
    if not location or location == ANY:
        return None
    return LOCATION_CHIPS.get(location)


def chip_region(location: str | None) -> str:
    """위치 칩 → RAG 메타데이터 필터용 권역. 칩이 없으면 '상관없음'."""
    chip = chip_of(location)
    return region_of(chip.lat, chip.lng) if chip else ANY


def address_terms(location: str | None) -> tuple[str, ...]:
    """위치 칩 → 주소 후처리 필터에 쓸 자치구·법정동 이름들 (칩 라벨 포함)."""
    chip = chip_of(location)
    if not chip:
        return ()
    return (*chip.terms,)
