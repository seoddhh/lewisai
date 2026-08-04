"""좌표 계산과 지역 칩(동네) 관련 도구 모음.

여기 있는 거리 계산은 전부 직선거리다. "이 반경 안에 있나", "어느 동네에 가까운가"를
따지는 용도라 그걸로 충분하다. 실제 도로 경로는 프론트가 그린다.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


def haversine_km(a_lat: float, a_lng: float, b_lat: float, b_lng: float) -> float:
    """두 좌표 사이의 직선거리(km). 실제로 걸어가는 거리가 아니라 지도 위 직선거리다."""
    R = 6371.0
    d_lat = math.radians(b_lat - a_lat)
    d_lng = math.radians(b_lng - a_lng)
    x = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(d_lng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


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

# 서울 25개 자치구 → 9권역 칩 (권위 매핑 — 주소의 자치구가 area 의 정답이다).
# 좌표 nearest_chip 은 한강·칩경계에서 강 건너로 튀므로(예: 압구정 좌표가 성수 칩에 붙음),
# 주소에 자치구가 있으면 항상 이 표를 우선한다. nearest_chip 은 주소가 없을 때만 폴백.
#  - 중앙 14구: 칩과 1:1
#  - 9칩 밖 11구: 인접 자치구 기준 흡수 (사용자 확정 정책)
DISTRICT_TO_CHIP: dict[str, str] = {
    # 중앙 14구 (칩 직속)
    "종로구": "종로·중구", "중구": "종로·중구",
    "강북구": "강북·성북", "성북구": "강북·성북",
    "마포구": "홍대·마포",
    "용산구": "용산·이태원",
    "영등포구": "여의도·영등포",
    "강남구": "강남·서초", "서초구": "강남·서초",
    "성동구": "성수·건대", "광진구": "성수·건대",
    "송파구": "잠실·송파",
    "관악구": "관악·사당", "동작구": "관악·사당",
    # 9칩 밖 11구 → 인접 칩 흡수
    "서대문구": "홍대·마포", "은평구": "홍대·마포",
    "동대문구": "종로·중구",
    "중랑구": "성수·건대",
    "노원구": "강북·성북", "도봉구": "강북·성북",
    "강동구": "잠실·송파",
    "강서구": "여의도·영등포", "양천구": "여의도·영등포", "구로구": "여의도·영등포",
    "금천구": "관악·사당",
}

_DISTRICT_RE = re.compile(r"([가-힣]+구)")


def chip_of_address(address: str | None) -> str | None:
    """주소에서 자치구를 찾아 지역 칩 이름으로 바꾼다. 못 찾으면 None.

    장소의 area 는 이 방식으로 정하는 게 맞다. 좌표로 가까운 칩을 찾는 것보다 정확하다
    (강 건너 동네가 더 가까울 수 있다).

    지금 실행 중에 부르는 곳은 없다. places.json 에 장소를 손으로 추가할 때
    area 를 무엇으로 둘지 정하는 기준으로 참고하는 함수다.
    """
    if not address:
        return None
    m = _DISTRICT_RE.search(address)
    return DISTRICT_TO_CHIP.get(m.group(1)) if m else None


def chip_of(location: str | None) -> LocationChip | None:
    if not location or location == ANY:
        return None
    return LOCATION_CHIPS.get(location)


def address_terms(location: str | None) -> tuple[str, ...]:
    """위치 칩 → 주소 후처리 필터에 쓸 자치구·법정동 이름들 (칩 라벨 포함)."""
    chip = chip_of(location)
    if not chip:
        return ()
    return (*chip.terms,)


def nearest_chip(lat: float, lng: float) -> str:
    """좌표에서 가장 가까운 위치 칩(권역)의 이름. 식당 권역 캐시 키로 쓴다."""
    return min(LOCATION_CHIPS, key=lambda k: haversine_km(lat, lng, LOCATION_CHIPS[k].lat, LOCATION_CHIPS[k].lng))


def nearest_terms(lat: float, lng: float, n: int = 2) -> tuple[str, ...]:
    """좌표에서 가장 가까운 권역 칩의 동네 매칭어 앞 n개.

    장소 데이터에 주소가 없어서, Visit Seoul 키워드 검색을 그 장소의 동네로
    넓힐 때 쓴다 (예: DDP 좌표 → 종로·중구 → "종로", "중구").
    """
    return LOCATION_CHIPS[nearest_chip(lat, lng)].terms[:n]
