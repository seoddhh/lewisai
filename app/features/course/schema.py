"""course(테마코스) 입출력.

장소 선정과 "왜 이 장소인지"(reason/activities)는 AI 가, 방문 순서·지도 폴리라인은
strangemap 프론트(courseRouting.ts)가 담당한다 — 서버는 좌표만 실어 보낸다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── 서울로 경로 생성 칩 (트리플식 단계별 위저드와 어휘 1:1) ──────────────────
Audience = Literal["local", "tourist"]   # 서울 시민 | 서울 여행자
COMPANIONS = ("혼자", "친구와", "연인과", "배우자와", "아이와", "부모님과")
Companion = Literal["혼자", "친구와", "연인과", "배우자와", "아이와", "부모님과"]
TimeSlot = Literal["오전", "오후", "밤"]
Location = Literal[
    "종로·중구", "강북·성북", "홍대·마포", "용산·이태원", "여의도·영등포",
    "강남·서초", "성수·건대", "잠실·송파", "관악·사당", "상관없음",
]
CongestionPref = Literal["여유", "보통", "상관없음"]
Pace = Literal["packed", "relaxed"]      # 알찬/빼곡 | 여유로운/널널 일정

# 목적 칩 — audience 별 어휘가 다르다 (로컬: 오늘의 목적 / 여행자: 여행의 목적).
# 각 목적은 task 조건을 갖는다:
#  - query : 검색 확장어 (RAG 후보 검색)
#  - rule  : 장소 "선정" 렌즈 — 어떤 장소를 고를지 (retrieve/select 단계)
#  - act   : 행동 "생성" 렌즈 — 그 장소에서 무엇을 하는지(activities)의 결.
#            동반자(COMPANION_RULES)와 곱해져 같은 장소라도 목적마다 다른 행동이 나오게 한다.
#            특정 상호가 아니라 '행동 유형'으로 써서 환각을 막는다.
PURPOSE_RULES: dict[str, dict] = {
    # 로컬 — 오늘의 목적은?
    "힐링": {"query": "조용한 산책 자연 휴식",
             "rule": "번잡한 상권보다 공원·한강·정원처럼 쉬어 갈 수 있는 곳을 우선하라.",
             "act": "물멍·숲길 산책·벤치에서 쉬기처럼 속도를 늦추고 자연을 느끼는 행동"},
    "놀거리": {"query": "놀거리 체험 활기찬 재미",
              "rule": "구경만 하는 곳보다 직접 해볼 거리가 있는 활기찬 장소를 우선하라.",
              "act": "직접 해보는 체험·게임·즉석 먹거리처럼 몸으로 부딪혀 즐기는 활동적인 행동"},
    "데이트": {"query": "데이트 분위기 야경 감성",
              "rule": "둘이 걷기 좋고 분위기 있는 장소를 우선하라.",
              "act": "함께 걷는 산책·야경 감상·사진 찍기·감성 카페처럼 둘의 분위기를 나누는 행동"},
    "관광": {"query": "서울 대표 명소 관광",
            "rule": "서울을 대표하는 명소를 우선하라.",
            "act": "대표 포토스폿에서 사진·랜드마크 둘러보기·주변 명소 도보 산책처럼 명소를 눈에 담는 행동"},
    "문화생활": {"query": "전시 공연 미술관 박물관 문화",
               "rule": "전시·공연·박물관처럼 볼거리 콘텐츠가 있는 장소를 우선하라.",
               "act": "전시·상설관 관람·도슨트 듣기·작품 앞에서 이야기 나누기처럼 콘텐츠를 감상하는 행동"},
    # 여행자 — 여행의 목적은?
    "체험·액티비티": {"query": "체험 액티비티 클래스 만들기",
                  "rule": "직접 참여하는 체험이 있는 장소를 우선하라.",
                  "act": "원데이 클래스·만들기 체험·참여형 액티비티처럼 손으로 직접 해보는 행동"},
    "핫플레이스": {"query": "핫플레이스 SNS 감성 트렌디 카페",
                "rule": "요즘 SNS에서 찾는 트렌디한 곳을 우선하라.",
                "act": "요즘 뜨는 카페·팝업스토어·포토존에서 사진처럼 트렌드를 즐기고 남기는 행동"},
    "자연 힐링": {"query": "자연 공원 한강 숲 휴식",
               "rule": "도심 속 자연을 느끼며 쉴 수 있는 곳을 우선하라.",
               "act": "강변·공원 산책·잔디에서 쉬기·자연 풍경 감상처럼 도심 속 자연을 누리는 행동"},
    "유명 관광지": {"query": "서울 필수 관광 명소 랜드마크",
                "rule": "처음 온 여행자가 놓치면 아쉬운 대표 명소를 우선하라.",
                "act": "랜드마크 관람·전망 감상·대표 포토스폿 촬영처럼 필수 명소를 도는 행동"},
    "문화·예술·역사": {"query": "고궁 역사 전시 미술관 전통",
                   "rule": "역사와 예술을 깊게 볼 수 있는 곳을 우선하라.",
                   "act": "고궁·전시 관람·해설 투어·전통 체험처럼 역사와 예술을 깊게 들여다보는 행동"},
    "쇼핑": {"query": "쇼핑 시장 상권 편집숍",
            "rule": "쇼핑 상권·시장을 우선하라.",
            "act": "상권·시장 둘러보기·편집숍 구경·기념품 고르기처럼 사고 구경하는 행동"},
    "맛집 탐방": {"query": "맛집 먹자골목 시장 음식",
               "rule": "먹을거리가 풍부한 시장·골목 상권을 우선하라.",
               "act": "먹자골목·시장 먹거리 탐방·현지 맛집 식사처럼 먹는 즐거움을 좇는 행동"},
}

# 동반자 칩 — 검색(장소 선정)에는 관여하지 않고, "거기서 무엇을 하는지"(activities)와
# 선정 이유·서사의 이유로만 쓴다. 임베딩 장소가 지역구 단위(예: "잠실·송파")라 장소 태깅 대신 목적 + 동행자로 생성
# 예: 친구와 잠실 → 야구 관람·보드게임 / 연인과 잠실 → 석촌호수 산책·팝업 구경.
COMPANION_RULES: dict[str, str] = {
    "혼자": "몰입·자유 — 사색·전시 관람·독립적으로 즐기는 활동. '함께/둘이' 표현은 쓰지 말 것.",
    "친구와": "함께 즐기는 활동·체험형 — 관람·게임·먹거리 탐방·같이 찍는 사진.",
    "연인과": "둘만의 분위기 — 산책·야경·팝업/전시 구경·감성 카페.",
    "배우자와": "편안한 일상 데이트 — 여유로운 식사·가벼운 산책·전시.",
    "아이와": "체험·안전·교육형 — 체험학습·넓고 개방된 공간·간식. 늦은 밤·주점류는 제외.",
    "부모님과": "여유롭고 정적인 — 고궁·정원·전통·앉아서 쉴 곳, 이동은 최소로.",
}
LOCAL_PURPOSES = ("힐링", "놀거리", "데이트", "관광", "문화생활")
TOURIST_PURPOSES = ("체험·액티비티", "핫플레이스", "자연 힐링", "유명 관광지",
                    "문화·예술·역사", "쇼핑", "맛집 탐방")
Purpose = Literal[
    "힐링", "놀거리", "데이트", "관광", "문화생활",
    "체험·액티비티", "핫플레이스", "자연 힐링", "유명 관광지",
    "문화·예술·역사", "쇼핑", "맛집 탐방",
]

# 칩 시간대 → 시간 범위 (시). 칩보다 구체적인 time_window 가 있으면 그쪽이 우선.
_TIME_SLOT_WINDOWS: dict[str, tuple[int, int]] = {"오전": (9, 12), "오후": (12, 18), "밤": (18, 23)}


class TimeWindow(BaseModel):
    """코스 시간 범위 (예: 오후 2시~8시 = start 14, end 20). end 는 자정 넘김 허용(28=새벽 4시)."""

    start: int = Field(ge=0, le=24)
    end: int = Field(ge=1, le=28)

    def overlaps(self, op_start: int, op_end: int) -> bool:
        """이 시간 범위에 운영시간(op_start~op_end)이 겹치는가. 상시(0,24)·심야(예: 5~익일1시) 대응."""
        if (op_start, op_end) == (0, 24):
            return True
        o_end = op_end if op_end > op_start else op_end + 24
        # 자정을 넘는 구간끼리 비교할 수 있게 운영 구간을 하루 뒤로도 밀어 본다
        for shift in (0, 24):
            if max(op_start + shift, self.start) < min(o_end + shift, self.end):
                return True
        return False

    def label(self) -> str:
        return f"{self.start}시~{self.end % 24}시" if self.end > 24 else f"{self.start}시~{self.end}시"


# 여행자는 시각 선택이 없다 — 하루 전체(09~21시)를 기본 창으로 스케줄·식사 슬롯을 만든다
_TOURIST_DAY_WINDOW = (9, 21)


class CourseChips(BaseModel):
    """프론트 칩 선택값. 모두 선택 사항 — 비면 자연어(note)만으로 생성한다.

    companions/purposes/locations 는 중복 선택 가능 — 프론트 위저드가 여러 칩을 담아 보낸다.
    """

    audience: Audience | None = None
    companions: list[Companion] = Field(default_factory=list)
    time: TimeSlot | None = None
    purposes: list[Purpose] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    congestion: CongestionPref | None = None
    place_count: int = Field(4, ge=3, le=5, description="장소 수 칩 (3~5곳)")
    days: int = Field(1, ge=1, le=6, description="여행 일수 (당일치기=1 ~ 5박6일=6)")
    pace: Pace | None = None
    # 구체적 시간 범위 — 프론트 시간 선택 칩 또는 자연어 파싱("오후 2시부터 8시까지")에서 채워진다
    time_window: TimeWindow | None = None

    def resolved_window(self) -> TimeWindow | None:
        """스케줄/운영시간 필터에 쓸 시간 범위 — 구체 범위 > 칩 시간대 > 여행자 기본 > 없음."""
        if self.time_window:
            return self.time_window
        if self.time in _TIME_SLOT_WINDOWS:
            start, end = _TIME_SLOT_WINDOWS[self.time]
            return TimeWindow(start=start, end=end)
        if self.audience == "tourist":
            return TimeWindow(start=_TOURIST_DAY_WINDOW[0], end=_TOURIST_DAY_WINDOW[1])
        return None

    def stops_per_day(self) -> int:
        """하루 장소 수 — 일정 밀도 칩이 있으면 그쪽이 우선 (빼곡=5, 널널=3)."""
        if self.pace == "packed":
            return 5
        if self.pace == "relaxed":
            return 3
        return self.place_count

    def purpose_rule(self) -> dict:
        """선택된 목적들의 task 조건을 하나로 합친다.

        query·rule 은 장소 선정용, act 은 행동(activities) 생성용 렌즈.
        여러 목적을 골랐으면 각 문장을 중복 제거해 병기한다.
        """
        rules = [PURPOSE_RULES[p] for p in self.purposes if p in PURPOSE_RULES]
        if not rules:
            return {}
        return {
            "query": " ".join(dict.fromkeys(r["query"] for r in rules)),
            "rule": " ".join(dict.fromkeys(r["rule"] for r in rules)),
            "act": " / ".join(dict.fromkeys(r["act"] for r in rules)),
        }

    def companion_rule(self) -> str:
        """선택된 동반자들의 관계 렌즈를 합쳐 activities·서사 생성에 건다 (검색엔 미관여)."""
        return " ".join(
            dict.fromkeys(
                COMPANION_RULES[c] for c in self.companions if c in COMPANION_RULES
            )
        )

    def summary(self) -> str:
        """프롬프트/트레이스에 넣을 한 줄 요약."""
        window = self.resolved_window()
        parts = [
            {"local": "서울 시민", "tourist": "서울 여행자"}.get(self.audience or ""),
            f"여행 기간: {self.days}일" if self.days > 1 else "",
            f"동반: {'·'.join(self.companions)}" if self.companions else "",
            f"시간대: {self.time}" if self.time else "",
            f"시간 범위: {window.label()}" if window else "",
            f"목적: {'·'.join(self.purposes)}" if self.purposes else "",
            f"위치: {'·'.join(self.locations)}" if self.locations else "",
            f"혼잡도 선호: {self.congestion}" if self.congestion else "",
            {"packed": "일정: 빼곡하게", "relaxed": "일정: 여유롭게"}.get(self.pace or ""),
            f"하루 {self.stops_per_day()}곳",
        ]
        return ", ".join(p for p in parts if p)


class CourseRequest(BaseModel):
    note: str = Field("", description="자유서술 (예: 야경 보면서 데이트)")
    chips: CourseChips = Field(default_factory=CourseChips)
    # 구 계약 호환 — 칩이 없을 때만 사용 (region 은 강북/강남/강서/강동 권역)
    region: str = "상관없음"
    time: str = ""


class NearbyCard(BaseModel):
    """Visit Seoul 주변 정보 카드 (식당 / 문화행사·관광)."""

    title: str
    summary: str = ""
    address: str = ""
    lat: float | None = None
    lng: float | None = None
    dist_km: float | None = None
    period: str = ""       # 행사 기간 (관광지·식당은 빈 문자열)
    use_time: str = ""
    kind: str = ""         # event | attraction | restaurant


class StopNearby(BaseModel):
    restaurants: list[NearbyCard] = []
    attractions: list[NearbyCard] = []


class CourseStop(BaseModel):
    # ── 기존 계약 (프론트 UI 동결 필드) ──
    name: str
    preview: str = ""
    description: str = ""
    duration: str = ""
    tip: str | None = None
    # ── 추가 계약 (좌표는 프론트 라우팅 입력, reason/activities 는 AI 선정 근거) ──
    lat: float | None = None
    lng: float | None = None
    reason: str = ""                # 왜 이 코스에 이 장소를 골랐는지
    activities: list[str] = []      # 이 장소에서 무엇을 할 수 있는지
    congestion: str | None = None   # 실시간 혼잡도 레벨
    nearby: StopNearby = Field(default_factory=StopNearby)
    # ── 스케줄 계약 (시간 범위 요청일 때만 채워짐 — 없으면 기존 응답과 동일) ──
    start_time: str | None = None   # "14:00"
    end_time: str | None = None     # "15:30"
    slot_type: Literal["place", "meal", "flex"] = "place"
    day: int | None = None          # 멀티데이 코스의 N일차 (하루 코스는 None)
    # 식사 슬롯 한정 — 앵커 장소 3km 이내 Visit Seoul 실데이터 식당 (코스 내 중복 없음)
    meal_options: list[NearbyCard] = []
    # 직전 스톱에서의 이동 추정 (시간표 코스 한정) — 1.5km 이내 walk, 초과 transit
    travel_min: int | None = None
    travel_mode: Literal["walk", "transit"] | None = None


class Course(BaseModel):
    title: str
    subtitle: str = ""
    description: str = ""
    stops: list[CourseStop]
    tags: list[str] = []
    scheduled: bool = False         # True 면 stops 순서·시간이 서버가 계산한 시간표
    days: int = 1                   # 여행 일수 — stops 의 day 필드와 함께 일자별 렌더링용
    day_areas: dict[int, str] = {}  # 일차 → 권역 (여행자 멀티데이 권역 분산일 때만)


class CourseResponse(BaseModel):
    course: Course
    source: str = "ai"
