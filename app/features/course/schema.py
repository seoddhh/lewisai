"""course(테마코스) 입출력.

장소 선정·"왜 이 장소인지"(reason/activities)·방문 순서는 AI 가 확정한다.
strangemap 프론트(courseRouting.ts)는 그 순서 위에서 실제 도로 경로(폴리라인)만 그린다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

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

# ── 식사 칩 ────────────────────────────────────────────────────────────────
# 끼니는 목적 칩과 별개다. 사용자가 고른 끼니만, 정해진 시각에 코스에 들어간다.
# 식당은 검색이 아니라 data/meal_cache 에서 가져온다.
Meal = Literal["아침", "점심", "저녁"]
MEALS = ("아침", "점심", "저녁")
# 끼니가 들어가는 시각. 그 시간에 실제로 문 연 식당이 넉넉한지 확인하고 정한 값이다.
MEAL_TIMES: dict[str, int] = {"아침": 10, "점심": 13, "저녁": 19}
MEAL_DURATION_MIN = 60

# 목적 칩 — 현지인·여행자가 같은 7개를 쓴다(보여주는 질문 문구만 다르다).
# 목적 하나가 네 가지 정보를 들고 있다:
#  - query : 후보를 검색할 때 덧붙이는 검색어
#  - rule  : 후보 중에서 어떤 장소를 고를지 AI 에게 주는 기준
#  - act   : 그 장소에서 뭘 할지 쓸 때의 기준. 동반자 조건과 합쳐져서,
#            같은 장소라도 목적에 따라 다른 행동이 나온다.
#            가게 이름 대신 행동의 종류로 써야 AI 가 없는 가게를 지어내지 않는다.
#  - slug  : 장소 데이터에 붙어 있는 태그 이름. 검색할 때 목적으로 걸러내는 데 쓴다.
PURPOSE_RULES: dict[str, dict] = {
    "자연·힐링": {"slug": "nature",
               "query": "조용한 산책 자연 공원 한강 숲 정원 휴식",
               "rule": "번잡한 상권보다 공원·한강·정원처럼 쉬어 갈 수 있는 곳을 우선하라.",
               "act": "숲길 산책·전망 쉼터에서 쉬기·잔디밭 휴식처럼 천천히 둘러보는 행동"},
    "문화·예술": {"slug": "culture",
                "query": "전시 공연 미술관 박물관 고궁 역사 전통",
                "rule": "전시·공연·박물관·고궁처럼 보고 읽을 콘텐츠가 있는 장소를 우선하라.",
                "act": "전시·상설관 관람·해설 듣기·전시 해설 자료 읽기처럼 콘텐츠를 살펴보는 행동"},
    "관광 명소": {"slug": "landmark",
                "query": "서울 대표 명소 랜드마크 전망 관광",
                "rule": "서울을 대표하는, 처음 온 사람이 놓치면 아쉬운 명소를 우선하라.",
                "act": "랜드마크 관람·전망 감상·대표 포토스폿 촬영처럼 명소를 눈에 담는 행동"},
    "체험·놀거리": {"slug": "activity",
                 "query": "체험 액티비티 클래스 만들기 놀거리 재미",
                 "rule": "구경만 하는 곳보다 직접 해볼 거리가 있는 활기찬 장소를 우선하라.",
                 "act": "원데이 클래스·만들기 체험·게임처럼 몸으로 직접 해보며 즐기는 행동"},
    "데이트": {"slug": "date",
              "query": "데이트 분위기 야경 감성",
              "rule": "둘이 걷기 좋고 오래 머물 만한 장소를 우선하라.",
              "act": "산책·야경 감상·사진 촬영·카페 휴식처럼 둘이 함께 하는 행동"},
    "핫플레이스": {"slug": "hotplace",
                "query": "핫플레이스 SNS 감성 트렌디 카페",
                "rule": "요즘 SNS에서 찾는 트렌디한 곳을 우선하라.",
                "act": "요즘 찾는 카페·팝업스토어 방문·포토존 촬영처럼 최근 화제인 곳을 둘러보는 행동"},
    "쇼핑": {"slug": "shopping",
            "query": "쇼핑 시장 상권 편집숍 백화점",
            "rule": "쇼핑 상권·시장을 우선하라.",
            "act": "상권·시장 둘러보기·편집숍 구경·기념품 고르기처럼 사고 구경하는 행동"},
}

# 예전에 쓰던 칩 이름들. 브라우저에 옛 값이 남아 있거나 사용자가 비슷한 말로 적어도
# 현행 칩으로 알아듣게 해준다.
PURPOSE_ALIASES: dict[str, str] = {
    "힐링": "자연·힐링", "자연 힐링": "자연·힐링",
    "문화생활": "문화·예술", "문화·예술·역사": "문화·예술",
    "관광": "관광 명소", "유명 관광지": "관광 명소",
    "놀거리": "체험·놀거리", "체험·액티비티": "체험·놀거리",
}
# 없어진 목적. 들어와도 에러 대신 조용히 무시한다(칩 하나 때문에 요청 전체가 깨지지 않게).
# "맛집 탐방"은 목적이 아니라 끼니 칩(meals)으로 옮겼다.
PURPOSE_RETIRED: frozenset[str] = frozenset({"맛집 탐방", "맛집탐방"})

# 동반자별로 "거기서 뭘 할지"를 정하는 기준. 장소를 고르는 데는 쓰지 않고,
# 행동·선정 이유·소개 문구를 쓸 때만 쓴다.
# 예: 친구와 잠실 → 야구 관람·보드게임 / 연인과 잠실 → 석촌호수 산책·팝업 구경.
COMPANION_RULES: dict[str, str] = {
    "혼자": "몰입·자유 — 사색·전시 관람·독립적으로 즐기는 활동. '함께/둘이' 표현은 쓰지 말 것.",
    "친구와": "함께 즐기는 활동·체험형 — 관람·게임·먹거리 탐방·같이 찍는 사진.",
    "연인과": "둘이 함께 — 산책·야경 감상·팝업/전시 구경·조용한 카페.",
    "배우자와": "편안한 일상 데이트 — 여유로운 식사·가벼운 산책·전시.",
    "아이와": "체험·안전·교육형 — 체험학습·넓고 개방된 공간·간식. 늦은 밤·주점류는 제외.",
    "부모님과": "여유로운 속도 — 고궁·정원·전통, 앉아서 쉴 곳이 있고 이동은 최소로.",
}
# 동반자별로 어떤 장소를 위로 올릴지 정하는 점수. 바로 위 COMPANION_RULES 가 "무엇을 할지"를
# 맡는다면, 이쪽은 "어떤 장소가 먼저 나올지"를 맡는다.
#
# 후보에서 빼는 게 아니라 점수만 더한다. 아예 걸러내면 후보가 너무 적어지기 때문이다.
# 점수는 0~1 사이 검색 점수에 더해지므로 ±0.15 안으로 둔다.
# 이보다 크면 거리 점수를 눌러버려서 동선이 엉킨다.
COMPANION_WEIGHTS: dict[str, dict] = {
    "혼자": {"energy": {"calm": 0.06}, "stay_max": None},
    "친구와": {"energy": {"lively": 0.08}, "stay_max": None},
    "연인과": {"energy": {"calm": 0.05}, "night_ok": 0.05, "stay_max": None},
    "배우자와": {"energy": {"calm": 0.05}, "stay_max": None},
    # 아이와: 날씨·화장실 때문에 실내를 선호하고, 90분 넘게 머무는 곳과 늦은 밤은 감점
    "아이와": {"indoor": 0.08, "stay_max": 90, "stay_penalty": 0.10, "night_ok": -0.06},
    # 부모님과: 조용하고 앉아 쉴 수 있는 곳을 위로, 늦은 밤은 감점
    "부모님과": {"energy": {"calm": 0.08}, "night_ok": -0.05, "stay_max": None},
}

# 목적별 가점. 위와 같은 방식으로, 같은 동네라도 목적에 따라 다른 장소가 먼저 나오게 한다.
PURPOSE_WEIGHTS: dict[str, dict] = {
    "자연·힐링": {"indoor": -0.08, "energy": {"calm": 0.08}},
    "문화·예술": {"indoor": 0.06, "energy": {"calm": 0.04}},
    "체험·놀거리": {"energy": {"lively": 0.08}},
    "핫플레이스": {"energy": {"lively": 0.08}, "night_ok": 0.04},
    "쇼핑": {"indoor": 0.05, "energy": {"lively": 0.04}},
    "관광 명소": {},
    "데이트": {"energy": {"calm": 0.05}, "night_ok": 0.06},
}

Purpose = Literal[
    "자연·힐링", "문화·예술", "관광 명소", "체험·놀거리",
    "데이트", "핫플레이스", "쇼핑",
]
PURPOSE_SLUGS: dict[str, str] = {p: r["slug"] for p, r in PURPOSE_RULES.items()}
# 검색할 때 태그로 걸러낼 수 있는 목적들. "데이트"만 뺀다 —
# 공원·고궁·전시관·쇼핑몰이 전부 데이트하기 좋은 곳이라 태그를 붙여봐야 후보가 안 좁혀진다.
# 대신 데이트 칩은 검색어(query)와 위 가점(PURPOSE_WEIGHTS)으로 반영한다.
FILTERABLE_PURPOSES: frozenset[str] = frozenset(PURPOSE_SLUGS) - {"데이트"}

# 칩 시간대 → 시간 범위 (시). 칩보다 구체적인 time_window 가 있으면 그쪽이 우선.
_TIME_SLOT_WINDOWS: dict[str, tuple[int, int]] = {"오전": (9, 12), "오후": (12, 18), "밤": (18, 23)}


class TimeWindow(BaseModel):
    """코스를 도는 시간대. 오후 2시~8시면 start=14, end=20.

    자정을 넘겨도 되게 end 는 24를 넘을 수 있다(28이면 새벽 4시).
    """

    start: int = Field(ge=0, le=24)
    end: int = Field(ge=1, le=28)

    def overlaps(self, op_start: int, op_end: int) -> bool:
        """가게 운영시간(op_start~op_end)이 이 시간대와 겹치는지 본다.

        24시간 영업이나 새벽까지 하는 곳(예: 5시~다음날 1시)도 제대로 판단한다.
        """
        if (op_start, op_end) == (0, 24):
            return True
        o_end = op_end if op_end > op_start else op_end + 24
        # 자정을 넘는 경우를 맞추려고, 운영시간을 하루 뒤로 밀어서도 한 번 더 겹치는지 본다
        for shift in (0, 24):
            if max(op_start + shift, self.start) < min(o_end + shift, self.end):
                return True
        return False

    def label(self) -> str:
        return f"{self.start}시~{self.end % 24}시" if self.end > 24 else f"{self.start}시~{self.end}시"


# 여행자에게는 시간대를 묻지 않으므로, 하루 전체(9~21시)를 기본으로 쓴다
_TOURIST_DAY_WINDOW = (9, 21)


class CourseChips(BaseModel):
    """사용자가 위저드에서 고른 칩들. 전부 선택 사항이라, 다 비어 있으면 자유입력만으로 만든다.

    동반자·목적·지역은 여러 개를 고를 수 있어서 목록으로 받는다.
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
    # 코스에 넣을 끼니. 고른 것만 정해진 시각에 들어가고, 비어 있으면 식사를 안 넣는다.
    meals: list[Meal] = Field(default_factory=list)
    # 시간대 칩보다 구체적인 시간. 시간 선택 칩이나 "오후 2시부터 8시까지" 같은 문장에서 채워진다.
    time_window: TimeWindow | None = None

    @field_validator("purposes", mode="before")
    @classmethod
    def _merge_legacy_purposes(cls, v):
        """옛 목적 이름을 현행 이름으로 바꾸고, 없어진 목적은 버린다.

        브라우저에 남은 옛 값이 들어와도 요청 전체가 깨지지 않게 하려는 것이다.
        바꾸고 나서 중복은 없앤다.
        """
        if not isinstance(v, list):
            return v
        merged = [PURPOSE_ALIASES.get(p, p) if isinstance(p, str) else p for p in v]
        kept = [p for p in merged if not (isinstance(p, str) and p in PURPOSE_RETIRED)]
        return list(dict.fromkeys(kept))

    @field_validator("meals", mode="before")
    @classmethod
    def _clean_meals(cls, v):
        """모르는 값은 버리고, 아침 → 점심 → 저녁 순으로 정렬한다."""
        if not isinstance(v, list):
            return v
        kept = [m for m in dict.fromkeys(v) if m in MEALS]
        return sorted(kept, key=lambda m: MEAL_TIMES[m])

    def resolved_window(self) -> TimeWindow | None:
        """최종적으로 쓸 시간 범위. 구체적인 시간 > 시간대 칩 > 여행자 기본값 순으로 고른다.

        고른 끼니가 범위 밖이면 범위를 늘린다. "오후(12~18시)"에 "저녁"을 골랐다면
        19시 식사가 범위 밖이라 사라져 버리는데, 저녁을 고른 사람의 뜻은
        "저녁까지 있겠다"이기 때문이다.

        범위가 아예 없으면 끼니만으로 만들지는 않는다. 시간표가 없는 코스라 넣을 자리가 없다.
        """
        w = self._base_window()
        if not w or not self.meals:
            return w
        start = min([w.start] + [MEAL_TIMES[m] for m in self.meals])
        # 식사에 1시간이 걸리니, 마지막 끼니 시각 + 1시간까지는 범위에 들어와야 한다
        end = max([w.end] + [MEAL_TIMES[m] + MEAL_DURATION_MIN // 60 for m in self.meals])
        return TimeWindow(start=start, end=min(end, 28))

    def _base_window(self) -> TimeWindow | None:
        if self.time_window:
            return self.time_window
        if self.time in _TIME_SLOT_WINDOWS:
            start, end = _TIME_SLOT_WINDOWS[self.time]
            return TimeWindow(start=start, end=end)
        if self.audience == "tourist":
            return TimeWindow(start=_TOURIST_DAY_WINDOW[0], end=_TOURIST_DAY_WINDOW[1])
        return None

    def meal_slots(self) -> list[tuple[str, int, int]]:
        """고른 끼니를 (이름, 시작 분, 끝나는 분) 형태로 만든다. 시간 범위를 벗어난 끼니는 뺀다.

        보통은 resolved_window() 가 범위를 늘려주니 다 들어오지만,
        시간을 직접 지정한 요청에서는 범위 밖 끼니가 생길 수 있다.
        """
        w = self.resolved_window()
        if not w or not self.meals:
            return []
        out = []
        for m in self.meals:
            s = MEAL_TIMES[m] * 60
            if w.start * 60 <= s and s + MEAL_DURATION_MIN <= w.end * 60:
                out.append((m, s, s + MEAL_DURATION_MIN))
        return out

    def stops_per_day(self) -> int:
        """하루에 몇 곳을 갈지. 일정 밀도 칩을 골랐으면 그쪽을 따른다(빼곡 5곳, 널널 3곳)."""
        if self.pace == "packed":
            return 5
        if self.pace == "relaxed":
            return 3
        return self.place_count

    def purpose_rule(self) -> dict:
        """고른 목적들의 조건을 하나로 합친다. 여러 개를 골랐으면 문장을 이어 붙인다."""
        rules = [PURPOSE_RULES[p] for p in self.purposes if p in PURPOSE_RULES]
        if not rules:
            return {}
        return {
            "query": " ".join(dict.fromkeys(r["query"] for r in rules)),
            "rule": " ".join(dict.fromkeys(r["rule"] for r in rules)),
            "act": " / ".join(dict.fromkeys(r["act"] for r in rules)),
        }

    def purpose_slugs(self) -> list[str]:
        """검색할 때 태그 필터로 쓸 목적 이름들.

        "데이트"는 장소에 태그를 안 붙였기 때문에 여기서 빼야 한다.
        넣으면 아무 장소도 안 걸려서 검색이 헛돈다.
        """
        return [PURPOSE_SLUGS[p] for p in self.purposes if p in FILTERABLE_PURPOSES]

    def rerank_weights(self) -> dict:
        """동반자 가점과 목적 가점을 하나로 합친다. 검색 결과 순서를 조정하는 데 쓴다.

        같은 항목이 겹치면 더한다. "아이와 + 문화·예술"이면 실내 가점이 두 번 쌓여서
        실내 문화시설이 확실히 위로 온다. 너무 커지지 않게 쓰는 쪽에서 ±0.2 로 자른다.
        """
        out: dict = {"indoor": 0.0, "night_ok": 0.0, "energy": {}, "stay_max": None,
                     "stay_penalty": 0.0}
        sources = [COMPANION_WEIGHTS.get(c, {}) for c in self.companions]
        sources += [PURPOSE_WEIGHTS.get(p, {}) for p in self.purposes]
        for w in sources:
            out["indoor"] += w.get("indoor", 0.0)
            out["night_ok"] += w.get("night_ok", 0.0)
            for energy, v in (w.get("energy") or {}).items():
                out["energy"][energy] = out["energy"].get(energy, 0.0) + v
            if cap := w.get("stay_max"):
                out["stay_max"] = min(out["stay_max"] or cap, cap)
                out["stay_penalty"] = max(out["stay_penalty"], w.get("stay_penalty", 0.0))
        return out

    def companion_rule(self) -> str:
        """고른 동반자들의 기준을 합친다. 행동·소개 문구를 쓸 때만 쓰고 검색에는 안 쓴다."""
        return " ".join(
            dict.fromkeys(
                COMPANION_RULES[c] for c in self.companions if c in COMPANION_RULES
            )
        )

    def summary(self, *, for_description: bool = False) -> str:
        """사용자가 고른 조건을 프롬프트에 넣을 한 줄로 만든다.

        for_description=True 는 코스 전체 소개글을 쓸 때만 쓴다. 소개글에 나오면
        어색한 값(시각, 식사 없음, 시민/여행자 구분)을 아예 빼고 준다.
        프롬프트로 "쓰지 마"라고 해도 계속 새어 나와서, 재료에서 빼는 쪽을 택했다.
        예를 들어 "서울 시민"을 주면 AI 가 이걸 동행자로 착각해 "서울 시민인 부모님과"
        같은 문장을 썼다.
        """
        window = None if for_description else self.resolved_window()
        parts = [
            "" if for_description else
            {"local": "서울 시민", "tourist": "서울 여행자"}.get(self.audience or ""),
            f"여행 기간: {self.days}일" if self.days > 1 else "",
            f"동반: {'·'.join(self.companions)}" if self.companions else "",
            f"시간대: {self.time}" if self.time else "",
            f"시간 범위: {window.label()}" if window else "",
            f"목적: {'·'.join(self.purposes)}" if self.purposes else "",
            (
                (f"식사: {'·'.join(self.meals)}" if self.meals else "")
                if for_description
                else (f"식사: {'·'.join(f'{m} {MEAL_TIMES[m]}시' for m in self.meals)}"
                      if self.meals else "식사: 없음")
            ),
            f"위치: {'·'.join(self.locations)}" if self.locations else "",
            f"혼잡도 선호: {self.congestion}" if self.congestion else "",
            {"packed": "일정: 빼곡하게", "relaxed": "일정: 여유롭게"}.get(self.pace or ""),
            f"하루 {self.stops_per_day()}곳",
        ]
        return ", ".join(p for p in parts if p)


class CourseRequest(BaseModel):
    note: str = Field("", description="자유서술 (예: 야경 보면서 데이트)")
    chips: CourseChips = Field(default_factory=CourseChips)
    seed: int | None = Field(
        None, description='"다시 만들기" 용. 같은 칩으로 다른 코스를 받으려면 매번 다른 값을 '
                          "보낸다. 생략하면 칩+note 로 시드가 정해져 결과가 결정적이다.")


class NearbyCard(BaseModel):
    """장소 주변에 붙는 정보 카드 하나. 식당·행사·관광지가 같은 모양을 쓴다."""

    title: str
    summary: str = ""
    address: str = ""
    lat: float | None = None
    lng: float | None = None
    dist_km: float | None = None
    period: str = ""       # 행사 기간. 식당·관광지는 비어 있다
    use_time: str = ""     # 운영시간
    kind: str = ""         # event | attraction | restaurant


class StopNearby(BaseModel):
    """한 장소 주변의 식당과 관광지 목록."""

    restaurants: list[NearbyCard] = []
    attractions: list[NearbyCard] = []


class CourseStop(BaseModel):
    """코스에 들어가는 장소 한 곳. 프론트 카드 하나에 그대로 대응된다."""

    name: str
    preview: str = ""
    description: str = ""
    duration: str = ""
    tip: str | None = None
    # 좌표는 프론트가 지도에 경로를 그릴 때 쓴다
    lat: float | None = None
    lng: float | None = None
    reason: str = ""                # 왜 이 장소를 골랐는지
    activities: list[str] = []      # 여기서 뭘 할 수 있는지
    congestion: str | None = None   # 방문 시각의 예상 혼잡도. 예: "18시 예상 혼잡도: 여유"
    nearby: StopNearby = Field(default_factory=StopNearby)

    # 아래는 시간표가 있는 코스에서만 채워진다
    start_time: str | None = None   # "14:00"
    end_time: str | None = None     # "15:30"
    slot_type: Literal["place", "meal", "flex"] = "place"
    day: int | None = None          # 며칠째인지. 하루 코스면 None
    meal_options: list[NearbyCard] = []  # 식사 슬롯일 때 고를 수 있는 주변 식당들
    # 직전 장소에서 여기까지 오는 데 걸리는 시간. 1.5km 안이면 걷고, 넘으면 대중교통으로 본다
    travel_min: int | None = None
    travel_mode: Literal["walk", "transit"] | None = None


class Course(BaseModel):
    """완성된 코스 하나. 프론트가 받는 최종 결과물이다."""

    title: str
    subtitle: str = ""
    description: str = ""
    stops: list[CourseStop]
    tags: list[str] = []
    scheduled: bool = False         # True 면 stops 의 순서와 시간이 서버가 짠 시간표라는 뜻
    days: int = 1                   # 며칠짜리 코스인지
    day_areas: dict[int, str] = {}  # 며칠째에 어느 동네인지. 여러 날을 동네별로 나눴을 때만 채워진다
    # 며칠째의 설명. 여러 날 코스는 전체 설명을 짧게 두고 날짜별로 나눠 쓴다
    # (프론트가 날짜 탭마다 그 날 설명만 보여준다). 하루 코스면 비어 있다.
    day_descriptions: dict[int, str] = {}


class CourseResponse(BaseModel):
    course: Course
    source: str = "ai"  # ai | mock — AI 가 만든 것인지 임시 데이터인지
