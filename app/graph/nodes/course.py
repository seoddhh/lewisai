"""course 파이프라인 — RAG 후보 → AI 장소 선정(+선정 이유) → 혼잡도 → 주변 정보 → 서사.

역할 분리:
 - AI: "어떤 장소를, 왜 골랐는지"와 "거기서 뭘 할 수 있는지(행동추천)".
   지금의 서울(오늘 날짜·시간대·실시간 혼잡도)에 맞춰 쓴다.
 - 주변 정보 3분할: 식당은 Visit Seoul 권역 캐시(meal_cache), 행사는 서울시 citydata
   실시간(events), 관광지는 임베딩 세트(seoul_places). nearby_node 는 식당·행사만 붙인다.
 - strangemap 프론트(courseRouting.ts): 방문 순서·지도 폴리라인·거리.
   → 폴리라인 경로는 프론트에서 작업 Ai 서빙은 해당 장소의 좌표값만 반환
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date

from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.core.geo import address_terms, chip_of, haversine_km, nearest_terms
from app.core.json_parse import parse_json_object, salvage_objects
from app.core.plan import allocate_stops, build_skeleton, describe
from app.core.scheduler import DWELL_CEIL, DWELL_FLOOR, duration_label
from app.core.llm import extract_text, get_llm
from app.features.course.schema import MEAL_DURATION_MIN, CourseChips
from app.graph.state import AgentState
from app.rag import retriever
from app.tools.congestion import get_forecast_level
from app.tools.events import get_events
from app.tools.meal_cache import meal_card, pool_for_stops
from app.tools.visitseoul import (
    MEAL_KINDS,
    VisitSeoulError,
    place_keyword,
    search_nearby,
)

logger = logging.getLogger("lewisai.course")

_NEARBY_RADIUS_KM = 1.5   # 스톱 하나에 붙일 주변 정보 반경
_NEARBY_PER_STOP = 2      # 스톱당 식당/관광 각 N개
_NEARBY_BUDGET = 14       # 코스 전체 상세 조회 예산 (Visit Seoul rate limit ~1.4 req/s)

# ── 지리 클러스터링 (코스는 지리적 근접성 우선) ──────────────────────────
_FETCH_K = 40                    # 벡터스토어에서 넉넉히 당겨오는 후보 수 (필터 단계마다)
_MIN_POOL = 12                   # 이 아래면 필터를 한 단계 풀어 다시 당긴다 (_search_pool)
_RADIUS_STEPS = (5.0, 7.0, 10.0)  # 앵커 기준 반경 하드필터 — 부족하면 순차 완화
_GEO_ALPHA = 0.3                 # 최종 점수 = α·의미유사도 + (1−α)·근접도 (코스는 근접 우선)
# 거리 min-max 정규화 스케일 하한 — 후보가 좁게 모여 있을 때(수백 m) 미세한 거리 차이가
# 0~1 로 뻥튀기되어 다른 신호(유사도)를 눌러버리는 것을 막는다. 이 안은 다 도보권.
_DIST_SCALE_FLOOR_KM = 2.0
# 식사 추천 반경 — 앵커 장소 기준. 걸어갈 만한 1.5km 를 우선 보고, 3곳을 못 채울 때만
# 3km 로 넓힌다. Visit Seoul 음식 데이터가 자치구별로 크게 편중돼 있어서다
# (표본 실측: 종로 65 · 용산 51 · 마포 49 vs 관악 3 · 성북 5 — 관악·사당은 1.5km 로 못 채움).
MEAL_RADIUS_NEAR_KM = 1.5
MEAL_RADIUS_KM = 3.0

# 여행자 멀티데이 + 위치 "상관없음" → 날마다 다른 권역을 배정한다 (실제 여행 패턴).
# 1일차는 여행 목적에 맞는 권역으로 시작하고, 나머지는 회전 순서를 따른다.
_AREA_ROTATION = ("종로·중구", "홍대·마포", "성수·건대", "강남·서초", "용산·이태원", "잠실·송파")
_PURPOSE_FIRST_AREA = {
    "핫플레이스": "성수·건대", "체험·놀거리": "홍대·마포",
    "쇼핑": "강남·서초", "자연·힐링": "여의도·영등포",
    "맛집 탐방": "종로·중구", "문화·예술": "종로·중구", "관광 명소": "종로·중구",
}


def _day_areas(chips: CourseChips) -> dict[int, str] | None:
    """일차 → 권역 배정.

    동네를 여러 개(여행일수만큼) 고르면 그 순서대로 하루씩 배정한다.
    동네를 하나만 골랐으면 그 동네를 존중해 분산하지 않는다(None).
    "상관없음"(또는 미선택)이면 목적에 맞춰 자동으로 날마다 다른 권역을 돈다.
    """
    if chips.audience != "tourist" or chips.days <= 1:
        return None
    picked = [loc for loc in chips.locations if chip_of(loc)]
    if len(picked) >= 2:
        return {d: picked[(d - 1) % len(picked)] for d in range(1, chips.days + 1)}
    if picked:
        return None
    first = _PURPOSE_FIRST_AREA.get(chips.purposes[0] if chips.purposes else "", _AREA_ROTATION[0])
    order = [first, *[a for a in _AREA_ROTATION if a != first]]
    return {d: order[(d - 1) % len(order)] for d in range(1, chips.days + 1)}

_SELECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 '서울로' 코스 큐레이터다. 후보 목록에서만 장소를 골라 JSON 으로만 답하라.\n"
            'form: {{"days":[{{"day":1,"stops":[{{"name":"후보에 있는 장소명",'
            '"reason":"이 사람에게 왜 이 장소인지 한 문장",'
            '"activities":["거기서 할 수 있는 일 2~3개"],'
            '"dwell_min":그 활동에 걸리는 체류시간(분, 정수)}}]}}]}}\n'
            "- 총 {days}일, 하루 정확히 {count}곳. 후보에 없는 이름은 절대 넣지 말고, "
            "같은 장소를 여러 날에 중복해 넣지 말 것.\n"
            "{day_note}"
            "{skeleton_rule}"
            "- **dwell_min 은 activities 에 적은 일을 실제로 하는 데 걸리는 시간**을 분 단위 "
            "정수로 쓴다(30~180). 같은 장소라도 '전시 두 개 관람 + 해설 투어'면 길고 "
            "'포토스폿에서 사진'이면 짧다. 서버가 이 값을 구간 예산에 맞춰 비율대로 조정하므로 "
            "합이 딱 맞지 않아도 되고, 상대적인 길이만 정확하면 된다.\n"
            "- 같은 날의 장소는 걸어서 이어질 만큼 가까운 곳끼리 묶을 것 (후보의 거리 참고).\n"
            "- **스톱 개수({count}곳)를 먼저 정확히 채운 뒤, 그 안에서 가능한 한 대분류(후보 줄 첫 항목: "
            "자연·역사·문화·쇼핑·명소·체험)를 섞어라.** 같은 대분류만 연달아 3곳 넣지 말고(예: 백화점·"
            "면세점·쇼핑몰만 이어 붙이기), 후보에 다른 대분류가 있으면 그쪽을 우선 고른다. "
            "**개수를 줄여서 다양성을 맞추지 말 것 — 개수가 먼저다.**\n"
            "- 요청에 시간 범위가 있으면 그 시간에 문 닫는 장소는 고르지 말 것 (후보의 운영시간 참고).\n"
            "{purpose_rule}"
            "{companion_rule}"
            "{purpose_act}"
            "- reason 은 '왜 이 동반·목적의 사용자에게, 이 코스 흐름에 이 장소를 골랐는지'(선정 근거)만 "
            "한 문장으로 쓴다. 동반·목적·계절·코스 내 역할(예: 3일차 역사 테마의 마무리)에 근거하되, "
            "**activities 에 담을 '할 일'을 여기서 서술하지 말 것(추천 이유 ≠ 할 일).** "
            "운영시간 나열, '~하기 좋습니다' 같은 어디에나 붙는 일반론도 금지. "
            "감상적 수식 없이 근거를 짚어 쓰고, 문체는 항상 해요체(예: '~라서 골랐어요')로 통일한다.\n"
            "- activities 는 그 장소에서 실제로 할 수 있는 행동인데, **누구와(동반) 가는지 × 무엇을 하러(목적) "
            "가는지에 따라 같은 장소라도 하는 일이 달라진다.** 위의 동반 렌즈와 목적별 행동 결을 곱해, "
            "그 장소의 소개 글에 실제로 등장하는 특징(시설·풍경·볼거리)에 걸어 행동을 2~3개 제안하라.\n"
            "- **행동은 서로 다른 유형으로 벌릴 것** — 관람/체험/먹거리/산책·사진/쇼핑 등 한 장소 안에서 같은 "
            "유형만 반복하지 말고, 코스 전체에서도 스톱마다 행동이 겹치지 않게 다양화한다.\n"
            "- 가게 상호를 지어내지 말 것(환각 금지). 그 장소·권역에서 실제로 할 만한 '행동 유형'으로 쓴다. "
            "예: 친구와+놀거리 잠실 → ['야구 경기 관람','보드게임 카페에서 한 판'] / "
            "연인과+데이트 잠실 → ['석촌호수 산책','팝업스토어 구경'] / "
            "부모님과+문화·예술 고궁 → ['천천히 정원 산책','고궁 해설 투어'].\n"
            "- 하루 안에서는 자연스러운 방문 흐름 순서로 나열 (시간표는 서버가 다시 계산한다).",
        ),
        (
            "human",
            '오늘: {today}\n요청: "{note}"\n선택 조건: {chips}\n\n[후보]\n{candidates}',
        ),
    ]
)

# compose 는 하나의 프롬프트였는데 **전역 서사**와 **스톱 카드**로 쪼갰다.
#
# 이유는 레이턴시다. 실측(solar-open2)에서 compose 가 코스 생성 총 시간의 66~69% 를
# 먹었고(현지인 1일 18.5s / 여행자 2일 46.2s), 그 시간은 프롬프트 크기가 아니라
# **출력 토큰 수에 비례**한다. 한 콜이 전역 서사 + 스톱 N개 카드를 다 쓰니 출력이 길다.
#
# 쪼갠 축이 "일차"가 아니라 "스톱"인 것도 의도한 것이다. 일차 축은 days==1 에 효과가
# 0인데 1일 코스가 이미 27.9s 다. 스톱 축은 하루 코스에서도 병렬이 걸린다.
#
# 공통 문체 규칙은 두 프롬프트 모두에 들어가야 한다 — 콜이 나뉘어도 결과물은 한 사람이
# 쓴 것처럼 읽혀야 하기 때문이다.
_STYLE_RULES = (
    "- **여행 플래너의 안내문처럼 쓴다** — 무엇을 보고 어디로 이동하고 얼마나 걸리는지 같은 실용 정보가 먼저다. "
    "분위기 묘사는 곁들이되 문단당 한 문장을 넘기지 말고, **'설렘·낭만·감성 가득·마법 같은·특별한 추억' 처럼 "
    "장소를 안 겪어도 쓸 수 있는 광고성 수식과 감탄은 쓰지 말 것.** 형용사보다 구체적인 사실로 쓴다.\n"
    "- **말투(문체)는 동반·목적과 무관하게 항상 같게** — 모든 문장은 담백한 해요체(예: '~해요', '~이에요')로 "
    "통일한다. 렌즈는 내용만 바꾸고 존댓말/반말 같은 문체를 바꾸는 게 아니다.\n"
)

_COMPOSE_GLOBAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 '서울로' 코스 플래너다. 아래 확정된 장소들로 여행 일정의 **제목과 전체 소개**를 써서 "
            "JSON 으로만 답하라. 개별 장소 소개와 일차별 요약은 다른 곳에서 쓰므로 여기서는 쓰지 않는다.\n"
            'form: {{"title":"...","subtitle":"...","description":"...","tags":["..."]}}\n'
            "- description(코스 전체)은 그날 동선을 순서대로 요약한다 — 어디서 시작해 어디로 옮겨 가며 무엇을 "
            "하는지, 시간대 변화(오전→오후→저녁)와 이동 흐름이 드러나게. 장소를 하나하나 재서술하지는 말 것"
            "(구체는 각 스톱 카드가 담는다).\n"
            "{day_desc_rule}"
            "- **누구와(동반) × 무엇하러(목적)에 따라 title·subtitle·description 의 강조점과 소재가 달라져야 한다** — "
            "아래 렌즈에 맞춰, 같은 장소들이라도 그 조합에 맞는 내용으로 쓴다(다른 조합엔 안 맞게). "
            "예: 연인과+데이트 → 둘이 걷기 좋은 구간·야경 시간대, 친구와+핫플 → 함께 즐길 거리·요즘 찾는 곳, "
            "부모님과+힐링 → 앉아 쉴 곳·짧은 이동 거리.\n"
            + _STYLE_RULES +
            "  title·subtitle 은 명사형 가능.\n"
            "{companion_lens}"
            "{purpose_lens}"
            "- 방문 시각(예: 14:00~15:30)이 주어진 장소는 그 시간대에 맞게 쓰되 "
            "시각은 이미 확정된 값이니 바꾸거나 언급을 지어내지 말 것.",
        ),
        ("human", '요청: "{note}"\n선택 조건: {chips}\n\n[확정된 장소]\n{stops}'),
    ]
)

_COMPOSE_DAY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 '서울로' 코스 플래너다. 아래는 여러 날 일정 중 **{day}일차 하루**다. "
            "그날의 동선 요약을 써서 JSON 으로만 답하라.\n"
            'form: {{"day_description":"..."}}\n'
            "- 그날 동선을 순서대로 2~3문장으로 요약한다 — 어디서 시작해 어디로 옮겨 가며 무엇을 하는지, "
            "시간대 변화(오전→오후→저녁)와 이동 흐름이 드러나게.\n"
            "- **주어진 {day}일차 장소만 근거로 쓴다.** 다른 날 얘기를 섞지 말고, 장소를 하나하나 "
            "재서술하지도 말 것(구체는 각 스톱 카드가 담는다).\n"
            + _STYLE_RULES +
            "{companion_lens}"
            "{purpose_lens}"
            "- 방문 시각(예: 14:00~15:30)이 주어진 장소는 그 시간대에 맞게 쓰되 "
            "시각은 이미 확정된 값이니 바꾸거나 언급을 지어내지 말 것.",
        ),
        ("human", '요청: "{note}"\n선택 조건: {chips}\n\n[{day}일차 장소]\n{stops}'),
    ]
)

_COMPOSE_STOPS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 '서울로' 코스 플래너다. 아래 장소들의 **카드 문구**를 써서 JSON 으로만 답하라.\n"
            'form: {{"stops":[{{"name":"확정 장소명","preview":"한 줄",'
            '"description":"어떤 곳인지 2~3문장"}}]}}\n'
            "- stops 의 name 과 개수는 주어진 목록과 정확히 일치시킬 것 (순서 변경·추가·누락 금지). "
            "**모든 장소에 preview·description 을 빠짐없이** 채운다.\n"
            "- **각 장소의 description 은 ① 그곳이 어떤 곳인지(공간·규모·볼거리)를 사실 위주로 짚은 뒤 ② 주어진 "
            "'여기서 할 일'(activities)을 이어 2~3문장에 담는다** — 장소 소개와 할 일이 한 흐름으로 읽히게. "
            "activities 에 없는 행동을 지어내지 말 것.\n"
            "- description 에 '선정 이유'(reason)는 되풀이하지 말 것 — 추천 이유는 카드에 따로 보인다. "
            "운영시간 나열도 금지(시간표·팁 담당).\n"
            + _STYLE_RULES +
            "{companion_lens}"
            "{purpose_lens}"
            "- 실시간 혼잡도가 주어진 장소는 그 상황을 description 에 자연스럽게 반영.\n"
            "- 방문 시각(예: 14:00~15:30)이 주어진 장소는 그 시간대에 맞게 쓰되 "
            "시각은 이미 확정된 값이니 바꾸거나 언급을 지어내지 말 것.\n"
            "- 장소에 일차(예: 2일차)가 붙어 있으면 일차 흐름이 이어지게 description 을 쓸 것.\n"
            "- description 은 선정 이유에 담긴 데이터(운영시간·특징)와 어긋나지 않게 쓸 것.\n"
            "- 이 목록은 전체 코스의 일부일 수 있다. 코스 전체를 요약하려 들지 말고 "
            "**주어진 장소만** 쓴다.",
        ),
        ("human", '요청: "{note}"\n선택 조건: {chips}\n\n[이번에 쓸 장소]\n{stops}'),
    ]
)

# 한 콜이 맡을 스톱 수. 콜당 출력을 줄이는 게 목적이므로 작을수록 빠르지만, 콜 수만큼
# TTFT(~1s)와 동시 요청 속도 저하가 붙는다. 3곳이면 하루 코스(3~5곳)가 2콜로 갈린다.
_STOPS_PER_CALL = 3


def _chips(state: AgentState) -> CourseChips:
    raw = state.get("req", {}).get("chips") or {}
    try:
        return CourseChips(**raw)
    except Exception:  # noqa: BLE001 — 프론트 칩 값이 어긋나도 코스는 만들어야 한다
        logger.warning("칩 파싱 실패, 기본값 사용: %s", raw)
        return CourseChips()


def _cand(doc) -> dict:
    m = doc.metadata
    return {
        # 이름은 반드시 정규화한다 — 원천 데이터에 앞뒤 공백(일반/탭/전각 U+3000)이 섞여 있고,
        # LLM 은 당연히 공백 없는 이름으로 답해서 select 의 화이트리스트 매칭이 조용히 실패한다
        # (실측: 후보 4.8%가 오염 → 고른 장소가 통째로 유실). 색인 쪽도 고쳤지만 여기서도 막는다.
        "name": (m.get("display_name") or "").strip(),
        "lat": m.get("lat"),
        "lng": m.get("lng"),
        "category": m.get("category", ""),
        # 하루 안 대분류 다양성 규칙(_SELECT_PROMPT)이 보고 판단할 축 — 후보 줄에 실어 보낸다
        "coarse": m.get("coarse_category", ""),
        "area_name": m.get("area_name", ""),
        "op_start": int(m.get("op_start", 0)),
        "op_end": int(m.get("op_end", 24)),
        # (0,24)가 "상시개방"인지 "운영시간 미확인"인지 구분한다. 미확인이면 시간창
        # 필터를 무조건 통과하므로, 아는 시간이 안 맞는 곳만 걸러내려면 이 값이 필요하다.
        "hours_known": bool(m.get("hours_known", False)),
        # 좌표가 같은(=사실상 같은 자리) 장소 묶음. 한 코스에 하나만 넣기 위한 키.
        "same_place_group": m.get("same_place_group", ""),
        "is_filming": bool(m.get("is_filming", False)),
        "content_title": m.get("content_title", ""),
        "text": doc.page_content,
    }


def _hours_label(c: dict) -> str:
    if (c["op_start"], c["op_end"]) == (0, 24):
        return "상시개방" if c.get("hours_known") else "운영시간 미확인"
    return f"운영 {c['op_start']}시~{c['op_end']}시"


def _closed_in_window(c: dict, window) -> bool:
    """이 후보가 **아는 운영시간 기준으로** 요청 시간대에 문을 닫는가.

    hours_known=False 는 (0,24)로 채워져 있을 뿐 "상시개방"이 아니므로 판정하지
    않는다(통과). 아는 시간이 안 맞는 곳만 하드 제외한다 — 예전에는 문 여는 후보가
    모자라면 닫힌 곳을 꼬리에 붙여서, 밤 코스에 18시에 닫는 박물관이 섞였다.
    """
    if not window or not c.get("hours_known"):
        return False
    return not window.overlaps(c["op_start"], c["op_end"])


def _dedupe_same_place(cands: list[dict]) -> list[dict]:
    """좌표가 같은 장소 묶음에서 상위 1곳만 남긴다.

    광화문광장 ⟷ 해치마당처럼 같은 자리인데 이름이 달라 화이트리스트로는 못 막는다.
    후보 단계에서 걸러 LLM 이 아예 보지 못하게 한다 (선정 후 제거보다 낭비가 적다).
    """
    out, seen = [], set()
    for c in cands:
        g = c.get("same_place_group")
        if g:
            if g in seen:
                continue
            seen.add(g)
        out.append(c)
    return out


def _geo_rerank(
    pool: list[tuple[dict, float]], anchor: tuple[float, float], want: int,
) -> list[dict]:
    """앵커 기준 반경 하드필터(부족하면 완화) + 거리 우선 블렌드 재정렬.

    pool: (후보, 의미거리) — 의미거리는 작을수록 유사(Chroma score).
    반환: 블렌드 상위 후보 dict[] (거리는 downstream 을 위해 dist_km 로 부착).
    """
    with_dist = [
        (c, sim, haversine_km(anchor[0], anchor[1], c["lat"], c["lng"]))
        for c, sim in pool
    ]
    # 반경 5→7→10km 로 완화하며 want 개 이상 확보. 그래도 없으면 반경 무시(폴백).
    near: list[tuple[dict, float, float]] = []
    for radius in _RADIUS_STEPS:
        near = [row for row in with_dist if row[2] <= radius]
        if len(near) >= want:
            break
    if not near:
        near = with_dist

    # 의미거리·지리거리를 각각 min-max 정규화 후 블렌드 (둘 다 작을수록 좋음 → 뒤집어 1=최상)
    sims = [s for _, s, _ in near]
    dists = [d for _, _, d in near]
    s_lo, s_hi = min(sims), max(sims)
    d_lo = min(dists)
    d_hi = max(max(dists), d_lo + _DIST_SCALE_FLOOR_KM)

    def _norm(v: float, lo: float, hi: float) -> float:
        return 1.0 if hi == lo else (hi - v) / (hi - lo)

    w_sim, w_dist = _GEO_ALPHA, 1 - _GEO_ALPHA
    scored = [
        (dict(c, dist_km=round(d, 2)),
         w_sim * _norm(s, s_lo, s_hi) + w_dist * _norm(d, d_lo, d_hi))
        for c, s, d in near
    ]
    scored.sort(key=lambda row: row[1], reverse=True)
    return [c for c, _ in scored]


def _search_pool(query: str, *, area: str | None, slugs: list[str],
                 k: int = _FETCH_K, window=None) -> list[tuple[dict, float]]:
    """후보 풀 조회 — 권역·목적 메타필터를 걸고, 얇으면 단계적으로 푼다.

    전역 상위 K 만 뽑던 이전 방식은 데이터가 종로·용산에 쏠려 있어 그 두 권역이 K 를
    다 먹었다(실측: 목적×권역 108조합 중 31개가 반경 5km 내 후보 4곳 미만 → 반경
    완화 폴백이 돌아 강남 요청에 종로 장소가 딸려왔다). 권역을 검색 단계에서 걸어
    "그 동네 안에서 목적에 맞는 곳"부터 채운다.

    완화 순서: 권역+목적 → 권역만 → 목적만 → 무필터. 각 단계는 _MIN_POOL 을 못 채울
    때만 다음으로 내려가고, 앞 단계 결과는 순서를 지켜 유지한다(권역 우선).

    window 가 주어지면 **아는 운영시간이 그 시간대와 안 맞는 곳은 하드 제외**한다.
    _MIN_POOL 판정 전에 걸러야 "문 닫은 곳으로 12개를 채우고 만족" 하는 일이 없다.
    """
    base: dict = {"doc_type": "place"}
    if area:
        base["area"] = area
    # 선택 목적 중 하나라도 붙은 장소 (pt_<slug> 불리언 메타는 ingest 가 심는다)
    any_purpose = ([{f"pt_{s}": {"$eq": True}} for s in slugs] if slugs else [])
    steps: list[dict] = []
    if area and any_purpose:
        steps.append({**base, "$or": any_purpose} if len(any_purpose) > 1
                     else {**base, f"pt_{slugs[0]}": True})
    if area:
        steps.append(base)
    if any_purpose:
        steps.append({"doc_type": "place", "$or": any_purpose} if len(any_purpose) > 1
                     else {"doc_type": "place", f"pt_{slugs[0]}": True})
    steps.append({"doc_type": "place"})

    pool: list[tuple[dict, float]] = []
    seen: set[str] = set()
    for filters in steps:
        for doc, sim in retriever.search_with_score(query, k=k, filters=filters):
            c = _cand(doc)
            if c["lat"] is None or c["lng"] is None or c["name"] in seen:
                continue
            if _closed_in_window(c, window):     # 아는 시간이 안 맞으면 하드 제외
                continue
            seen.add(c["name"])
            pool.append((c, sim))
        if len(pool) >= _MIN_POOL:
            break
    return pool


async def retrieve_node(state: AgentState) -> dict:
    """칩+자연어 → RAG 후보. 코스는 지리 근접이 우선이라 앵커 반경 클러스터링 후 재랭킹.

    앵커: 위치 칩이 있으면 칩 중심좌표, 없으면 의미 유사도 1위 후보의 좌표
    (자연어 "종로에서…"도 상위 결과가 종로권이라 지오코딩 없이 클러스터가 잡힌다).
    검색 자체도 그 권역·목적으로 필터해 동네마다 다른 후보가 잡히게 한다.
    """
    req = state.get("req", {})
    chips = _chips(state)
    rule = chips.purpose_rule()
    slugs = chips.purpose_slugs()

    # 목적 칩은 라벨 그대로가 아니라 task 조건(PURPOSE_RULES)의 검색 확장어로 검색한다
    query = " ".join(
        p for p in [req.get("note", ""), rule.get("query", " ".join(chips.purposes)),
                    " ".join(chips.companions), " ".join(chips.locations), chips.time or ""] if p
    ) or "서울 코스"

    window = chips.resolved_window()
    day_areas = _day_areas(chips)
    if day_areas:
        # 일차별로 그 권역의 후보를 따로 검색해 재랭킹 → day_hint 를 붙인 일차별 후보 그룹.
        # 앞 일차가 쓴 장소는 다음 일차에서 제외한다.
        per = chips.stops_per_day() + 2
        taken: set[str] = set()
        cands = []
        for d, area in day_areas.items():
            area_chip = chip_of(area)
            pool_d = [(c, s) for c, s in _search_pool(query, area=area, slugs=slugs,
                                                      window=window)
                      if c["name"] not in taken]
            if not pool_d:
                continue
            picks = _dedupe_same_place(
                _geo_rerank(pool_d, (area_chip.lat, area_chip.lng), per))[:per]
            taken.update(c["name"] for c in picks)
            cands.extend({**c, "day_hint": d} for c in picks)
        return {"candidates": cands, "chips": chips.model_dump(), "day_areas": day_areas}

    area = next((loc for loc in chips.locations if chip_of(loc)), None)
    pool = _search_pool(query, area=area, slugs=slugs, window=window)
    if not pool:
        return {"candidates": [], "chips": chips.model_dump()}

    chip = chip_of(area) if area else None
    anchor = (chip.lat, chip.lng) if chip else (pool[0][0]["lat"], pool[0][0]["lng"])

    want = chips.days * chips.stops_per_day() + 6
    cands = _dedupe_same_place(_geo_rerank(pool, anchor, want))

    return {"candidates": cands[:want], "chips": chips.model_dump()}


async def _llm_call(tag: str, chain, prompt_vars: dict) -> str:
    """LLM 1콜 → 텍스트. 소요시간·출력량을 로깅한다.

    select·compose 는 콜을 쪼개 동시에 던지는 구조라, 노드 단위 시간만으로는 어느 콜이
    벽시계를 잡고 있는지 안 보인다. 청크 크기를 조정하려면 이 로그가 필요하다.
    """
    t0 = time.perf_counter()
    text = extract_text((await chain.ainvoke(prompt_vars)).content)
    logger.info("%s %.1fs · 출력 %d자", tag, time.perf_counter() - t0, len(text))
    return text


async def _select_days(prompt_vars: dict, by_name: dict[str, dict]) -> list[dict]:
    """select LLM 1콜 → day_rows(`[{"day":n,"stops":[...]}]`). 예외를 밖으로 내지 않는다.

    토큰 한도로 JSON 이 잘려도 완결된 stop 들은 건져 이유·행동을 살린다(전멸 방지).
    여러 콜을 동시에 던지므로 한 콜의 실패가 다른 콜을 못 죽이게 여기서 흡수한다.
    """
    raw = ""
    try:
        raw = await _llm_call("select 콜", _SELECT_PROMPT | get_llm(), prompt_vars)
        data = parse_json_object(raw)
        # 구형 응답({"stops":[...]})도 1일차로 수용
        return data.get("days") or [{"day": 1, "stops": data.get("stops", [])}]
    except Exception as err:  # noqa: BLE001
        salvaged = [o for o in salvage_objects(raw) if o.get("name")]
        logger.warning("select LLM JSON 파싱 실패 — 부분 복구 %d곳: %s", len(salvaged), err)
        # 후보의 day_hint 로 일차를 되살려 그룹화 — 멀티데이에서 2·3일차 복구분이 per_day 컷에 안 잘리게.
        by_day: dict[int, list] = {}
        for o in salvaged:
            cand = by_name.get(o.get("name"))
            d = (cand.get("day_hint") if cand else None) or 1
            by_day.setdefault(d, []).append(o)
        return [{"day": d, "stops": s} for d, s in sorted(by_day.items())]


async def select_places_node(state: AgentState) -> dict:
    """AI 담당 구간 — 장소 선정(일자별) + 데이터에 근거한 선정 이유 + 할 수 있는 일.

    일차별 권역 분산 코스는 일차마다 콜을 나눠 동시에 던진다 (아래 주석 참고).
    """
    req = state.get("req", {})
    chips = _chips(state)
    cands = state.get("candidates", [])
    by_name = {c["name"]: c for c in cands}

    # 소개 글을 넉넉히 실어 reason·activities 가 실제 데이터(RAG 본문)를 근거로 쓰게 한다.
    # 장소가 지역구 단위라 본문에 "그 권역에 뭐가 있는지"가 담겨 있어야 행동 추천이 구체화된다.
    def _cand_line(c: dict) -> str:
        # K-콘텐츠는 카테고리로 나누지 않고 "이런 곳이다" 정보로만 넘긴다
        kc = (f" · K-콘텐츠 촬영지「{c['content_title']}」"
              if c.get("is_filming") and c.get("content_title") else "")
        return (
            f"- {c['name']} ({c.get('coarse') or '기타'} · {c['category']} · {_hours_label(c)}"
            f"{' · 앵커에서 ' + str(c['dist_km']) + 'km' if c.get('dist_km') is not None else ''}"
            f"{kc}): {c['text'][:280]}"
        )

    day_areas = state.get("day_areas") or {}
    pr = chips.purpose_rule()
    rule = pr.get("rule", "")
    act = pr.get("act", "")
    comp_rule = chips.companion_rule()
    per_day = chips.stops_per_day()

    # 시간 골격 — 하루가 어떻게 나뉘고 각 구간에 몇 곳이 들어가는지 LLM 에 알려준다.
    # 이게 있어야 "점심 13시 직후 첫 장소" 라는 맥락에 맞는 activities 가 나오고,
    # 구간 길이에 맞는 dwell_min 을 제안할 수 있다.
    skeleton = state.get("skeleton") or build_skeleton(chips)
    quota, _left = allocate_stops(skeleton, per_day)
    sk_text = describe(skeleton, quota)
    skeleton_rule = (
        f"- 하루 시간 골격은 이렇다 (서버가 확정했다):\n{sk_text}\n"
        "  각 구간에 배정된 곳 수를 지켜 고르고, 구간 길이에 맞는 dwell_min 을 제안하라. "
        "식사 시각 직후의 장소는 '방금 밥을 먹고 왔다'는 맥락에 맞게 activities 를 쓴다.\n"
        if sk_text else ""
    )

    base = {
        "today": date.today().isoformat(),
        "note": req.get("note", ""),
        "chips": chips.summary(),
        "count": per_day,
        "skeleton_rule": skeleton_rule,
        "purpose_rule": f"- 목적 조건: {rule}\n" if rule else "",
        "companion_rule": f"- 동반 조건(행동 렌즈): {comp_rule}\n" if comp_rule else "",
        "purpose_act": f"- 목적별 행동 결(행동 렌즈): {act}\n" if act else "",
    }

    # 일차별 권역 분산(day_areas)이면 후보가 day_hint 로 이미 서로소다 — 일차마다 따로
    # 고르게 해도 중복이 날 수 없으므로 콜을 나눠 **동시에** 던진다. 콜당 출력이 1/N 로
    # 줄어 벽시계가 짧아지고, max_tokens 절단으로 select 결과가 통째로 날아가던 문제도
    # 같이 사라진다 (실측: 여행자 2일이 4,000토큰에 잘려 mock 폴백으로 갔다).
    day_groups: dict[int, list[dict]] = {}
    if day_areas and chips.days > 1:
        for c in cands:
            day_groups.setdefault(c.get("day_hint") or 1, []).append(c)

    if len(day_groups) > 1:
        days_order = sorted(day_groups)
        results = await asyncio.gather(*(
            _select_days(
                {
                    **base,
                    "candidates": "\n".join(_cand_line(c) for c in day_groups[d]),
                    "days": 1,
                    "day_note": f"- 이 목록은 {chips.days}일 코스의 **{d}일차"
                                f"({day_areas.get(d, '')})** 후보다. {d}일차 하루만 고르면 된다"
                                " (다른 날은 따로 정해진다).\n",
                },
                by_name,
            )
            for d in days_order
        ))
        # 콜 하나가 곧 하루다 — 일차를 여기서 확정해 붙인다 (LLM 의 day 값은 항상 1)
        day_rows = [
            {"day": d, "stops": [s for r in rows for s in (r.get("stops") or [])]}
            for d, rows in zip(days_order, results)
        ]
    else:
        day_rows = await _select_days(
            {**base, "candidates": "\n".join(_cand_line(c) for c in cands),
             "days": chips.days, "day_note": ""},
            by_name,
        )

    picked: list[dict] = []
    seen: set[str] = set()
    for i, day_row in enumerate(day_rows[: chips.days]):
        day_no = int(day_row.get("day") or i + 1)
        for row in (day_row.get("stops") or [])[:per_day]:
            name = row.get("name", "")
            if name in by_name and name not in seen:  # 화이트리스트 강제 (환각 차단)
                seen.add(name)
                cand = by_name[name]
                # 일차별 권역 분산이면 LLM 이 섞어 담아도 day_hint 가 일차를 확정한다
                day = cand.get("day_hint") or day_no
                picked.append(
                    {
                        **cand,
                        "day": day if chips.days > 1 else None,
                        "reason": row.get("reason", ""),
                        "activities": [a for a in row.get("activities", []) if a][:3],
                        # 희망 체류시간 — 서버가 구간 예산에 맞춰 다시 조정한다.
                        # 값이 없거나 이상하면 None → scheduler 가 카테고리 매핑으로 폴백.
                        "dwell_min": _clamp_dwell(row.get("dwell_min")),
                    }
                )

    if len(picked) < 2:  # 폴백: 후보 상위를 일자별로 나눠 담는다 (이유 없이)
        fallback = [
            {**c, "day": (c.get("day_hint") or i // per_day + 1) if chips.days > 1 else None,
             "reason": "", "activities": [], "dwell_min": None}
            for i, c in enumerate(cands[: per_day * chips.days])
        ]
        return {"selected": fallback, "source": "mock"}
    return {"selected": picked, "source": "ai"}


def _clamp_dwell(raw) -> int | None:
    """LLM 이 낸 체류시간을 신뢰 구간으로 자른다. 파싱 불가면 None(카테고리 폴백)."""
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        return None
    return max(DWELL_FLOOR, min(DWELL_CEIL, v)) if v > 0 else None


async def enrich_node(state: AgentState) -> dict:
    """예상 혼잡도 반영 — 각 장소의 '방문 시각' 예보 레벨을 붙인다 (현재값 아님).

    시간표(schedule) 뒤에 온다: 방문 시각(start_time)을 알아야 그 시(時)의 예보를
    조회할 수 있기 때문. 혼잡도로 장소를 바꾸지 않는다 (표시 전용 — 선택 다양성 보존).
    등록된 실시간 측정 지점(area_name)이 있는 장소만 값이 붙는다. 병렬 조회.
    """
    selected = state.get("selected", [])
    if not selected:
        return {"congestion": {}}

    # 방문 시각 = 시간표의 place 슬롯 start_time. 시간표 없는 코스는 None → 현재값 폴백.
    start_by_name = {
        s["name"]: s.get("start_time")
        for s in state.get("schedule", [])
        if s.get("slot_type") == "place"
    }

    async def _fc(stop: dict) -> str | None:
        if not stop.get("area_name"):
            return None
        return await get_forecast_level(stop["area_name"], start_by_name.get(stop["name"]))

    levels = await asyncio.gather(*(_fc(s) for s in selected))
    congestion = {s["name"]: lvl for s, lvl in zip(selected, levels) if lvl}
    return {"congestion": congestion}


def _nearest_cards(pool: list[dict], stop: dict, radius_km: float, n: int) -> list[dict]:
    """식당 풀에서 이 스톱 반경 안 카드를 가까운 순 n개 (스톱 기준 dist_km 부착)."""
    near: list[tuple[float, dict]] = []
    for r in pool:
        if r.get("lat") is None or r.get("lng") is None:
            continue
        dist = haversine_km(stop["lat"], stop["lng"], r["lat"], r["lng"])
        if dist <= radius_km:
            near.append((dist, r))
    near.sort(key=lambda row: row[0])
    return [{**r, "dist_km": round(dist, 2)} for dist, r in near[:n]]


def _event_cards(events: list[dict], stop: dict, n: int) -> list[dict]:
    """스톱이 속한 명소의 실시간 행사 상위 n개 — 좌표가 있으면 거리도 붙인다.

    행사는 이미 그 스톱의 명소(area_name)에 속하므로 반경 필터로 걸러내지 않는다.
    """
    out: list[dict] = []
    for e in events[:n]:
        dist = (round(haversine_km(stop["lat"], stop["lng"], e["lat"], e["lng"]), 2)
                if e.get("lat") is not None and e.get("lng") is not None else None)
        out.append({**e, "dist_km": dist})
    return out


async def _live_meal_pool(selected: list[dict], chips: CourseChips) -> list[dict]:
    """권역 캐시가 비었을 때의 폴백 — 스톱 이름 + 동네 키워드로 식당을 라이브 조회.

    목록 API 는 키워드 검색이라 스톱 이름이 가장 좋은 단서다 (keyword="경복궁" →
    별빛야행·서촌 카페). 스톱마다 부르면 rate limit(~1.4 req/s)에 걸리므로 이름들을
    한 번에 키워드로 넘겨 식당 풀을 만든다. (예전 nearby 로직 — 이제 식당 전용.)
    """
    c_lat = sum(s["lat"] for s in selected) / len(selected)
    c_lng = sum(s["lng"] for s in selected) / len(selected)
    spread = max(haversine_km(c_lat, c_lng, s["lat"], s["lng"]) for s in selected)
    pool_radius = spread + max(_NEARBY_RADIUS_KM, MEAL_RADIUS_KM)

    terms = tuple(dict.fromkeys(t for loc in chips.locations for t in address_terms(loc)))
    keywords = tuple(place_keyword(s["name"]) for s in selected) + ((terms[0],) if terms else ())
    extra_terms = [
        t for s in selected for t in nearest_terms(s["lat"], s["lng"])
        if t not in keywords
    ]
    rest_keywords = keywords + tuple(dict.fromkeys(extra_terms))[:4]

    # 시간표 요청은 끼니마다 식당 3곳 이상이 필요해 예산을 더 태운다 (상세 1건당 ~0.75초).
    budget = min(8 + 6 * chips.days, 26) if chips.resolved_window() else _NEARBY_BUDGET
    try:
        items = await search_nearby(
            lat=c_lat, lng=c_lng, radius_km=pool_radius, region_terms=terms,
            keywords=rest_keywords, kinds=MEAL_KINDS, limit=budget, budget=budget,
        )
    except VisitSeoulError as err:
        logger.warning("visitseoul 식당 폴백 조회 실패: %s", err)
        return []

    pool: list[dict] = []
    seen: set[str] = set()
    for it in items:
        d = it.detail
        if d.lat is None or d.lng is None or d.title in seen:
            continue
        seen.add(d.title)
        pool.append(meal_card(d, it.kind))
    return pool


async def nearby_node(state: AgentState) -> dict:
    """확정된 각 장소의 주변 정보 — 식당(권역 캐시) + 실시간 행사(citydata).

    3분할: 코스 장소는 임베딩(places.json)이, 식당은 Visit Seoul 권역 캐시가,
    행사는 서울시 citydata 가 담당한다.

    식사 슬롯의 식당은 meals 노드가 따로 채운다 — 여기 식당 풀은 "스톱 주변에 뭐가
    있나" 를 보여주는 카드용이다. 두 용도가 다르므로 분리해 둔다.
    """
    selected = state.get("selected", [])
    if not selected:
        return {"nearby": {}}
    chips = _chips(state)

    # 식당 — 권역 캐시 우선. 캐시가 없거나 얇으면 라이브 조회로 보충 (제목 중복 제거).
    pool = state.get("meal_pool") or pool_for_stops(selected)
    if len(pool) < get_settings().meal_pool_min:
        seen = {r["title"] for r in pool}
        pool += [r for r in await _live_meal_pool(selected, chips) if r["title"] not in seen]

    # 행사 — 스톱이 속한 명소(area_name)의 citydata 실시간 행사. 명소당 한 번만 조회.
    areas = [a for a in {s.get("area_name") for s in selected} if a]
    events_by_area = dict(zip(areas, await asyncio.gather(*(get_events(a) for a in areas))))

    nearby = {
        s["name"]: {
            "restaurants": _nearest_cards(pool, s, _NEARBY_RADIUS_KM, _NEARBY_PER_STOP),
            "attractions": _event_cards(events_by_area.get(s.get("area_name"), []), s, _NEARBY_PER_STOP),
        }
        for s in selected
    }
    return {"nearby": nearby, "meal_pool": pool}


def _by_day(selected: list[dict]) -> dict[int, list[dict]]:
    """selected → 일차별 스톱. 하루 코스는 전부 1일차. 순서는 selected 그대로."""
    out: dict[int, list[dict]] = {}
    for s in selected:
        out.setdefault(s.get("day") or 1, []).append(s)
    return out


def _stop_chunks(selected: list[dict]) -> list[list[dict]]:
    """스톱을 compose 콜 단위로 묶는다 — **일차 경계를 넘지 않게, 크기는 고르게**.

    한 콜 안의 장소는 같은 날이어야 "일차 흐름이 이어지게 쓰라"는 규칙이 성립한다.
    날이 갈리면 콜도 가른다.

    크기를 고르게 나누는 게 중요하다. 벽시계는 **가장 긴 콜 하나**로 결정되는데,
    앞에서부터 3곳씩 자르면 하루 4곳이 3+1 로 갈려 3곳짜리 콜이 혼자 병목이 된다
    (실측: 3곳 965자 10.6s vs 1곳 280자 3.6s). 2+2 면 둘 다 짧다.
    """
    chunks: list[list[dict]] = []
    for _, rows in sorted(_by_day(selected).items()):
        n_calls = max(1, -(-len(rows) // _STOPS_PER_CALL))  # ceil
        size, extra = divmod(len(rows), n_calls)
        i = 0
        for j in range(n_calls):
            take = size + (1 if j < extra else 0)
            chunks.append(rows[i : i + take])
            i += take
    return chunks


def _match_stop_name(raw: str | None, names: list[str]) -> str | None:
    """LLM 이 답한 장소명을 확정 장소명으로 정규화. 못 맞추면 None.

    LLM 이 프롬프트의 "장소명 (분류)" 표기를 name 에 그대로 복사하기도 해서
    괄호 접미사를 떼고 다시 맞춰 본다 (안 맞추면 서사가 빈 카드로 나간다).
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw in names:
        return raw
    head = raw.split(" (")[0].strip()
    if head in names:
        return head
    return next((n for n in names if raw.startswith(n) or n in raw), None)


async def compose_node(state: AgentState) -> dict:
    """확정된 장소에 코스 서사를 입힌다. 장소·개수는 고정, 순서는 프론트가 다시 계산한다.

    LLM 콜은 전역 서사 1개 + 스톱 청크 N개를 **동시에** 던진다 (레이턴시 — 위 프롬프트
    주석 참고). 이 노드는 그 결과를 합쳐 최종 stops 페이로드까지 조립한다.
    """
    selected = state.get("selected", [])
    req = state.get("req", {})
    chips = _chips(state)
    congestion = state.get("congestion", {})
    nearby = state.get("nearby", {})
    schedule = state.get("schedule", [])
    sched_by_name = {s["name"]: s for s in schedule if s.get("slot_type") == "place"}
    flex_names = {s["name"] for s in schedule if s.get("slot_type") == "flex"}

    def _slot_label(name: str) -> str:
        slot = sched_by_name.get(name)
        if slot:
            return f" · {slot['start_time']}~{slot['end_time']}"
        return " · 시간표 밖 자유 방문 제안" if name in flex_names else ""

    def _stop_line(s: dict) -> str:
        return (
            f"- {str(s['day']) + '일차 · ' if s.get('day') else ''}{s['name']} ({s.get('category','')})"
            f"{_slot_label(s['name'])}"
            f"{' · 예상 혼잡도 ' + congestion[s['name']] if s['name'] in congestion else ''}"
            f"{' · 선정 이유: ' + s['reason'] if s.get('reason') else ''}"
            f"{' · 여기서 할 일: ' + ', '.join(s['activities']) if s.get('activities') else ''}"
        )

    stop_lines = "\n".join(_stop_line(s) for s in selected)

    narrative: dict[str, dict] = {}
    title = subtitle = description = ""
    tags: list[str] = []
    day_descriptions: dict[int, str] = {}
    source = state.get("source", "ai")
    # 멀티데이는 전체 description 을 짧게 두고 일차별 설명(day_descriptions)으로 쪼갠다 —
    # 클라 일차 탭에 맞춰 그 일차 설명만 보이게. 하루 코스는 description 하나로 흐름을 담는다.
    day_desc_rule = (
        f"- 이 코스는 {chips.days}일 일정이다. **description(코스 전체)에는 날짜별 서술을 넣지 말고** "
        "일정 전체가 어떤 구성인지만 2~3문장으로 짧게 쓴다 (일차별 동선 요약은 따로 쓴다).\n"
        if chips.days > 1
        else "- 하루 코스다. description 4~6문장으로 하루 동선을 담는다.\n"
    )

    comp_rule = chips.companion_rule()
    act = chips.purpose_rule().get("act", "")
    lens = {
        "note": req.get("note", "서울 코스"),
        "chips": chips.summary(),
        "companion_lens": f"- 동반 렌즈(어조·관계): {comp_rule}\n" if comp_rule else "",
        "purpose_lens": f"- 목적별 행동 결: {act}\n" if act else "",
    }

    # 전역 서사 1콜 + 일차 요약 N콜 + 스톱 카드 M콜을 동시에 던진다. 서로 참조하지 않으므로
    # 안전하고, 벽시계는 가장 느린 콜 하나로 수렴한다. 그래서 **콜당 출력을 고르게 나누는
    # 것**이 핵심이다 — 일차 요약을 전역에서 떼어낸 것도 그 콜이 혼자 길어졌기 때문이다
    # (실측: 2일 코스 전역 콜 997자 11.5s → 일차 분리 후 절반 이하).
    #
    # 부분 실패는 그 콜의 몫만 잃는다 — 전역이 죽으면 제목/소개가 폴백(mock),
    # 일차 콜이 죽으면 그 날 요약만 비고, 스톱 청크가 죽으면 그 장소들만 빈 카드.
    day_stops = _by_day(selected)
    day_keys = sorted(day_stops) if chips.days > 1 else []
    chunks = _stop_chunks(selected)
    results = await asyncio.gather(
        _llm_call("compose 전역", _COMPOSE_GLOBAL_PROMPT | get_llm(),
                  {**lens, "stops": stop_lines, "day_desc_rule": day_desc_rule}),
        *(
            _llm_call(f"compose {d}일차", _COMPOSE_DAY_PROMPT | get_llm(),
                      {**lens, "day": d,
                       "stops": "\n".join(_stop_line(s) for s in day_stops[d])})
            for d in day_keys
        ),
        *(
            _llm_call(f"compose 스톱 {i + 1}/{len(chunks)}",
                      _COMPOSE_STOPS_PROMPT | get_llm(),
                      {**lens, "stops": "\n".join(_stop_line(s) for s in chunk)})
            for i, chunk in enumerate(chunks)
        ),
        return_exceptions=True,
    )
    global_res = results[0]
    day_res = results[1 : 1 + len(day_keys)]
    stop_res = results[1 + len(day_keys) :]

    if isinstance(global_res, BaseException):
        logger.warning("compose 전역 LLM 실패, mock 서사: %s", global_res)
        source = "mock"
    else:
        try:
            data = parse_json_object(global_res)
            title = data.get("title", "")
            subtitle = data.get("subtitle", "")
            description = data.get("description", "")
            tags = data.get("tags", []) or []
        except Exception as err:  # noqa: BLE001
            logger.warning("compose 전역 파싱 실패, mock 서사: %s", err)
            source = "mock"

    for d, res in zip(day_keys, day_res):
        if isinstance(res, BaseException):
            logger.warning("compose %d일차 요약 실패: %s", d, res)
            continue
        try:
            text = str(parse_json_object(res).get("day_description") or "").strip()
        except Exception as err:  # noqa: BLE001
            logger.warning("compose %d일차 요약 파싱 실패: %s", d, err)
            continue
        if text:
            day_descriptions[d] = text

    sel_names = [s["name"] for s in selected]
    for chunk, res in zip(chunks, stop_res):
        names = [s["name"] for s in chunk]
        if isinstance(res, BaseException):
            logger.warning("compose 스톱 청크 실패 (%s): %s", ", ".join(names), res)
            continue
        try:
            rows = parse_json_object(res).get("stops", [])
        except Exception as err:  # noqa: BLE001
            # 잘린 JSON 이라도 완결된 스톱 객체는 건진다 (청크가 통째로 빈 카드가 되지 않게)
            rows = [o for o in salvage_objects(res) if o.get("name")]
            logger.warning("compose 스톱 청크 파싱 실패 — 부분 복구 %d곳: %s", len(rows), err)
        for row in rows:
            name = _match_stop_name(row.get("name"), names) or _match_stop_name(
                row.get("name"), sel_names
            )
            if name:
                narrative[name] = row

    def _place_stop(s: dict) -> dict:
        n = narrative.get(s["name"], {})
        lvl = congestion.get(s["name"])
        tip = n.get("tip")
        slot = sched_by_name.get(s["name"])
        if lvl and lvl != "여유":
            # 방문 시각이 있으면 "18시 예상 혼잡도 …" 로, 없으면 시각 없이
            hh = slot["start_time"][:2].lstrip("0") if slot and slot.get("start_time") else ""
            note = f"{hh}시 예상 혼잡도 {lvl}" if hh else f"예상 혼잡도 {lvl}"
            tip = f"{tip} · {note}" if tip else note
        # 체류시간은 계산값이다 — 시간표가 있으면 그 슬롯, 없으면 select 가 낸 dwell_min.
        # compose LLM 에게는 더 이상 duration 을 묻지 않는다: 시간표가 있으면 어차피
        # 버려지던 값이라 출력 토큰(=레이턴시)만 먹었고, 없을 때도 dwell_min 이 더 정확하다.
        dur_min = slot["duration_min"] if slot else s.get("dwell_min")
        duration = duration_label(dur_min) if dur_min else "1시간"
        return {
            "name": s["name"],
            "preview": n.get("preview", s.get("text", "")[:40]),
            "description": n.get("description", ""),
            "duration": duration,
            # 클라이언트가 분 단위를 직접 계산하지 않게 정수도 같이 보낸다
            # (duration 은 "1시간 30분" 같은 표시 문자열이라 파싱해야 했다)
            "duration_min": dur_min,
            "tip": tip,
            "lat": s["lat"],
            "lng": s["lng"],
            "reason": s.get("reason", ""),
            "activities": s.get("activities", []),
            "congestion": lvl,
            "nearby": nearby.get(s["name"], {"restaurants": [], "attractions": []}),
            "start_time": slot["start_time"] if slot else None,
            "end_time": slot["end_time"] if slot else None,
            "slot_type": "flex" if s["name"] in flex_names else "place",
            "day": s.get("day"),
            "meal_options": [],
            "travel_min": (slot.get("travel_min") or None) if slot else None,
            "travel_mode": (slot.get("travel_mode") or None) if slot else None,
        }

    def _meal_stop(m: dict) -> dict:
        dur_min = m.get("duration_min") or MEAL_DURATION_MIN
        return {
            "name": m["name"],
            "preview": m.get("summary", ""),
            "description": "",
            "duration": duration_label(dur_min),
            "duration_min": dur_min,
            "tip": None,
            "lat": m.get("lat"),
            "lng": m.get("lng"),
            "reason": "",
            "activities": [],
            "congestion": None,
            "nearby": {"restaurants": [], "attractions": []},
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "slot_type": "meal",
            "day": m.get("day"),
            "meal_options": m.get("meal_options", []),
            # place 슬롯과 키 모양을 맞춘다 — 예전엔 meal 에만 이 키가 없어서
            # 클라이언트가 undefined 와 null 을 구분해야 했다
            "travel_min": None,
            "travel_mode": None,
        }

    by_name = {s["name"]: s for s in selected}
    if schedule:
        # 시간표 순서 그대로 — 장소 사이에 식사 슬롯이 끼어든다
        stops = [
            _meal_stop(slot) if slot["slot_type"] == "meal" else _place_stop(by_name[slot["name"]])
            for slot in schedule
            if slot["slot_type"] == "meal" or slot["name"] in by_name
        ]
    else:
        stops = [_place_stop(s) for s in selected]

    course = {
        "title": title or "서울 추천 코스",
        "subtitle": subtitle,
        "description": description,
        "stops": stops,
        "tags": tags or ["코스"],
        "scheduled": bool(sched_by_name),
        "days": chips.days,
        "day_areas": state.get("day_areas") or {},
        "day_descriptions": day_descriptions,  # 멀티데이 일차별 설명 (하루 코스는 빈 객체)
    }
    return {"result": {"course": course, "source": source}, "source": source}
