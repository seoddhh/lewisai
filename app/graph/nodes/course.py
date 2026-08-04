"""코스를 만드는 노드들. 검색으로 후보를 모으고 → AI 가 장소를 고르고 →
혼잡도·주변 정보를 붙이고 → 시간표를 짜고 → 소개 문구를 쓰는 순서로 흘러간다.

여기(서버)가 정하는 것: 어떤 장소를 왜 골랐는지, 거기서 뭘 할지, 그리고 방문 순서까지.
프론트가 하는 것: 그 순서를 받아 지도에 실제 도로 경로(폴리라인)와 거리를 그리는 것.

주변 정보는 출처가 셋으로 나뉜다 — 식당은 미리 구워둔 캐시(meal_cache),
행사는 서울시 실시간 API, 관광지는 임베딩 검색.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
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

_NEARBY_RADIUS_KM = 1.5   # 장소 주변 정보를 몇 km 안에서 찾을지
_NEARBY_PER_STOP = 2      # 장소 하나에 붙일 식당·행사 개수
_NEARBY_BUDGET = 14       # 코스 하나를 만들며 Visit Seoul 을 부를 수 있는 최대 횟수(호출 제한 때문)

# ── 후보 검색 설정 — 코스는 무엇보다 장소들이 서로 가까워야 한다 ──────────
_FETCH_K = 40                    # 한 번 검색할 때 넉넉히 가져오는 후보 수
_MIN_POOL = 12                   # 후보가 이보다 적으면 조건을 한 단계 풀고 다시 검색한다
_RADIUS_STEPS = (5.0, 7.0, 10.0)  # 기준점에서 몇 km 안까지 볼지. 부족하면 순서대로 넓힌다
_GEO_ALPHA = 0.3                 # 점수에서 의미 유사도가 차지하는 비중. 나머지는 거리 몫이다
# 후보가 수백 m 안에 몰려 있으면, 그 안의 사소한 거리 차이가 0~1 로 부풀려져
# 의미 점수를 눌러버린다. 어차피 다 걸어갈 거리라 최소 2km 로 놓고 계산한다.
_DIST_SCALE_FLOOR_KM = 2.0
# 후보를 고를 때 상위 몇 곳 중에서 무작위로 뽑을지. 이 안은 점수가 비슷하다고 본다.
# 키우면 매번 다른 코스가 나오지만 품질이 떨어진다. 3이면 세 번 다시 만들어도 거의 안 겹친다.
_PICK_WINDOW = 3
# 식당 후보를 실시간으로 받아올 때의 반경. 실제로 어느 식당을 넣을지 고르는 반경은
# meals 노드가 따로 갖고 있고, 여기 값은 얼마나 넓게 받아올지만 정한다.
MEAL_RADIUS_KM = 3.0

# 여행자가 여러 날 오면서 동네를 안 골랐을 때, 날마다 다른 동네를 배정하는 데 쓴다.
# 첫날은 목적에 맞는 동네로 시작하고 이후로는 아래 순서대로 돈다.
_AREA_ROTATION = ("종로·중구", "홍대·마포", "성수·건대", "강남·서초", "용산·이태원", "잠실·송파")
_PURPOSE_FIRST_AREA = {
    "핫플레이스": "성수·건대", "체험·놀거리": "홍대·마포",
    "쇼핑": "강남·서초", "자연·힐링": "여의도·영등포",
    "맛집 탐방": "종로·중구", "문화·예술": "종로·중구", "관광 명소": "종로·중구",
}


def _day_areas(chips: CourseChips) -> dict[int, str] | None:
    """며칠째에 어느 동네를 갈지 정한다. 여행자가 여러 날 올 때만 쓴다.

    동네를 여러 개 골랐으면 고른 순서대로 하루씩 배정한다.
    하나만 골랐으면 그 동네에서만 놀고 싶다는 뜻이므로 나누지 않는다(None).
    아무것도 안 골랐으면 목적에 맞는 동네부터 시작해 날마다 옮겨 다닌다.
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
            "- **스톱 개수({count}곳)를 먼저 정확히 채운 뒤, 그 안에서 가능한 한 종류(후보 줄의 "
            "`종류=` 값: 자연·역사·문화·쇼핑·명소·체험)를 섞어라.** 같은 종류를 3곳 이상 넣지 말고"
            "(예: 백화점·면세점·쇼핑몰만 이어 붙이기), 후보에 다른 종류가 있으면 그쪽을 우선 고른다. "
            "**개수를 줄여서 다양성을 맞추지 말 것 — 개수가 먼저다.**\n"
            "- 후보 줄의 `실내/야외 · 조용/활기 · 밤가능 · 보통N분` 은 그 장소의 성격이다. "
            "동반·목적 조건에 맞는 성격을 우선 고르고, **하루 안에서 성격도 한쪽으로 쏠리지 않게** "
            "한다(실내만 5곳, 활기찬 곳만 5곳 금지).\n"
            "- `있는 것:` 에 적힌 시설·볼거리가 그 장소에 **실제로 있는 것**이다. "
            "activities 는 여기에 걸어 쓰고, 없는 것을 지어내지 말 것.\n"
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

# 글쓰기는 "코스 전체 소개"와 "장소별 카드" 두 프롬프트로 나눠서 동시에 돌린다.
# 한 번에 다 쓰게 하면 결과물이 길어지는 만큼 느려지는데, 나누면 그만큼 빨라진다.
#
# 대신 아래 문체 규칙은 두 프롬프트에 다 넣는다.
# 나눠 써도 한 사람이 쓴 글처럼 읽혀야 하기 때문이다.
_STYLE_RULES = (
    "- **여행 플래너의 안내문처럼 쓴다** — 무엇을 보고 어디로 이동하고 얼마나 걸리는지 같은 실용 정보가 먼저다. "
    "분위기 묘사는 곁들이되 문단당 한 문장을 넘기지 말고, **'설렘·낭만·감성·감성적인·감성 가득·마법 같은·"
    "특별한 추억·잊지 못할·힐링 가득' 처럼 장소를 안 겪어도 쓸 수 있는 광고성 수식과 감탄은 쓰지 말 것.** "
    "판단 기준은 단어 목록이 아니라 이것이다 — **그 표현을 다른 장소에 그대로 옮겨도 말이 되면 빼라.** "
    "형용사보다 구체적인 사실로 쓴다.\n"
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
            "- **description 은 동선 서술이 아니라 '이 코스를 왜 이렇게 구성했는지' 설명이다.** "
            "사용자가 준 요청 문장·선택 조건을 근거로, 그것이 이 코스에 어떻게 반영됐는지 쓴다. "
            "**문장은 3개까지만 쓴다(4문장 이상 금지).** 한 문장에 한 가지 근거만 담고 길게 늘이지 않는다. "
            "**장소 이름은 통틀어 두 곳까지만** 예로 들 수 있다 — 고른 장소를 전부 훑으면 안 된다. "
            "장소를 순서대로 훑거나 '어디에 들렀다가 어디로 간다'는 식의 동선 나열은 쓰지 말 것 — "
            "그건 시간표와 장소 카드가 따로 보여준다.\n"
            "- **description 에 숫자 시간을 절대 쓰지 말 것.** '14:00', '2시', '12시부터 18시까지', "
            "'6시간 동안', '저녁 7시 식사' 처럼 시각·소요시간을 나타내는 표현은 선택 조건에 시간 범위나 "
            "식사 시각이 들어 있더라도 옮겨 적지 않는다. 끼니는 '저녁 식사를 넣었어요'처럼 시각 없이 쓴다. "
            "시간 얘기는 '오전·오후·밤' 중 사용자가 고른 값을 되짚을 때만 허용한다.\n"
            "- **장소 이름을 방문 순서대로 잇지 말 것.** 'A를 둘러본 뒤 B로 이동해 C까지' 같은 문장은 "
            "금지다. 장소명은 근거를 짚을 때 예시로 한둘만 들 수 있고, 코스 전체를 훑는 데 쓰지 않는다.\n"
            "- **방문 순서를 가리키는 표현도 쓰지 말 것** — '처음에·먼저·마지막에·이어서·초반에' 같은 말로 "
            "특정 장소의 자리를 단정하면 안 된다. 순서는 주어지지 않았고 다른 곳에서 정한다.\n"
            "- **선택 조건에 적히지 않은 항목은 이 코스에 없는 것이다.** 없는 것을 있다고 쓰지 말 것 "
            "— 조건에 끼니가 없는데 '저녁 식사를 넣었어요'라고 쓰면 안 된다. "
            "동시에 **없다는 사실을 굳이 밝히지도 말 것** ('점심을 포함하지 않아', '혼잡도 조건이 없어' 금지). "
            "주어진 조건만 가지고 쓴다.\n"
            "- 첫 문장은 **무엇을 바탕으로 골랐는지**를 밝힌다 — 사용자가 준 조건을 그대로 되짚어 "
            "'~를 바탕으로 코스를 구성했어요' 꼴로 쓴다 "
            "(예: '친구와 종로·중구에서 보낼 오후, 문화·예술 위주라는 조건을 바탕으로 3곳을 골랐어요'). "
            "요청 문장(note)이 있으면 그 내용을 우선 근거로 삼고, 없으면 선택 조건만으로 쓴다. "
            "**주어지지 않은 조건은 지어내지 말 것** — 고르지 않은 항목은 아예 언급하지 않는다.\n"
            "- 나머지 문장은 **그 조건이 구성에 반영된 방식**을 사실로 짚는다. 쓸 수 있는 근거는 예를 들어 "
            "어떤 권역으로 묶어 이동을 줄였는지, 실내·실외 비중(날씨 영향), 장소 수와 일정 밀도(빼곡/여유), "
            "끼니를 골랐다면 어느 대목에 넣었는지, 혼잡도 선호를 어떻게 반영했는지, 여러 날이면 날짜를 "
            "어떤 기준으로 나눴는지다. **[확정된 장소] 목록에서 실제로 확인되는 것만 쓴다.**\n"
            "{day_desc_rule}"
            "- **누구와(동반) × 무엇하러(목적)에 따라 title·subtitle·description 의 강조점과 소재가 달라져야 한다** — "
            "아래 렌즈에 맞춰, 같은 장소들이라도 그 조합에 맞는 내용으로 쓴다(다른 조합엔 안 맞게). "
            "예: 연인과+데이트 → 둘이 걷기 좋은 구간·야경 시간대, 친구와+핫플 → 함께 즐길 거리·요즘 찾는 곳, "
            "부모님과+힐링 → 앉아 쉴 곳·짧은 이동 거리.\n"
            + _STYLE_RULES +
            "  title·subtitle 은 명사형 가능.\n"
            "{companion_lens}"
            "{purpose_lens}"
            "- [확정된 장소]에는 시각도 일차도 적혀 있지 않다. **모르는 값을 지어내지 말 것** — "
            "방문 시각, 며칠째인지, 장소 간 이동 시간을 추측해 쓰면 안 된다.\n"
            "- 장소 개수를 쓸 때는 주어진 '확정 장소 수'를 **그대로** 쓴다. 직접 세지 말 것.",
        ),
        ("human",
         '요청: "{note}"\n선택 조건: {chips}\n확정 장소 수: {n_places}곳\n\n[확정된 장소]\n{stops}'),
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


_STOPS_PER_CALL = 3


def _fallback_description(chips: CourseChips, note: str, n_stops: int) -> str:
    """AI 를 안 쓰고 칩만으로 코스 소개글을 만든다. 소개글 생성이 실패했을 때 쓴다.

    소개글이 어차피 "고른 조건을 어떻게 반영했는지"를 쓰는 자리라,
    칩만 있어도 골자는 만들 수 있다. 문장이 매끄럽진 않아도 빈칸으로 나가진 않는다.
    """
    subject = "·".join(chips.companions) + " 함께" if chips.companions else None
    where = "·".join(chips.locations) if chips.locations and "상관없음" not in chips.locations else None
    what = "·".join(chips.purposes) if chips.purposes else None

    # 고른 조건만 나열한다. 안 고른 항목은 아예 언급하지 않는다.
    given = [p for p in (subject, where, chips.time, what) if p]
    if note.strip():
        given.append(f'"{note.strip()}"')
    lead = (
        f"{', '.join(given)} 조건을 바탕으로 {n_stops}곳을 골랐어요."
        if given
        else f"주신 조건을 바탕으로 {n_stops}곳을 골랐어요."
    )

    extra = []
    if chips.days > 1:
        extra.append(f"{chips.days}일 일정이라 날마다 권역을 묶어 이동을 줄였어요")
    if chips.pace:
        extra.append("일정은 빼곡하게 채웠어요" if chips.pace == "packed" else "일정은 여유 있게 뒀어요")
    if chips.meals:
        extra.append(f"{'·'.join(chips.meals)} 식사를 코스 안에 넣었어요")
    if chips.congestion and chips.congestion != "상관없음":
        extra.append(f"혼잡도는 '{chips.congestion}' 선호를 반영했어요")
    return lead + (" " + ", ".join(extra) + "." if extra else "")


def _chips(state: AgentState) -> CourseChips:
    raw = state.get("req", {}).get("chips") or {}
    try:
        return CourseChips(**raw)
    except Exception:  # noqa: BLE001 — 칩 값이 이상해도 코스는 만들어 줘야 한다
        logger.warning("칩 파싱 실패, 기본값 사용: %s", raw)
        return CourseChips()


def _cand(doc) -> dict:
    """검색 결과 문서 하나를 코스에서 다루기 쉬운 후보 딕셔너리로 바꾼다."""
    m = doc.metadata
    return {
        # 이름 앞뒤 공백은 꼭 지운다. 원본 데이터에 공백이 섞여 있는데 AI 는 공백 없는
        # 이름으로 답하기 때문에, 안 지우면 이름이 안 맞아 고른 장소가 조용히 사라진다.
        "name": (m.get("display_name") or "").strip(),
        "lat": m.get("lat"),
        "lng": m.get("lng"),
        # 장소 종류. 하루 안에 비슷한 곳만 몰리지 않게 하는 기준이 된다
        "category": m.get("category", ""),
        "area": m.get("area", ""),
        "area_name": m.get("area_name", ""),
        "description": m.get("description", ""),
        "highlights": [h for h in (m.get("highlights") or "").split(",") if h],
        # 아래 네 개는 동반자·목적에 맞춰 순서를 조정할 때 쓴다. 태그가 없으면 None 이고 가점 0이다
        "indoor": m.get("indoor"),
        "night_ok": m.get("night_ok"),
        "stay_min": m.get("stay_min"),
        "energy": m.get("energy"),
        "op_start": int(m.get("op_start", 0)),
        "op_end": int(m.get("op_end", 24)),
        # 운영시간이 0~24시일 때, 정말 24시간인지 아니면 그냥 모르는 건지 구분해 준다
        "hours_known": bool(m.get("hours_known", False)),
        # 좌표가 같은(사실상 같은 자리인) 장소들의 묶음 이름. 한 코스에 하나만 넣으려고 쓴다
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
    """요청한 시간대에 이 장소가 닫혀 있는지 본다.

    운영시간을 모르는 곳은 판단하지 않고 통과시킨다. 확실히 안 맞는 곳만 걸러낸다
    (안 그러면 밤 코스에 18시에 닫는 박물관이 섞인다).
    """
    if not window or not c.get("hours_known"):
        return False
    return not window.overlaps(c["op_start"], c["op_end"])


def _dedupe_same_place(cands: list[dict]) -> list[dict]:
    """같은 자리에 있는 장소들 중 점수가 제일 높은 하나만 남긴다.

    광화문광장과 해치마당처럼 이름은 다른데 사실상 같은 곳인 경우가 있다.
    AI 에게 보여주기 전에 미리 걸러야 둘 다 고르는 일이 없다.
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


def _pick_seed(req: dict) -> int:
    """후보를 뽑을 때 쓸 시드를 정한다.

    "다시 만들기"로 들어온 seed 가 있으면 그걸 쓴다. 없으면 요청 내용을 해시해서 만든다.
    이렇게 하면 같은 요청은 항상 같은 코스가 나오고(디버깅이 된다),
    다시 만들기를 누르면 다른 코스가 나온다.
    """
    if (s := req.get("seed")) is not None:
        try:
            return int(s)
        except (TypeError, ValueError):
            pass
    raw = json.dumps({"chips": req.get("chips"), "note": req.get("note", "")},
                     sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def _personal_bonus(c: dict, w: dict) -> float:
    """후보 한 곳이 동반자·목적 조건에 얼마나 맞는지를 점수로 매긴다(최대 ±0.2).

    후보에서 빼는 게 아니라 순서만 민다. 걸러내면 후보가 너무 적어지기 때문이다.
    태그가 없는 장소는 0점이다. 감점이 아니라 중립이라, 태그가 없다고 밀려나지 않는다.
    """
    b = 0.0
    if c.get("indoor") is not None:
        b += w["indoor"] if c["indoor"] else -w["indoor"]
    if c.get("night_ok") is not None:
        b += w["night_ok"] if c["night_ok"] else -w["night_ok"]
    if energy := c.get("energy"):
        b += w["energy"].get(energy, 0.0)
    # "아이와"처럼 오래 머물기 부담스러운 조합에서만, 체류시간이 긴 곳을 깎는다
    if w["stay_max"] and (stay := c.get("stay_min")) and stay > w["stay_max"]:
        b -= w["stay_penalty"]
    return max(-0.2, min(0.2, b))


def _quota_pick(ranked: list[dict], want: int, seed: int) -> list[dict]:
    """장소 종류별로 돌아가며 하나씩 뽑아, 한 종류가 후보를 독차지하지 않게 한다.

    그냥 점수 순으로 자르면 데이터가 많은 종류가 다 차지한다. 예를 들어 쇼핑 장소가
    유난히 많아서, 쇼핑 목적으로 검색하면 후보가 전부 백화점이 되어 버린다.
    그러면 AI 에게 "종류를 섞어라"라고 해도 섞을 후보 자체가 없다.

    각 종류에서 1등을 그대로 뽑지 않고 상위 몇 곳 중 하나를 무작위로 고른다.
    항상 1등만 뽑으면 다시 만들어도 늘 같은 코스가 나오기 때문이다.
    다만 점수가 비슷한 상위권 안에서만 흔들어야 품질이 안 떨어진다.
    """
    rng = random.Random(seed)
    order = {c["name"]: i for i, c in enumerate(ranked)}
    queues: dict[str, list[dict]] = {}
    for c in ranked:
        queues.setdefault(c.get("category") or "기타", []).append(c)

    out: list[dict] = []
    while len(out) < want and any(queues.values()):
        # 이번 바퀴에 돌 종류들. 각 종류의 1등이 잘한 순서대로 돈다
        heads = sorted((q for q in queues.values() if q), key=lambda q: order[q[0]["name"]])
        for q in heads:
            if len(out) >= want:
                break
            out.append(q.pop(rng.randrange(min(_PICK_WINDOW, len(q)))))
    return out


def _geo_rerank(
    pool: list[tuple[dict, float]], anchor: tuple[float, float], want: int,
    *, weights: dict | None = None,
) -> list[dict]:
    """기준점(anchor) 근처만 남긴 뒤, 거리·의미·개인 취향을 합친 점수로 다시 줄 세운다.

    pool 은 (후보, 의미 점수) 쌍이고, 이 점수는 작을수록 검색어와 비슷하다는 뜻이다.
    돌려주는 후보에는 기준점에서 몇 km 인지가 dist_km 로 붙는다.
    """
    with_dist = [
        (c, sim, haversine_km(anchor[0], anchor[1], c["lat"], c["lng"]))
        for c, sim in pool
    ]
    # 반경을 5 → 7 → 10km 로 넓히며 필요한 만큼 모은다. 그래도 모자라면 반경을 포기한다.
    near: list[tuple[dict, float, float]] = []
    for radius in _RADIUS_STEPS:
        near = [row for row in with_dist if row[2] <= radius]
        if len(near) >= want:
            break
    if not near:
        near = with_dist

    # 의미 점수와 거리를 각각 0~1 로 맞춘 뒤 섞는다. 둘 다 작을수록 좋으니 뒤집어서 1이 최고가 되게 한다
    sims = [s for _, s, _ in near]
    dists = [d for _, _, d in near]
    s_lo, s_hi = min(sims), max(sims)
    d_lo = min(dists)
    d_hi = max(max(dists), d_lo + _DIST_SCALE_FLOOR_KM)

    def _norm(v: float, lo: float, hi: float) -> float:
        return 1.0 if hi == lo else (hi - v) / (hi - lo)

    w_sim, w_dist = _GEO_ALPHA, 1 - _GEO_ALPHA
    w = weights or {"indoor": 0.0, "night_ok": 0.0, "energy": {}, "stay_max": None,
                    "stay_penalty": 0.0}
    scored = [
        (dict(c, dist_km=round(d, 2)),
         w_sim * _norm(s, s_lo, s_hi) + w_dist * _norm(d, d_lo, d_hi) + _personal_bonus(c, w))
        for c, s, d in near
    ]
    scored.sort(key=lambda row: row[1], reverse=True)
    return [c for c, _ in scored]


def _search_pool(query: str, *, area: str | None, slugs: list[str],
                 k: int = _FETCH_K, window=None) -> list[tuple[dict, float]]:
    """후보 장소를 검색해 모은다. 동네와 목적으로 좁혀 찾되, 모자라면 조건을 하나씩 푼다.

    푸는 순서는 동네+목적 → 동네만 → 목적만 → 조건 없음이다. 앞 단계에서 나온 후보가
    앞자리를 지키므로, 동네가 맞는 곳이 먼저 온다.

    동네로 안 좁히면 데이터가 많은 종로·용산이 후보를 다 차지해서,
    강남을 요청해도 종로 장소가 딸려 나온다.

    window 를 주면 그 시간대에 닫는 곳은 아예 뺀다. 개수를 세기 전에 빼야
    문 닫은 곳으로 숫자만 채우는 일이 없다.
    """
    base: dict = {"doc_type": "place"}
    if area:
        base["area"] = area
    # 고른 목적 중 하나라도 태그가 붙은 장소. 이 태그는 인제스트할 때 심어둔 것이다
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
            if _closed_in_window(c, window):     # 그 시간에 닫는 곳은 뺀다
                continue
            seen.add(c["name"])
            pool.append((c, sim))
        if len(pool) >= _MIN_POOL:
            break
    return pool


async def retrieve_node(state: AgentState) -> dict:
    """후보 장소를 모아 오는 단계. 코스는 장소들이 서로 가까워야 하므로,
    기준점을 하나 잡고 그 주변에서 찾은 뒤 다시 줄 세운다.

    기준점은 지역 칩이 있으면 그 동네 중심이고, 없으면 검색 1위 장소의 좌표다.
    "종로에서…"처럼 문장으로 말해도 검색 상위가 종로 쪽이라 자연히 기준점이 잡힌다.
    """
    req = state.get("req", {})
    chips = _chips(state)
    rule = chips.purpose_rule()
    slugs = chips.purpose_slugs()
    weights = chips.rerank_weights()
    seed = _pick_seed(req)

    # 목적은 칩 이름 그대로가 아니라, 미리 정해둔 검색어(PURPOSE_RULES 의 query)로 바꿔 검색한다
    query = " ".join(
        p for p in [req.get("note", ""), rule.get("query", " ".join(chips.purposes)),
                    " ".join(chips.companions), " ".join(chips.locations), chips.time or ""] if p
    ) or "서울 코스"

    window = chips.resolved_window()
    day_areas = _day_areas(chips)
    if day_areas:
        # 날마다 동네가 다르면, 날짜별로 따로 검색해서 후보에 며칠째인지(day_hint)를 붙인다.
        # 앞날에 쓴 장소는 뒷날 후보에서 빼서 같은 곳이 두 번 안 나오게 한다.
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
            ranked = _dedupe_same_place(
                _geo_rerank(pool_d, (area_chip.lat, area_chip.lng), per, weights=weights))
            # 날마다 시드를 다르게 준다. 같은 동네가 여러 날 걸려도 같은 장소가 안 뽑히게
            picks = _quota_pick(ranked, per, seed + d)
            taken.update(c["name"] for c in picks)
            cands.extend({**c, "day_hint": d} for c in picks)
        return {"candidates": cands, "day_areas": day_areas}

    area = next((loc for loc in chips.locations if chip_of(loc)), None)
    pool = _search_pool(query, area=area, slugs=slugs, window=window)
    if not pool:
        return {"candidates": []}

    chip = chip_of(area) if area else None
    anchor = (chip.lat, chip.lng) if chip else (pool[0][0]["lat"], pool[0][0]["lng"])

    want = chips.days * chips.stops_per_day() + 6
    ranked = _dedupe_same_place(_geo_rerank(pool, anchor, want, weights=weights))
    return {"candidates": _quota_pick(ranked, want, seed)}


async def _llm_call(tag: str, chain, prompt_vars: dict) -> str:
    """AI 를 한 번 부르고 답을 텍스트로 받는다. 얼마나 걸렸고 몇 자를 썼는지 로그에 남긴다.

    선정·글쓰기는 여러 번 나눠 동시에 부르기 때문에, 이 로그가 없으면
    어느 콜이 오래 걸리는지 알 수 없다.
    """
    t0 = time.perf_counter()
    text = extract_text((await chain.ainvoke(prompt_vars)).content)
    logger.info("%s %.1fs · 출력 %d자", tag, time.perf_counter() - t0, len(text))
    return text


async def _select_days(prompt_vars: dict, by_name: dict[str, dict]) -> list[dict]:
    """AI 에게 장소를 고르게 하고, 날짜별 목록으로 받는다. 실패해도 예외를 밖으로 안 던진다.

    답이 길어 JSON 이 중간에 잘려도, 온전한 부분만 건져서 쓴다.
    여러 콜을 동시에 던지므로 하나가 실패해도 나머지가 죽지 않게 여기서 막는다.
    """
    raw = ""
    try:
        raw = await _llm_call("select 콜", _SELECT_PROMPT | get_llm(), prompt_vars)
        data = parse_json_object(raw)
        # 날짜 구분 없이 stops 만 온 경우도 1일차로 받아준다
        return data.get("days") or [{"day": 1, "stops": data.get("stops", [])}]
    except Exception as err:  # noqa: BLE001
        salvaged = [o for o in salvage_objects(raw) if o.get("name")]
        logger.warning("select LLM JSON 파싱 실패 — 부분 복구 %d곳: %s", len(salvaged), err)
        # 건져낸 장소를 후보의 day_hint 로 날짜별로 다시 묶는다. 안 그러면 2·3일차 몫이 잘려나간다
        by_day: dict[int, list] = {}
        for o in salvaged:
            cand = by_name.get(o.get("name"))
            d = (cand.get("day_hint") if cand else None) or 1
            by_day.setdefault(d, []).append(o)
        return [{"day": d, "stops": s} for d, s in sorted(by_day.items())]


async def select_places_node(state: AgentState) -> dict:
    """AI 가 후보 중에서 갈 곳을 고르는 단계. 고른 이유와 거기서 할 일도 같이 받는다.

    날마다 동네가 다른 코스는 날짜별로 나눠서 동시에 물어본다.
    """
    req = state.get("req", {})
    chips = _chips(state)
    cands = state.get("candidates", [])
    by_name = {c["name"]: c for c in cands}

    # 후보 한 곳을 AI 에게 보여줄 한 덩어리 텍스트로 만든다.
    # 검색 본문을 그대로 쓰지 않고 필요한 값만 골라 쓰므로, 이 모양을 바꿔도 재인제스트가 필요 없다.
    def _cand_line(c: dict) -> str:
        bits = [f"종류={c.get('category') or '기타'}", f"운영={_hours_label(c)}"]
        if c.get("dist_km") is not None:
            bits.append(f"거리={c['dist_km']}km")
        # AI 가 동반자·목적에 맞는 곳을 고를 때 참고할 특징들. 태그가 없으면 안 적는다
        if c.get("indoor") is not None:
            bits.append("실내" if c["indoor"] else "야외")
        if c.get("energy"):
            bits.append("조용" if c["energy"] == "calm" else "활기")
        if c.get("night_ok"):
            bits.append("밤가능")
        if c.get("stay_min"):
            bits.append(f"보통{c['stay_min']}분")
        if c.get("is_filming") and c.get("content_title"):
            bits.append(f"K-콘텐츠 촬영지「{c['content_title']}」")
        head = f"- {c['name']} | " + " | ".join(bits)
        # "있는 것"은 할 일을 쓸 때의 재료라, 소개문보다 앞에 둔다
        body = c.get("description") or c.get("text", "")[:240]
        if hl := c.get("highlights"):
            return f"{head}\n  있는 것: {', '.join(hl)}\n  소개: {body}"
        return f"{head}\n  소개: {body}"

    day_areas = state.get("day_areas") or {}
    pr = chips.purpose_rule()
    rule = pr.get("rule", "")
    act = pr.get("act", "")
    comp_rule = chips.companion_rule()
    per_day = chips.stops_per_day()

    # 하루가 어떤 구간으로 나뉘고 각 구간에 몇 곳이 들어가는지를 AI 에게 알려준다.
    # 이걸 줘야 "점심 먹고 바로 가는 곳"에 맞는 할 일이 나오고, 구간 길이에 맞는 체류시간을 제안한다.
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

    # 날마다 동네가 다르면 후보도 날짜별로 이미 갈라져 있다. 그래서 날짜별로 따로 물어봐도
    # 같은 장소가 겹칠 수 없고, 동시에 물어보면 그만큼 빨라진다.
    # 답이 짧아지니 길이 제한에 걸려 결과가 통째로 날아가는 일도 없어진다.
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
        # 콜 하나가 하루치다. AI 는 늘 1일차로 답하므로 며칠째인지는 여기서 붙인다
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
            if name in by_name and name not in seen:  # 후보에 없는 이름은 버린다(AI 가 지어낸 것)
                seen.add(name)
                cand = by_name[name]
                # AI 가 날짜를 섞어 답해도, 후보에 붙여둔 day_hint 가 있으면 그쪽을 믿는다
                day = cand.get("day_hint") or day_no
                picked.append(
                    {
                        **cand,
                        "day": day if chips.days > 1 else None,
                        "reason": row.get("reason", ""),
                        "activities": [a for a in row.get("activities", []) if a][:3],
                        # AI 가 제안한 체류시간. 나중에 시간표를 짜면서 다시 조정한다.
                        # 값이 이상하면 None 이 되고, 그때는 장소 종류로 대신 정한다.
                        "dwell_min": _clamp_dwell(row.get("dwell_min")),
                    }
                )

    if len(picked) < 2:  # AI 선정이 사실상 실패했으면, 후보 상위를 날짜별로 그냥 나눠 담는다
        fallback = [
            {**c, "day": (c.get("day_hint") or i // per_day + 1) if chips.days > 1 else None,
             "reason": "", "activities": [], "dwell_min": None}
            for i, c in enumerate(cands[: per_day * chips.days])
        ]
        return {"selected": fallback, "source": "mock"}
    return {"selected": _enforce_variety(picked, cands), "source": "ai"}


# 하루에 같은 종류가 이 개수를 넘으면 다른 종류로 바꾼다. 프롬프트에 적은 규칙과 같은 값이다
_MAX_SAME_CATEGORY = 2


def _enforce_variety(picked: list[dict], cands: list[dict]) -> list[dict]:
    """하루에 비슷한 곳만 몰렸으면 마지막에 서버가 바로잡는다.

    프롬프트로도 같은 규칙을 주지만 AI 가 어길 때가 있어서, 여기서 한 번 더 확인한다.
    계산만 하므로 AI 를 다시 부르지 않고 시간도 안 든다.

    넘치는 장소는 그날 후보 중 아직 안 쓴 다른 종류로 바꾼다. 바꿀 후보가 없으면 그냥 둔다.
    장소를 빼서 개수를 줄이지는 않는다. 사용자가 고른 장소 수를 어기는 셈이기 때문이다.
    바뀐 장소는 이유·할 일이 원래 장소 것이라 비워둔다. 뒤에서 다시 쓴다.
    """
    used = {p["name"] for p in picked}
    out: list[dict] = []
    counts: dict[tuple, int] = {}

    for p in picked:
        key = (p.get("day"), p.get("category") or "기타")
        if counts.get(key, 0) < _MAX_SAME_CATEGORY:
            counts[key] = counts.get(key, 0) + 1
            out.append(p)
            continue
        swap = next(
            (c for c in cands
             if c["name"] not in used
             and (c.get("category") or "기타") != key[1]
             and counts.get((p.get("day"), c.get("category") or "기타"), 0) < _MAX_SAME_CATEGORY
             and (p.get("day") is None or (c.get("day_hint") or p["day"]) == p["day"])),
            None,
        )
        if not swap:
            counts[key] = counts.get(key, 0) + 1
            out.append(p)
            continue
        logger.info("다양성 교정: %s(%s) → %s(%s)", p["name"], key[1],
                    swap["name"], swap.get("category"))
        used.add(swap["name"])
        new_key = (p.get("day"), swap.get("category") or "기타")
        counts[new_key] = counts.get(new_key, 0) + 1
        out.append({**swap, "day": p.get("day"), "reason": "", "activities": [],
                    "dwell_min": None})
    return out


def _clamp_dwell(raw) -> int | None:
    """AI 가 제안한 체류시간을 상식적인 범위로 자른다. 숫자가 아니면 None 을 준다."""
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        return None
    return max(DWELL_FLOOR, min(DWELL_CEIL, v)) if v > 0 else None


async def enrich_node(state: AgentState) -> dict:
    """각 장소에 "방문할 시각의" 예상 혼잡도를 붙인다. 지금 이 순간의 혼잡도가 아니다.

    시간표를 짠 뒤에 오는 이유는, 방문 시각을 알아야 그 시간대 예보를 볼 수 있어서다.
    혼잡하다고 장소를 바꾸지는 않는다. 보여주기만 한다.
    혼잡도 측정 지점이 있는 장소에만 값이 붙는다.
    """
    selected = state.get("selected", [])
    if not selected:
        return {"congestion": {}}

    # 방문 시각은 시간표에서 가져온다. 시간표가 없는 코스는 None 이고, 그때는 현재 혼잡도를 쓴다.
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
    """식당 목록에서 이 장소 반경 안에 있는 곳을 가까운 순으로 n개 고른다."""
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
    """이 장소에서 열리는 행사 n개를 고른다. 좌표가 있으면 거리도 붙인다.

    행사는 이미 그 장소 소속으로 받아온 것이라 거리로 다시 거르지 않는다.
    """
    out: list[dict] = []
    for e in events[:n]:
        dist = (round(haversine_km(stop["lat"], stop["lng"], e["lat"], e["lng"]), 2)
                if e.get("lat") is not None and e.get("lng") is not None else None)
        out.append({**e, "dist_km": dist})
    return out


async def _live_meal_pool(selected: list[dict], chips: CourseChips) -> list[dict]:
    """캐시에 식당이 없을 때, Visit Seoul 에서 바로 받아온다.

    이 API 는 키워드 검색이라 장소 이름이 가장 좋은 단서다("경복궁"으로 찾으면 서촌 카페가 나온다).
    장소마다 따로 부르면 호출 제한에 걸리니, 이름들을 한 번에 넘겨 한 번만 부른다.
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

    # 시간표가 있는 코스는 끼니마다 식당 후보가 여럿 필요해서 더 많이 받아온다(1건에 0.75초쯤 든다)
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
    """각 장소 주변의 식당과 행사를 붙인다.

    여기서 붙이는 식당은 "이 근처에 뭐가 있나" 보여주는 카드용이다.
    실제 식사 슬롯에 들어갈 식당은 meals 노드가 따로 고른다.
    """
    selected = state.get("selected", [])
    if not selected:
        return {"nearby": {}}
    chips = _chips(state)

    # 식당은 미리 구워둔 캐시를 먼저 본다. 캐시가 부족하면 실시간 조회로 채운다
    pool = state.get("meal_pool") or pool_for_stops(selected)
    if len(pool) < get_settings().meal_pool_min:
        seen = {r["title"] for r in pool}
        pool += [r for r in await _live_meal_pool(selected, chips) if r["title"] not in seen]

    # 행사는 서울시 실시간 API 에서 받는다. 같은 명소는 한 번만 조회한다
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
    """고른 장소들을 날짜별로 묶는다. 하루 코스는 전부 1일차가 된다."""
    out: dict[int, list[dict]] = {}
    for s in selected:
        out.setdefault(s.get("day") or 1, []).append(s)
    return out


def _stop_chunks(selected: list[dict]) -> list[list[dict]]:
    """장소들을 AI 한 번에 맡길 묶음으로 나눈다. 날짜를 넘지 않게, 크기는 고르게 나눈다.

    한 묶음은 같은 날이어야 "그날 흐름이 이어지게 쓰라"는 지시가 성립한다.

    크기를 고르게 나누는 것도 중요하다. 동시에 돌리니 전체 시간은 제일 긴 묶음이 정하는데,
    앞에서부터 3곳씩 자르면 4곳이 3+1 로 갈려 3곳짜리 묶음이 혼자 오래 걸린다. 2+2 면 둘 다 짧다.
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
    """AI 가 답한 장소 이름을 실제 장소 이름과 맞춰 본다. 못 맞추면 None.

    AI 가 보여준 그대로 "장소명 (분류)" 형태로 답할 때가 있어서, 괄호를 떼고도 맞춰 본다.
    못 맞추면 그 장소 카드가 빈 채로 나간다.
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
    """확정된 장소들에 글을 입히는 마지막 단계. 코스 제목·소개와 장소별 카드 문구를 쓴다.

    여기서 장소를 더하거나 빼지는 않는다. 여러 콜을 동시에 던져 받은 글을 합쳐서
    프론트에 보낼 최종 형태로 조립한다.
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

    # 코스 전체 소개를 쓸 때는 시각도 며칠째인지도 주지 않는다. 소개는 "왜 이렇게 골랐는지"를
    # 쓰는 자리인데, 이 값들을 주면 프롬프트로 아무리 막아도 그대로 옮겨 적는다.
    # 날짜 구성 같은 건 칩만 봐도 알 수 있어서 근거를 쓰는 데 부족하지 않다.
    global_stop_lines = "\n".join(
        f"- {s['name']} ({s.get('category','')})"
        f"{' · 선정 이유: ' + s['reason'] if s.get('reason') else ''}"
        for s in selected
    )

    narrative: dict[str, dict] = {}
    title = subtitle = description = ""
    tags: list[str] = []
    day_descriptions: dict[int, str] = {}
    source = state.get("source", "ai")
    # 여러 날 코스는 전체 소개를 짧게 두고 날짜별 설명을 따로 쓴다(프론트가 날짜 탭으로 보여준다).
    # 하루 코스는 소개 하나에 다 담는다.
    day_desc_rule = (
        f"- 이 코스는 {chips.days}일 일정이다. **description 에 날짜별 서술을 절대 넣지 말 것** — "
        "'1일차는 A를 보고, 2일차는 B로 간다' 같은 문장은 금지다(일차별 요약은 다른 곳에서 쓴다). "
        "날짜 얘기는 '하루 N곳씩 권역별로 나눴어요'처럼 **나눈 기준 한 문장까지만** 허용하고, "
        "그 문장에도 특정 일차에 어느 장소가 들어가는지는 쓰지 않는다.\n"
        if chips.days > 1
        else "- 하루 코스다.\n"
    )

    comp_rule = chips.companion_rule()
    act = chips.purpose_rule().get("act", "")
    lens = {
        "note": req.get("note", "서울 코스"),
        "chips": chips.summary(),
        "companion_lens": f"- 동반 렌즈(어조·관계): {comp_rule}\n" if comp_rule else "",
        "purpose_lens": f"- 목적별 행동 결: {act}\n" if act else "",
    }

    # 전체 소개 1개 + 날짜별 요약 + 장소 카드 묶음을 한꺼번에 동시에 요청한다.
    # 서로 참고할 필요가 없어서 나눠도 되고, 전체 시간은 제일 느린 콜 하나가 정한다.
    # 그래서 콜마다 쓸 양을 고르게 나누는 게 중요하다.
    #
    # 하나가 실패해도 그 몫만 잃는다. 전체 소개가 실패하면 제목·소개만 임시 문구가 되고,
    # 날짜 요약이 실패하면 그 날 요약만 비고, 장소 묶음이 실패하면 그 장소 카드만 빈다.
    day_stops = _by_day(selected)
    day_keys = sorted(day_stops) if chips.days > 1 else []
    chunks = _stop_chunks(selected)
    results = await asyncio.gather(
        _llm_call("compose 전역", _COMPOSE_GLOBAL_PROMPT | get_llm(),
                  {**lens, "chips": chips.summary(for_description=True),
                   "stops": global_stop_lines, "day_desc_rule": day_desc_rule,
                   "n_places": len(selected)}),
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
            # JSON 이 잘렸어도 온전한 부분은 건진다. 안 그러면 이 묶음 카드가 통째로 빈다
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
            # 방문 시각을 알면 "18시 예상 혼잡도 …"로, 모르면 시각 없이 쓴다
            hh = slot["start_time"][:2].lstrip("0") if slot and slot.get("start_time") else ""
            note = f"{hh}시 예상 혼잡도 {lvl}" if hh else f"예상 혼잡도 {lvl}"
            tip = f"{tip} · {note}" if tip else note
        # 체류시간은 서버가 계산한다. 시간표가 있으면 그 값을, 없으면 선정 단계에서 받은 값을 쓴다.
        # 글 쓰는 AI 에게는 시간을 묻지 않는다. 어차피 안 쓸 값이라 답만 길어진다.
        dur_min = slot["duration_min"] if slot else s.get("dwell_min")
        duration = duration_label(dur_min) if dur_min else "1시간"
        return {
            "name": s["name"],
            "preview": n.get("preview", s.get("text", "")[:40]),
            "description": n.get("description", ""),
            "duration": duration,
            # 위 duration 은 "1시간 30분" 같은 표시용 문자열이라, 분 단위 숫자도 같이 보낸다
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
            # 값이 없어도 키는 넣는다. 장소 슬롯과 모양을 맞춰야 프론트가 헷갈리지 않는다
            "travel_min": None,
            "travel_mode": None,
        }

    by_name = {s["name"]: s for s in selected}
    if schedule:
        # 시간표 순서를 그대로 따른다. 장소 사이사이에 식사가 끼어 있다
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
        "description": description or _fallback_description(
            chips, req.get("note", ""), len(selected)
        ),
        "stops": stops,
        "tags": tags or ["코스"],
        "scheduled": bool(sched_by_name),
        "days": chips.days,
        "day_areas": state.get("day_areas") or {},
        "day_descriptions": day_descriptions,  # 날짜별 설명. 하루 코스면 비어 있다
    }
    return {"result": {"course": course, "source": source}, "source": source}
