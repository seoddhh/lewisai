
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.core.json_parse import parse_json_object
from app.core.llm import extract_text, get_llm
from app.features.course.schema import COMPANIONS, MEALS, PURPOSE_RULES
from app.graph.state import AgentState

# 의도 분류(router)는 제거했다 — 이 에이전트는 코스만 만들고, 칩 경로에서는 라우터가
# 어차피 무동작이었다. 잡담이 필요하면 /agent/chitchat 라우트가 별도로 살아 있다.

_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 자연어 요청에서 코스 생성에 필요한 정보를 추출해 JSON 으로만 답하라.\n"
            'form: {{"note":"원문 요지",'
            '"time":"오전|오후|저녁|밤|상관없음",'
            '"time_start":숫자|null,"time_end":숫자|null,'
            '"audience":"local|tourist"|null,"days":숫자|null,'
            '"companion":"혼자|친구와|연인과|배우자와|아이와|부모님과"|null,'
            '"purpose":"자연·힐링|문화·예술|관광 명소|체험·놀거리|'
            '데이트|핫플레이스|쇼핑"|null,'
            '"meals":["아침","점심","저녁"] 중 언급된 것만 (없으면 []),'
            '"pace":"packed|relaxed"|null}}\n'
            "- time 을 특정할 수 없으면 '상관없음', 나머지는 특정할 수 없으면 null.\n"
            '- meals: 식사 의도가 드러날 때만 넣는다. "저녁 먹고 놀 데" → ["저녁"], '
            '"점심부터 저녁까지" → ["점심","저녁"], "브런치" → ["아침"]. '
            "단순히 시간대만 말한 경우(예: '저녁에 갈 만한 곳')는 식사 의도가 아니므로 [] 로 둔다.\n"
            "- time_start/time_end 는 구체적 시각이 언급될 때만 0~28 정수(시)로. "
            '예: "오후 2시부터 8시까지" → 14, 20 / "저녁에" → 18, 22 / "밤 10시부터 새벽 2시" → 22, 26.\n'
            '- audience: 서울에 사는 사람의 일상 나들이면 "local", 서울로 여행 온 맥락이면 "tourist".\n'
            '- days: 여행 일수 — "당일치기" → 1, "1박2일" → 2, "2박3일" → 3 처럼 박+1.\n'
            '- companion/purpose 는 문맥에서 추론 가능할 때만. 예: "부모님 모시고" → companion "부모님과".\n'
            '- pace: "빼곡하게/알차게" → "packed", "널널하게/여유롭게" → "relaxed".\n'
            "- note 는 사용자의 의도를 담은 한 문장.",
        ),
        ("human", "{message}"),
    ]
)

# 파싱된 시간대 표현 → 시간 범위(시). 칩(schema._TIME_SLOT_WINDOWS)에 없는 '저녁'도 처리.
_TIME_WORD_WINDOWS: dict[str, tuple[int, int]] = {
    "오전": (9, 12), "오후": (12, 18), "저녁": (18, 22), "밤": (18, 23),
}


def _synth_chips(data: dict) -> dict:
    """자연어 추출 결과 → 칩과 같은 형태. 이후 파이프라인이 칩/자연어 구분 없이 한 갈래로 돈다."""
    chips: dict = {}
    ts, te = data.get("time_start"), data.get("time_end")
    if isinstance(ts, int) and isinstance(te, int) and 0 <= ts <= 24 and ts < te <= 28:
        chips["time_window"] = {"start": ts, "end": te}
    elif data.get("time") in _TIME_WORD_WINDOWS:
        start, end = _TIME_WORD_WINDOWS[data["time"]]
        chips["time_window"] = {"start": start, "end": end}
    # 허용값만 통과 — 어긋난 값 하나가 칩 전체 파싱 실패(_chips 폴백)로 번지지 않게
    allowed = {
        "companion": set(COMPANIONS),
        "purpose": set(PURPOSE_RULES),
        "audience": {"local", "tourist"},
        "pace": {"packed", "relaxed"},
    }
    for key, ok in allowed.items():
        if data.get(key) in ok:
            if key in ("companion", "purpose"):
                chips[key + "s"] = [data[key]]  # 자연어에선 하나만 추출 — 칩 스키마는 리스트
            else:
                chips[key] = data[key]
    if isinstance(data.get("days"), int) and 1 <= data["days"] <= 6:
        chips["days"] = data["days"]
    # 끼니 — 자연어에 식사 의도가 있을 때만. 칩 경로에선 사용자가 직접 고른다.
    meals = [m for m in (data.get("meals") or []) if m in MEALS]
    if meals:
        chips["meals"] = meals
    return chips


async def parse_intent_node(state: AgentState) -> dict:
    """자연어 → 칩 형태로 구조화. 칩 경로(req 주입)면 아무것도 하지 않는다.

    이 에이전트는 코스만 만들므로 의도 분류(router)는 없앴다 — 칩 경로에서는
    어차피 무동작이었고, 자연어도 전부 코스 요청으로 다룬다.
    """
    # 어댑터 경로: req 가 이미 주입됨 → 그대로 사용
    if state.get("req"):
        return {}
    message = state.get("message", "")

    # 구조화 추출 — 시간/동반/목적/끼니는 칩과 같은 자리(chips)로 합쳐서
    # 이후 노드(plan/retrieve/select/fit_schedule)가 칩 경로와 동일한 코드로 처리한다.
    try:
        msg = await (_PARSE_PROMPT | get_llm()).ainvoke({"message": message})
        data = parse_json_object(extract_text(msg.content))
        req = {
            "note": data.get("note") or message,
            "chips": _synth_chips(data),
        }
    except Exception:  # noqa: BLE001
        req = {"note": message, "chips": {}}
    return {"req": req}
