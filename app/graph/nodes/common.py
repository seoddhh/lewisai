
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.core.json_parse import parse_json_object
from app.core.llm import extract_text, get_llm
from app.features.course.schema import COMPANIONS, PURPOSE_RULES
from app.graph.state import AgentState

_INTENTS = ("course", "chitchat")

_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 서울 여행 코스 에이전트의 라우터다. 사용자 메시지를 아래 2개 의도 중 하나로 "
            '분류해 JSON 으로만 답하라: {{"intent":"..."}}\n'
            "- course: 장소·코스·동선·일정·데이트 코스·상황(동행·시간·목적) 맞춤 장소 추천 등 "
            "'서울에서 어디를 갈지' 묻는 모든 요청\n"
            "- chitchat: 그 외 일반 대화",
        ),
        ("human", "{message}"),
    ]
)

_PARSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 자연어 요청에서 코스 생성에 필요한 정보를 추출해 JSON 으로만 답하라.\n"
            'form: {{"note":"원문 요지","region":"강북|강남|강서|강동|상관없음",'
            '"time":"오전|오후|저녁|밤|상관없음",'
            '"time_start":숫자|null,"time_end":숫자|null,'
            '"audience":"local|tourist"|null,"days":숫자|null,'
            '"companion":"혼자|친구와|연인과|배우자와|아이와|부모님과"|null,'
            '"purpose":"힐링|놀거리|데이트|관광|문화생활|체험·액티비티|핫플레이스|자연 힐링|'
            '유명 관광지|문화·예술·역사|쇼핑|맛집 탐방"|null,'
            '"pace":"packed|relaxed"|null}}\n'
            "- region/time 을 특정할 수 없으면 '상관없음', 나머지는 특정할 수 없으면 null.\n"
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
    return chips


async def router_node(state: AgentState) -> dict:
    if state.get("intent") in _INTENTS:
        return {}
    message = state.get("message", "")
    try:
        msg = await (_ROUTER_PROMPT | get_llm()).ainvoke({"message": message})
        intent = parse_json_object(extract_text(msg.content)).get("intent", "")
    except Exception:  # noqa: BLE001
        intent = ""
    if intent not in _INTENTS:
        intent = "chitchat"
    return {"intent": intent}


async def parse_intent_node(state: AgentState) -> dict:
    # 어댑터 경로: req 가 이미 주입됨 → 그대로 사용
    if state.get("req"):
        return {}
    message = state.get("message", "")
    intent = state.get("intent", "chitchat")

    if intent == "chitchat":
        return {"req": {"message": message}}

    # course: 구조화 추출 — 시간/동반/목적은 칩과 같은 자리(chips)로 합쳐서
    # 이후 노드(retrieve/select/schedule)가 칩 경로와 동일한 코드로 처리한다.
    try:
        msg = await (_PARSE_PROMPT | get_llm()).ainvoke({"message": message})
        data = parse_json_object(extract_text(msg.content))
        req = {
            "note": data.get("note") or message,
            "region": data.get("region") or "상관없음",
            "time": data.get("time") or "상관없음",
            "chips": _synth_chips(data),
        }
    except Exception:  # noqa: BLE001
        req = {"note": message, "region": "상관없음", "time": "상관없음", "chips": {}}
    return {"req": req}


def dispatch_intent(state: AgentState) -> str:
    """parse_intent 이후 의도별 분기 키."""
    return state.get("intent", "chitchat")
