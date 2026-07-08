
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.core.json_parse import parse_json_object
from app.core.llm import extract_text, get_llm
from app.graph.state import AgentState

_INTENTS = ("place_intro", "recommend", "course", "chitchat")

_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 서울 여행 에이전트의 라우터다. 사용자 메시지를 아래 4개 의도 중 하나로 분류해 "
            'JSON 으로만 답하라: {{"intent":"..."}}\n'
            "- course: 여러 곳을 잇는 코스/동선/일정/데이트 코스 요청\n"
            "- recommend: 상황(동행·시간·목적)에 맞는 장소 추천\n"
            "- place_intro: 특정 한 장소의 소개/정보\n"
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
            '"time":"오전|오후|저녁|밤|상관없음"}}\n'
            "- region/time 을 특정할 수 없으면 '상관없음'.\n"
            "- note 는 사용자의 의도를 담은 한 문장.",
        ),
        ("human", "{message}"),
    ]
)


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
    if intent == "place_intro":
        # 장소명만 필요 — 원문을 place 로 넘기고 좌표는 downstream RAG 로 보완
        return {"req": {"place": message}}
    if intent == "recommend":
        # 최소 요청 — note 를 purpose 로 사용(세부 추출은 2단계 고도화)
        return {"req": {"purpose": message, "region": "상관없음", "congestion": "상관없음"}}

    # course: 구조화 추출
    try:
        msg = await (_PARSE_PROMPT | get_llm()).ainvoke({"message": message})
        data = parse_json_object(extract_text(msg.content))
        req = {
            "note": data.get("note") or message,
            "region": data.get("region") or "상관없음",
            "time": data.get("time") or "상관없음",
        }
    except Exception:  # noqa: BLE001
        req = {"note": message, "region": "상관없음", "time": "상관없음"}
    return {"req": req}


def dispatch_intent(state: AgentState) -> str:
    """parse_intent 이후 의도별 분기 키."""
    return state.get("intent", "chitchat")
