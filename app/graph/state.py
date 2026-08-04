"""노드들이 주고받는 공유 상태. 각 노드가 여기에 자기 결과를 채워 넣는다."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ── 들어올 때 ── 둘 중 하나로 들어온다
    message: str            # 사용자가 쓴 문장
    req: dict[str, Any]     # 칩으로 들어온 요청. 없으면 parse_intent 가 문장을 보고 만든다

    # ── 만들어지는 중간 결과 ──
    # 칩은 여기 담아 두지 않는다. 필요한 노드가 그때그때 req["chips"] 를 읽는다.
    skeleton: dict[str, Any]    # 하루의 시간 뼈대(식사 시각 + 장소가 들어갈 구간). plan 이 만든다
    candidates: list[dict]      # 검색으로 찾은 후보 장소들
    day_areas: dict[int, str]   # 며칠째에 어느 동네인지. 날짜별로 동네를 나눴을 때만 채워진다
    selected: list[dict]        # AI 가 고른 장소들. 고른 이유와 할 일이 같이 들어 있다
    congestion: dict[str, str]  # 장소 이름 → 방문 시각의 예상 혼잡도
    nearby: dict[str, dict]     # 장소 이름 → 주변 식당·행사
    meal_pool: list[dict]       # 식사 슬롯에 넣을 식당 후보들
    schedule: list[dict]        # 시간표. 시간 범위가 정해진 요청에서만 채워진다

    # ── 나갈 때 ──
    result: dict[str, Any]  # 프론트에 보낼 최종 응답
    source: str             # ai | mock — AI 가 만든 것인지 임시 데이터인지
