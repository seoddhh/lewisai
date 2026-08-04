"""후보 다양성·개인화 재랭킹 — 순수 함수 회귀 테스트.

이 로직들은 LLM 없이 도는 계산이라 테스트로 고정할 수 있다. 그리고 고정해야 한다 —
"같은 종류만 3곳" 이나 "재생성해도 같은 코스" 는 **에러를 내지 않고 조용히** 나빠지는
종류의 회귀다(예전에 프롬프트 규칙만 있고 검증이 없어 실제로 그랬다).
"""
from __future__ import annotations

from app.features.course.schema import CourseChips
from app.graph.nodes.course import (
    _enforce_variety,
    _personal_bonus,
    _pick_seed,
    _quota_pick,
)


def _c(name: str, category: str, **kw) -> dict:
    return {"name": name, "category": category, **kw}


# ── 종류 쿼터 ────────────────────────────────────────────────────────────
def test_quota_pick_breaks_category_monopoly():
    """편중된 후보에서도 상위권에 다른 종류가 섞인다."""
    ranked = [_c(f"쇼핑{i}", "쇼핑") for i in range(10)] + [
        _c("공원", "자연"), _c("박물관", "문화")]
    picked = _quota_pick(ranked, 3, seed=1)
    assert len({p["category"] for p in picked}) >= 2, "상위 3곳이 전부 같은 종류다"


def test_quota_pick_returns_requested_count():
    """다양성을 맞추려고 개수를 줄이지 않는다 (개수가 먼저다)."""
    ranked = [_c(f"쇼핑{i}", "쇼핑") for i in range(8)]
    assert len(_quota_pick(ranked, 5, seed=1)) == 5


def test_quota_pick_seed_changes_selection():
    """시드가 다르면 다른 코스가 나온다 — 재생성 버튼이 의미를 가지려면 필수."""
    ranked = [_c(f"문화{i}", "문화") for i in range(6)] + [
        _c(f"자연{i}", "자연") for i in range(6)]
    a = {p["name"] for p in _quota_pick(ranked, 4, seed=1)}
    b = {p["name"] for p in _quota_pick(ranked, 4, seed=999)}
    assert a != b, "시드를 바꿔도 결과가 같다 — 파이프라인이 완전히 결정적이다"


def test_quota_pick_same_seed_is_stable():
    """같은 시드는 항상 같은 결과 — A/B·디버깅이 가능해야 한다."""
    ranked = [_c(f"p{i}", "문화" if i % 2 else "자연") for i in range(10)]
    assert _quota_pick(ranked, 4, seed=7) == _quota_pick(ranked, 4, seed=7)


# ── 개인화 가점 ──────────────────────────────────────────────────────────
def test_child_companion_prefers_indoor_short():
    """아이와 → 실내·짧은 체류가 위로, 오래 붙잡는 곳은 감점."""
    w = CourseChips(companions=["아이와"]).rerank_weights()
    indoor_short = _personal_bonus({"indoor": True, "stay_min": 45}, w)
    outdoor_long = _personal_bonus({"indoor": False, "stay_min": 180}, w)
    assert indoor_short > outdoor_long


def test_untagged_place_is_neutral():
    """미태깅 장소는 0점 — 벌점이 아니다. 태깅이 절반만 끝나도 코스가 나와야 한다."""
    w = CourseChips(companions=["아이와"], purposes=["문화·예술"]).rerank_weights()
    assert _personal_bonus({}, w) == 0.0


def test_bonus_is_bounded():
    """가점이 지리 근접을 뒤집지 못하게 ±0.2 로 잘린다."""
    w = CourseChips(companions=["아이와", "부모님과"],
                    purposes=["문화·예술", "쇼핑", "핫플레이스"]).rerank_weights()
    for c in ({"indoor": True, "night_ok": True, "energy": "calm", "stay_min": 30},
              {"indoor": False, "night_ok": False, "energy": "lively", "stay_min": 240}):
        assert -0.2 <= _personal_bonus(c, w) <= 0.2


def test_purpose_changes_ranking_direction():
    """목적이 다르면 같은 장소의 가점 부호가 갈린다 (목적이 정렬에 반영된다)."""
    nature = CourseChips(purposes=["자연·힐링"]).rerank_weights()
    culture = CourseChips(purposes=["문화·예술"]).rerank_weights()
    indoor_calm = {"indoor": True, "energy": "calm"}
    assert _personal_bonus(indoor_calm, culture) > _personal_bonus(indoor_calm, nature)


# ── 목적 필터 slug ───────────────────────────────────────────────────────
def test_date_purpose_is_not_a_filter():
    """"데이트"는 메타필터로 안 나간다 — 장소 태그에서 뺐으므로 0건 매칭이 된다."""
    assert CourseChips(purposes=["데이트"]).purpose_slugs() == []
    assert CourseChips(purposes=["데이트", "문화·예술"]).purpose_slugs() == ["culture"]


# ── 다양성 후처리 ────────────────────────────────────────────────────────
def test_enforce_variety_swaps_third_same_category():
    """같은 종류 3곳째는 후보의 다른 종류로 교체된다."""
    picked = [_c("쇼핑A", "쇼핑"), _c("쇼핑B", "쇼핑"), _c("쇼핑C", "쇼핑")]
    cands = [*picked, _c("공원", "자연")]
    out = _enforce_variety(picked, cands)
    assert [p["name"] for p in out] == ["쇼핑A", "쇼핑B", "공원"]


def test_enforce_variety_keeps_count_when_no_swap():
    """바꿀 후보가 없으면 그대로 둔다 — 개수를 줄여 다양성을 맞추지 않는다."""
    picked = [_c(f"쇼핑{i}", "쇼핑") for i in range(3)]
    out = _enforce_variety(picked, picked)
    assert len(out) == 3


def test_enforce_variety_clears_stale_reason():
    """교체된 장소는 이전 장소의 선정 이유·행동을 물려받지 않는다."""
    picked = [_c("쇼핑A", "쇼핑"), _c("쇼핑B", "쇼핑"),
              _c("쇼핑C", "쇼핑", reason="쇼핑C라서", activities=["쇼핑하기"])]
    out = _enforce_variety(picked, [*picked, _c("공원", "자연")])
    assert out[2]["name"] == "공원"
    assert out[2]["reason"] == "" and out[2]["activities"] == []


def test_enforce_variety_counts_per_day():
    """쏠림은 하루 단위로 센다 — 2일차 쇼핑이 1일차 때문에 밀려나면 안 된다."""
    picked = [_c("쇼핑A", "쇼핑", day=1), _c("쇼핑B", "쇼핑", day=1),
              _c("쇼핑C", "쇼핑", day=2), _c("쇼핑D", "쇼핑", day=2)]
    out = _enforce_variety(picked, picked)
    assert [p["name"] for p in out] == ["쇼핑A", "쇼핑B", "쇼핑C", "쇼핑D"]


# ── 시드 ─────────────────────────────────────────────────────────────────
def test_seed_is_derived_from_request():
    """같은 요청 = 같은 시드, 다른 칩 = 다른 시드 (재현 가능해야 A/B 를 한다)."""
    req = {"chips": {"purposes": ["문화·예술"]}, "note": ""}
    assert _pick_seed(req) == _pick_seed({"chips": {"purposes": ["문화·예술"]}, "note": ""})
    assert _pick_seed(req) != _pick_seed({"chips": {"purposes": ["쇼핑"]}, "note": ""})


def test_explicit_seed_wins():
    """프론트가 보낸 seed("다시 만들기")가 있으면 그걸 쓴다."""
    assert _pick_seed({"chips": {}, "note": "", "seed": 42}) == 42
