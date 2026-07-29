"""LLM 텍스트 출력에서 JSON을 안전하게 파싱하는 코드
"""
from __future__ import annotations

import json
import re
from typing import Any


def _strip_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = _strip_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    obj = _extract_balanced(candidate, "{", "}")
    if obj is None:
        raise ValueError(f"JSON 객체를 찾지 못함: {text[:200]!r}")
    return json.loads(obj)


def parse_json_array(text: str) -> list[Any]:
    candidate = _strip_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    arr = _extract_balanced(candidate, "[", "]")
    if arr is None:
        raise ValueError(f"JSON 배열을 찾지 못함: {text[:200]!r}")
    return json.loads(arr)


def salvage_objects(text: str) -> list[dict[str, Any]]:
    """잘린(truncated) JSON 에서 **완결된 `{...}` 객체만** 최대한 건져낸다.

    LLM 이 토큰 한도로 응답을 중간에 끊으면 전체 JSON 은 깨지지만, 그 앞에 이미
    완성된 객체(예: stop 하나)는 온전하다. 중첩된 미완결 객체는 건너뛰고 balanced 한
    조각만 순차로 파싱해, 마지막 잘린 객체 하나만 잃고 나머지는 이유·행동까지 보존한다.
    """
    text = _strip_fences(text)
    out: list[dict[str, Any]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            obj = _extract_balanced(text[i:], "{", "}")
            if obj is not None:
                try:
                    parsed = json.loads(obj)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    out.append(parsed)
                    i += len(obj)
                    continue
        i += 1
    return out
