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
