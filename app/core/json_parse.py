"""AI 가 답한 텍스트에서 JSON 을 꺼낸다.

"JSON 으로만 답하라"고 해도 앞뒤에 설명을 붙이거나 ```json 으로 감싸서 답할 때가 있어서,
그냥 json.loads 만으로는 안 된다.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _strip_fences(text: str) -> str:
    """```json ... ``` 으로 감싼 코드블록이면 껍데기를 벗긴다."""
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """첫 여는 괄호부터 짝이 맞는 닫는 괄호까지를 잘라낸다.

    문자열 안에 든 괄호는 세지 않는다(따옴표 안이면 건너뛴다).
    """
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
    """텍스트에서 JSON 객체 하나를 꺼낸다. 그대로 안 되면 `{...}` 부분만 잘라 다시 시도한다."""
    candidate = _strip_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    obj = _extract_balanced(candidate, "{", "}")
    if obj is None:
        raise ValueError(f"JSON 객체를 찾지 못함: {text[:200]!r}")
    return json.loads(obj)


def salvage_objects(text: str) -> list[dict[str, Any]]:
    """중간에 잘린 JSON 에서 온전한 객체만 골라 건진다.

    AI 답변이 길이 제한으로 끊기면 JSON 전체는 깨지지만, 그 앞쪽 객체들은 멀쩡하다.
    이걸 건지면 마지막 하나만 잃고 나머지 장소는 살릴 수 있다.
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
