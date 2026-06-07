"""[공통 토대] 모델 출력에서 JSON 객체 후보를 뽑아내는 저수준 헬퍼.

generators.py와 monster 파일에 중복돼 있던 ``_balanced_json_object_candidates`` /
``_json_object_candidates``를 통합(둘 중 fallback이 있는 superset 버전 채택).
도메인별 ``_extract_*`` 함수(structured_json vs story 키 검사 등)는 의미가 달라 각 모듈에 둔다.
"""

from __future__ import annotations

import re


def balanced_json_object_candidates(cleaned: str) -> list[str]:
    """문자열 안의 균형 잡힌 ``{...}`` 후보들을 등장 순서대로 반환한다."""
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(cleaned[start : index + 1])
                start = None
    if not candidates:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            candidates.append(match.group(0))
    return candidates


def json_object_candidates(text: str) -> list[str]:
    """모델 텍스트에서 JSON 객체로 보이는 부분 문자열들을 반환한다(코드펜스 포함)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = balanced_json_object_candidates(cleaned)
    for start in [match.start() for match in re.finditer(r"\{", cleaned)][-80:]:
        suffix_candidates = balanced_json_object_candidates(cleaned[start:])
        if suffix_candidates:
            candidates.append(suffix_candidates[0])
    for match in re.finditer(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", text, flags=re.S):
        candidates.extend(balanced_json_object_candidates(match.group(1).strip()))
    return candidates
