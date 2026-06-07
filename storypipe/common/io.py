"""[공통 토대] 여러 모듈이 중복으로 갖고 있던 JSON/HTML/파일 헬퍼 통합.

(``_iter_images``는 모듈마다 의미가 달라(번호만/사전식/BASE_DIR 해석) 통합하지 않고 각자 유지한다.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """UTF-8 + indent=2로 JSON을 기록한다(부모 디렉터리 자동 생성)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def html_escape(value: Any) -> str:
    """HTML 특수문자를 이스케이프한다."""
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def file_url(path: str | Path) -> str:
    """로컬 경로를 file:/// URL로 변환한다(윈도우 역슬래시 정규화)."""
    return "file:///" + str(path).replace("\\", "/")
