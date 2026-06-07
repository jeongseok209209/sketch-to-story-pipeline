"""[공통 토대] 구조화된 단계 로그.

긴 모델 추론 단계가 많아 콘솔에서 진행 상황/소요 시간을 확인할 수 있게 한다.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from storypipe.common.config import PIPELINE_VERSION, STEP_LOG_VERSION

_STEP_CONTEXT: dict[str, str] = {
    "experiment": "",
    "model": "",
    "phase": "",
}


def set_step_context(
    experiment: str | None = None,
    model: str | None = None,
    phase: str | None = None,
) -> None:
    """단계 로그에 쓰일 기본 메타데이터를 설정한다."""
    if experiment is not None:
        _STEP_CONTEXT["experiment"] = experiment
    if model is not None:
        _STEP_CONTEXT["model"] = model
    if phase is not None:
        _STEP_CONTEXT["phase"] = phase


def _stage_prefix(
    step: int | str | None = None,
    experiment: str | None = None,
    model: str | None = None,
    phase: str | None = None,
    event: str | None = None,
) -> str:
    parts = [
        f"[pipeline {PIPELINE_VERSION}]",
        f"[log {STEP_LOG_VERSION}]",
    ]
    resolved_experiment = experiment if experiment is not None else _STEP_CONTEXT.get("experiment", "")
    resolved_model = model if model is not None else _STEP_CONTEXT.get("model", "")
    resolved_phase = phase if phase is not None else _STEP_CONTEXT.get("phase", "")
    if resolved_experiment:
        parts.append(f"[experiment {resolved_experiment}]")
    if resolved_model:
        parts.append(f"[model {resolved_model}]")
    if resolved_phase:
        parts.append(f"[phase {resolved_phase}]")
    if step is not None:
        step_text = f"{step:02d}" if isinstance(step, int) else str(step)
        parts.append(f"[step {step_text}]")
    if event:
        parts.append(f"[{event}]")
    return "".join(parts)


def log_stage(
    message: str,
    step: int | str | None = None,
    experiment: str | None = None,
    model: str | None = None,
    phase: str | None = None,
    event: str = "info",
) -> None:
    """구조화된 단계 로그 한 줄을 출력한다."""
    print(f"{_stage_prefix(step, experiment, model, phase, event)} {message}")


def log_model_device(model_name: str, device: Any, phase: str = "model-load") -> None:
    """모델 로드 후 선택된 디바이스를 기록한다."""
    log_stage(f"device selected: {device}", step="device", model=model_name, phase=phase)


@contextmanager
def timed_step(
    step: int | str,
    label: str,
    experiment: str | None = None,
    model: str | None = None,
    phase: str | None = None,
) -> Generator[None, None, None]:
    """소요 시간이 포함된 번호 단계 로그를 출력한다."""
    start = time.perf_counter()
    print(f"{_stage_prefix(step, experiment, model, phase, 'start')} {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{_stage_prefix(step, experiment, model, phase, 'done')} {label} ({elapsed:.2f}s)")
