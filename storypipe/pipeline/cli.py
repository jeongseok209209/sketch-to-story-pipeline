"""[담당 3 · 파이프라인] storypipe 4-커맨드 CLI (사용자 진입점).

채점자가 외울 명령은 단 4개:
  0) storypipe doctor            환경 점검 + 필요한 모델/의존성 설치 + 피드백
  1) storypipe run <story> <exp> 이야기 + 실험버전(a~j) 1개 실행
  2) storypipe run-all <story>   이야기의 전체 실험(A~J) 실행
  3) storypipe demo              "7. 새로운 이야기" 전체 실험 후 블라인드 평가까지

story는 숫자(1, 7) 권장(한글 경로 회피). exp는 a~j.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from storypipe.common.config import PIPELINE_VERSION, PROJECT_ROOT
from storypipe.common.logging import log_stage, set_step_context
from storypipe.common.runtime import ensure_runtime_ready
from storypipe.pipeline.doctor import run_doctor
from storypipe.pipeline.experiment_a import run_experiment_a, run_sequence_story
from storypipe.pipeline.outputs import (
    DEFAULT_INPUT_SEQUENCE,
    DEFAULT_OUTPUT_ROOT,
    _iter_images,
    _match_story_folder,
    _print_all_result_table,
    _reset_evaluation_for_successful_all_run,
    _resolve_workspace_path,
    _summary_failure,
    _summary_success,
    _utc_now,
    _write_all_summary,
    _write_a_standard_result,
    _write_failure_record,
)
from storypipe.pipeline.runner import run_selected_experiments

EXPERIMENT_KEYS = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
DASHBOARD_PATH = PROJECT_ROOT / "storypipe" / "evaluation" / "dashboard.py"


def _resolve_story_dir(input_dir: str, story: str | None) -> Path:
    """--story(숫자/폴더명)를 실제 이야기 폴더 경로로 해석한다."""
    input_root = _resolve_workspace_path(input_dir)
    if story is None:
        if _iter_images(input_root):
            return input_root
        raise SystemExit("이야기를 지정하세요. 예: storypipe run 1 e  (또는 storypipe run-all 7)")
    selected = _match_story_folder(input_root, story)
    log_stage(f"selected story folder: {selected}", step="input", model="input")
    return selected


# ── 단일 실험 실행 헬퍼 ─────────────────────────────────────────────────────
def _run_experiment_a_batch(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    records = [run_experiment_a(str(image_path), output_dir=str(output_dir)) for image_path in _iter_images(input_dir)]
    return _write_a_standard_result(records, output_dir)


def _run_one(experiment: str, input_dir: Path, output_root: Path) -> None:
    key = experiment.lower()
    if key == "a":
        set_step_context(experiment="A", phase="generation")
        log_stage("start Experiment A backend=gpt2_nllb", step="A")
        _run_experiment_a_batch(input_dir, output_root / "A")
    elif key == "b":
        set_step_context(experiment="B", phase="generation")
        log_stage("start Experiment B backend=exaone_gguf_structured", step="B")
        run_sequence_story(image_dir=str(input_dir), output_dir=str(output_root / "B"))
    else:
        set_step_context(experiment=key.upper(), phase="generation")
        log_stage(f"start Experiment {key.upper()}", step=key.upper())
        run_selected_experiments(experiments=[key], input_dir=input_dir, output_root=output_root)


# ── 전체 실험(가드 적용) ────────────────────────────────────────────────────
def _run_guarded(experiment: str, output_dir: Path, runner: Callable[[], Any], nxt: str | None) -> dict[str, Any]:
    started = _utc_now()
    set_step_context(experiment=experiment, phase="generation")
    log_stage(f"start Experiment {experiment}", step=experiment, event="start")
    try:
        runner()
    except Exception as exc:
        failure = _write_failure_record(experiment, output_dir, exc, started)
        tail = f"; continuing to {nxt}" if nxt else "; no remaining experiment"
        set_step_context(experiment=experiment, phase="generation")
        log_stage(f"Experiment {experiment} failed{tail}", step=experiment, event="failed")
        return _summary_failure(failure)
    set_step_context(experiment=experiment, phase="generation")
    log_stage(f"Experiment {experiment} succeeded", step=experiment, event="success")
    return _summary_success(experiment, output_dir, started)


def _run_all(input_dir: Path, output_root: Path) -> None:
    started = _utc_now()
    summaries: list[dict[str, Any]] = []
    order = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for index, experiment in enumerate(order):
        nxt = order[index + 1] if index + 1 < len(order) else None
        output_dir = output_root / experiment
        key = experiment.lower()
        if key == "a":
            runner: Callable[[], Any] = lambda: _run_experiment_a_batch(input_dir, output_root / "A")
        elif key == "b":
            runner = lambda: run_sequence_story(image_dir=str(input_dir), output_dir=str(output_root / "B"))
        else:
            runner = lambda key=key: run_selected_experiments(
                experiments=[key], input_dir=input_dir, output_root=output_root
            )
        summaries.append(_run_guarded(experiment, output_dir, runner, nxt))
    _write_all_summary(output_root, started, summaries)
    set_step_context(experiment="ALL", phase="summary")
    log_stage(f"saved all-run summary: {output_root / 'all_run_summary.json'}", step="summary")
    _print_all_result_table(summaries)


# ── 블라인드 평가 대시보드 ──────────────────────────────────────────────────
def _launch_dashboard(port: int) -> None:
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit("streamlit이 설치되지 않았습니다. `pip install -r requirements.txt` 후 다시 시도하세요.")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_PATH),
        "--server.headless",
        "true",
        "--server.port",
        str(port),
    ]
    try:
        # 대시보드는 outputs/를 저장소 루트 기준 상대경로로 읽으므로 cwd를 PROJECT_ROOT로 고정한다.
        subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        set_step_context(experiment="EVALUATE", phase="dashboard")
        log_stage("evaluation dashboard stopped by user", step="evaluate", event="stopped")


# ── 커맨드 구현 ─────────────────────────────────────────────────────────────
def _cmd_run(args: argparse.Namespace) -> None:
    ensure_runtime_ready()
    input_dir = _resolve_story_dir(args.input_dir, args.story)
    _run_one(args.experiment, input_dir, Path(args.output_root))


def _cmd_run_all(args: argparse.Namespace) -> None:
    ensure_runtime_ready()
    input_dir = _resolve_story_dir(args.input_dir, args.story)
    _run_all(input_dir, Path(args.output_root))


def _cmd_demo(args: argparse.Namespace) -> None:
    ensure_runtime_ready()
    input_dir = _resolve_story_dir(args.input_dir, args.story)
    output_root = Path(args.output_root)
    _run_all(input_dir, output_root)
    _reset_evaluation_for_successful_all_run(output_root)
    _launch_dashboard(args.port)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storypipe",
        description="손그림 → 한국어 동화 파이프라인 (4-커맨드: doctor / run / run-all / demo).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="환경 점검 + 필요한 모델/의존성 설치 + 피드백 (먼저 실행).")
    p_doctor.add_argument("--check-only", action="store_true", help="다운로드 없이 점검만 한다.")
    p_doctor.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="이야기 입력 루트.")
    p_doctor.add_argument("--story", help="점검할 이야기(선택).")

    p_run = sub.add_parser("run", help="이야기 + 실험버전(a~j) 1개 실행. 예: storypipe run 1 e")
    p_run.add_argument("story", help="이야기 번호 또는 폴더명 (예: 1, 7).")
    p_run.add_argument("experiment", choices=EXPERIMENT_KEYS, help="실험 버전 a~j.")
    p_run.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="이야기 입력 루트.")
    p_run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="출력 루트.")

    p_all = sub.add_parser("run-all", help="이야기의 전체 실험(A~J) 실행. 예: storypipe run-all 1")
    p_all.add_argument("story", help="이야기 번호 또는 폴더명 (예: 1, 7).")
    p_all.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="이야기 입력 루트.")
    p_all.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="출력 루트.")

    p_demo = sub.add_parser("demo", help='"7. 새로운 이야기" 전체 실험 후 블라인드 평가 대시보드까지.')
    p_demo.add_argument("--story", default="7", help="데모 이야기 (기본 7).")
    p_demo.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="이야기 입력 루트.")
    p_demo.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="출력 루트.")
    p_demo.add_argument("--port", type=int, default=8501, help="Streamlit 포트.")

    return parser


def main() -> None:
    print(f"[storypipe {PIPELINE_VERSION}]")
    args = _build_parser().parse_args()
    if args.command == "doctor":
        run_doctor(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "run-all":
        _cmd_run_all(args)
    elif args.command == "demo":
        _cmd_demo(args)


if __name__ == "__main__":
    main()
