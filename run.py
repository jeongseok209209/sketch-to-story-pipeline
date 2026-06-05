"""Unified command runner for experiments A/B/C/D/E/F."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from generators import ensure_exaone_gguf_runtime
from pipeline_a import run_experiment_a, run_sequence_story
from run_experiments_cd_qwen3b import run_selected_experiments
from utils import (
    BLIP_CAPTION_MODEL,
    BLIP_VQA_MODEL,
    GPT2_MODEL,
    NLLB_MODEL,
    PIPELINE_VERSION,
    QWEN25_VL_MODEL,
    ensure_huggingface_model_snapshots,
    ensure_openclip_pretrained,
    ensure_runtime_ready,
    log_stage,
    set_step_context,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_SEQUENCE = BASE_DIR / "inputs"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "outputs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sketch-to-story experiments by selecting a, b, c, d, e, or f."
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    a_parser = subparsers.add_parser("a", help="Run Experiment A on one image or a batch.")
    a_parser.add_argument("--image", default="1", help="Image path or number. Defaults to 1.")
    a_parser.add_argument("--batch", action="store_true", help="Run all images in --input-dir.")
    a_parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="Input image directory.")
    a_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "A"), help="Output directory.")
    a_parser.add_argument("--clip-threshold", type=float, default=0.22)
    a_parser.add_argument(
        "--story-backend",
        choices=(
            "gpt2_nllb",
            "exaone",
        ),
        default="gpt2_nllb",
    )
    a_parser.add_argument("--story-max-new-tokens", type=int)

    b_parser = subparsers.add_parser("b", help="Run sequence/B-style story generation.")
    b_parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="Ordered image directory.")
    b_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT / "B"),
        help="Output directory.",
    )
    b_parser.add_argument("--clip-threshold", type=float, default=0.22)
    b_parser.add_argument(
        "--story-backend",
        choices=("exaone_gguf_structured",),
        default="exaone_gguf_structured",
    )
    b_parser.add_argument("--story-max-new-tokens", type=int)

    for name in ("c", "d", "e", "f"):
        sub = subparsers.add_parser(name, help=f"Run Experiment {name.upper()}.")
        sub.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="Ordered image directory.")
        sub.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")

    all_parser = subparsers.add_parser("all", help="Run A, B, C, D, E, and F in order.")
    all_parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help="Ordered image directory.")
    all_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    all_parser.add_argument("--clip-threshold", type=float, default=0.22)
    all_parser.add_argument("--a-backend", choices=("gpt2_nllb", "exaone"), default="gpt2_nllb")
    all_parser.add_argument("--b-backend", choices=("exaone_gguf_structured",), default="exaone_gguf_structured")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the blind evaluation dashboard.")
    evaluate_parser.add_argument("--port", type=int, default=8501, help="Streamlit server port.")

    return parser


def _iter_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _run_a(args: argparse.Namespace) -> Any:
    if args.batch:
        records = []
        for image_path in _iter_images(args.input_dir):
            records.append(
                run_experiment_a(
                    str(image_path),
                    output_dir=args.output_dir,
                    clip_threshold=args.clip_threshold,
                    story_backend=args.story_backend,
                    story_max_new_tokens=args.story_max_new_tokens,
                )
            )
        return records

    image_arg = Path(args.image)
    if image_arg.exists():
        image_path = image_arg
    else:
        image_path = Path(args.input_dir) / f"{args.image}.png"
    return run_experiment_a(
        str(image_path),
        output_dir=args.output_dir,
        clip_threshold=args.clip_threshold,
        story_backend=args.story_backend,
        story_max_new_tokens=args.story_max_new_tokens,
    )


def _run_b(args: argparse.Namespace) -> Any:
    return run_sequence_story(
        image_dir=args.input_dir,
        output_dir=args.output_dir,
        clip_threshold=args.clip_threshold,
        story_backend=args.story_backend,
        story_max_new_tokens=args.story_max_new_tokens,
    )


def _run_evaluate(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            "Streamlit is not installed. Install dependencies first with "
            "`python -m pip install -r requirements.txt`, then run "
            "`python run.py evaluate` again."
        )
    dashboard = BASE_DIR / "evaluation_dashboard.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard),
        "--server.headless",
        "true",
        "--server.port",
        str(args.port),
    ]
    subprocess.run(command, check=True)


def _preflight_a_models() -> None:
    ensure_huggingface_model_snapshots(
        [
            BLIP_CAPTION_MODEL,
            BLIP_VQA_MODEL,
            GPT2_MODEL,
            NLLB_MODEL,
        ]
    )
    ensure_openclip_pretrained()


def _preflight_qwen_model() -> None:
    ensure_huggingface_model_snapshots([QWEN25_VL_MODEL])


def _preflight_exaone_gguf() -> None:
    ensure_exaone_gguf_runtime()


def _preflight_for_experiment(args: argparse.Namespace) -> None:
    """Prepare required downloads/tools before actual generation begins."""
    set_step_context(experiment=args.experiment.upper(), phase="preflight")
    if args.experiment == "all":
        log_stage("preparing all downloads and runtime checks before generation", step="preflight")
        _preflight_a_models()
        _preflight_qwen_model()
        _preflight_exaone_gguf()
        log_stage("all required assets are ready; starting generation", step="preflight")
        return

    if args.experiment == "a" and getattr(args, "story_backend", "") in {
        "exaone_gguf_structured",
    }:
        _preflight_exaone_gguf()
    elif args.experiment == "b" and getattr(args, "story_backend", "") == "exaone_gguf_structured":
        _preflight_exaone_gguf()
    elif args.experiment in {"c", "d", "e", "f"}:
        _preflight_qwen_model()
        _preflight_exaone_gguf()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_failure_record(
    experiment: str,
    output_dir: Path,
    exc: BaseException,
    started_at: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ended_at = _utc_now()
    record: dict[str, Any] = {
        "experiment": experiment,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "started_at": started_at,
        "ended_at": ended_at,
        "output_dir": str(output_dir),
        "pipeline_version": PIPELINE_VERSION,
    }
    if extra:
        record.update(extra)
    failure_path = output_dir / "failure.json"
    _write_json(failure_path, record)
    return {**record, "failure_path": str(failure_path)}


def _summary_success(experiment: str, output_dir: Path, started_at: str) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "status": "success",
        "started_at": started_at,
        "ended_at": _utc_now(),
        "output_dir": str(output_dir),
    }


def _summary_failure(failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": failure["experiment"],
        "status": "failed",
        "started_at": failure["started_at"],
        "ended_at": failure["ended_at"],
        "output_dir": failure["output_dir"],
        "failure_file": failure["failure_path"],
        "error_type": failure["error_type"],
        "error_message": failure["error_message"],
    }


def _run_guarded_experiment(
    experiment: str,
    output_dir: Path,
    runner: Callable[[], Any],
    next_experiment: str | None,
) -> dict[str, Any]:
    started_at = _utc_now()
    set_step_context(experiment=experiment, phase="generation")
    log_stage(f"start Experiment {experiment}", step=experiment, event="start")
    try:
        runner()
    except Exception as exc:
        failure = _write_failure_record(experiment, output_dir, exc, started_at)
        next_message = (
            f"; continuing to Experiment {next_experiment}"
            if next_experiment
            else "; no remaining experiment"
        )
        set_step_context(experiment=experiment, phase="generation")
        log_stage(
            f"Experiment {experiment} failed{next_message}",
            step=experiment,
            event="failed",
        )
        return _summary_failure(failure)

    set_step_context(experiment=experiment, phase="generation")
    log_stage(f"Experiment {experiment} succeeded", step=experiment, event="success")
    return _summary_success(experiment, output_dir, started_at)


def _run_cdef_guarded(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    experiments = ["C", "D", "E", "F"]
    summaries = []
    for index, experiment in enumerate(experiments):
        next_experiment = experiments[index + 1] if index + 1 < len(experiments) else None
        output_dir = output_root / experiment
        key = experiment.lower()
        summaries.append(
            _run_guarded_experiment(
                experiment,
                output_dir,
                lambda key=key: run_selected_experiments(
                    experiments=[key],
                    input_dir=args.input_dir,
                    output_root=output_root,
                ),
                next_experiment,
            )
        )
    return summaries


def _write_all_summary(
    output_root: Path,
    started_at: str,
    experiments: list[dict[str, Any]],
) -> None:
    summary = {
        "status": "completed",
        "pipeline_version": PIPELINE_VERSION,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "output_root": str(output_root),
        "success_count": sum(1 for item in experiments if item["status"] == "success"),
        "failure_count": sum(1 for item in experiments if item["status"] == "failed"),
        "experiments": experiments,
    }
    _write_json(output_root / "all_run_summary.json", summary)


def _shorten(value: Any, max_length: int = 72) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _print_all_result_table(experiments: list[dict[str, Any]]) -> None:
    headers = ["Experiment", "Status", "Output", "Failure", "Error"]
    rows = []
    for item in experiments:
        rows.append(
            [
                str(item.get("experiment", "")),
                str(item.get("status", "")),
                str(item.get("output_dir", "")),
                str(item.get("failure_file", "")),
                _shorten(item.get("error_message", "")),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print()
    print("All Experiment Results")
    print(border)
    print("| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |")
    print(border)
    for row in rows:
        print("| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |")
    print(border)
    print()


def _run_all(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    all_started_at = _utc_now()
    summaries: list[dict[str, Any]] = []
    a_args = argparse.Namespace(
        image="1",
        batch=True,
        input_dir=args.input_dir,
        output_dir=str(output_root / "A"),
        clip_threshold=args.clip_threshold,
        story_backend=args.a_backend,
        story_max_new_tokens=None,
    )
    b_args = argparse.Namespace(
        input_dir=args.input_dir,
        output_dir=str(output_root / "B"),
        clip_threshold=args.clip_threshold,
        story_backend=args.b_backend,
        story_max_new_tokens=None,
    )

    summaries.append(
        _run_guarded_experiment(
            "A",
            output_root / "A",
            lambda: _run_a(a_args),
            "B",
        )
    )
    summaries.append(
        _run_guarded_experiment(
            "B",
            output_root / "B",
            lambda: _run_b(b_args),
            "C",
        )
    )
    summaries.extend(_run_cdef_guarded(args))
    _write_all_summary(output_root, all_started_at, summaries)
    set_step_context(experiment="ALL", phase="summary")
    log_stage(f"saved all-run summary: {output_root / 'all_run_summary.json'}", step="summary")
    _print_all_result_table(summaries)


def main() -> None:
    ensure_runtime_ready()
    args = _build_parser().parse_args()
    _preflight_for_experiment(args)
    if args.experiment == "a":
        set_step_context(experiment="A", phase="generation")
        log_stage(f"start Experiment A backend={args.story_backend}", step="A")
        _run_a(args)
    elif args.experiment == "b":
        set_step_context(experiment="B", phase="generation")
        log_stage(f"start Experiment B backend={args.story_backend}", step="B")
        _run_b(args)
    elif args.experiment in {"c", "d", "e", "f"}:
        set_step_context(experiment=args.experiment.upper(), phase="generation")
        log_stage(f"start Experiment {args.experiment.upper()}", step=args.experiment.upper())
        run_selected_experiments(
            experiments=[args.experiment],
            input_dir=args.input_dir,
            output_root=args.output_root,
        )
    elif args.experiment == "all":
        _run_all(args)
    elif args.experiment == "evaluate":
        _run_evaluate(args)


if __name__ == "__main__":
    main()
