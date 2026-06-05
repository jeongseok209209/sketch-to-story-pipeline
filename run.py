"""Unified command runner for experiments A/B/C/D/E/F."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_a import run_experiment_a, run_sequence_story
from run_experiments_cd_qwen3b import run_selected_experiments


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
            "structured_template",
            "structured_exaone",
            "exaone_structured",
            "exaone_gguf_structured",
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
        choices=("structured_template", "prompt_twostep_short", "exaone_gguf_structured"),
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
    all_parser.add_argument("--a-backend", default="gpt2_nllb")
    all_parser.add_argument("--b-backend", default="exaone_gguf_structured")

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


def main() -> None:
    args = _build_parser().parse_args()
    if args.experiment == "a":
        _run_a(args)
    elif args.experiment == "b":
        _run_b(args)
    elif args.experiment in {"c", "d", "e", "f"}:
        run_selected_experiments(
            experiments=[args.experiment],
            input_dir=args.input_dir,
            output_root=args.output_root,
        )
    elif args.experiment == "all":
        a_args = argparse.Namespace(
            image="1",
            batch=True,
            input_dir=args.input_dir,
            output_dir=str(Path(args.output_root) / "A"),
            clip_threshold=args.clip_threshold,
            story_backend=args.a_backend,
            story_max_new_tokens=None,
        )
        b_args = argparse.Namespace(
            input_dir=args.input_dir,
            output_dir=str(Path(args.output_root) / "B"),
            clip_threshold=args.clip_threshold,
            story_backend=args.b_backend,
            story_max_new_tokens=None,
        )
        _run_a(a_args)
        _run_b(b_args)
        run_selected_experiments(
            experiments=["c", "d", "e", "f"],
            input_dir=args.input_dir,
            output_root=args.output_root,
        )
    elif args.experiment == "evaluate":
        _run_evaluate(args)


if __name__ == "__main__":
    main()
