"""Command line entry point for Experiment A."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate import evaluate
from generators import generate_story_en, translate_en_ko
from utils import timed_step
from vision import recognize_with_steps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_INPUT_DIR = "input"
BASE_DIR = Path(__file__).resolve().parent


def _ensure_local_venv_python() -> None:
    """Re-run this script with the project virtualenv Python when available."""
    venv_python = BASE_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if current_python == target_python:
        return

    print(f"Switching to project virtualenv: {target_python}")
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with Korean text preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_step_records(stage_dir: Path, steps: dict[str, Any]) -> None:
    """Save each stage record to an individual JSON file."""
    if stage_dir.exists():
        for old_json in stage_dir.glob("*.json"):
            old_json.unlink()
    for name, payload in steps.items():
        _write_json(stage_dir / f"{name}.json", payload)


def run_experiment_a(
    image_path: str,
    output_dir: str = "results/A",
    clip_threshold: float = 0.22,
) -> dict[str, Any]:
    """Run the full Experiment A pipeline and save the run record as JSON."""
    image = Path(image_path)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = out_dir / f"{image.stem}_steps"

    vision, steps = recognize_with_steps(str(image), clip_threshold=clip_threshold)
    story_en = generate_story_en(vision)
    steps["07_gpt2_story_en"] = {
        "step": 7,
        "name": "GPT-2 영문 동화 생성",
        "output": "story_en",
        "story_en": story_en,
    }
    story_final = translate_en_ko(story_en)
    steps["08_nllb_translation_ko"] = {
        "step": 8,
        "name": "NLLB 영한 번역",
        "output": "story_final",
        "story_final": story_final,
    }

    with timed_step(10, "evaluation and JSON save"):
        metrics = evaluate(vision, story_final)
        steps["09_evaluation"] = {
            "step": 9,
            "name": "평가 지표 계산",
            "output": "metrics",
            "metrics": metrics,
        }
        run_record = {
            "image_id": image.name,
            "experiment": "A",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_output_dir": str(stage_dir),
            "steps": steps,
            "vision": vision,
            "story_en": story_en,
            "story_final": story_final,
            "metrics": metrics,
        }
        output_path = out_dir / f"{image.stem}_experiment_a.json"
        _save_step_records(stage_dir, steps)
        _write_json(stage_dir / "run_record.json", run_record)
        _write_json(output_path, run_record)
        print(f"[10] saved: {output_path}")
        print(f"[10] step records saved: {stage_dir}")

    return run_record


def _resolve_image_path(image_arg: str, input_dir: str = DEFAULT_INPUT_DIR) -> Path:
    """Resolve an image path or numeric image name from the fixed input directory."""
    image = Path(image_arg)
    if image.exists():
        return image

    fixed_dir = Path(input_dir)
    if not fixed_dir.is_absolute():
        fixed_dir = BASE_DIR / fixed_dir
    if image.suffix.lower() in IMAGE_EXTENSIONS:
        candidate = fixed_dir / image.name
        if candidate.exists():
            return candidate
    else:
        for extension in sorted(IMAGE_EXTENSIONS):
            candidate = fixed_dir / f"{image_arg}{extension}"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Image not found. Put it in {fixed_dir}/ as {image_arg}.png, "
        f"{image_arg}.jpg, or {image_arg}.jpeg."
    )


def _iter_images(directory: str) -> list[Path]:
    """Return PNG/JPG files from a directory in lexical order."""
    root = Path(directory)
    if not root.is_absolute():
        root = BASE_DIR / root
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Run sketch-to-Korean-story Experiment A.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image",
        help="Image path, or a name like 1 that resolves to input/1.png or input/1.jpg.",
    )
    source.add_argument(
        "--batch",
        nargs="?",
        const=DEFAULT_INPUT_DIR,
        help="Run every PNG/JPG image in a directory. Defaults to input/.",
    )
    parser.add_argument("--output-dir", default="results/A", help="Directory for JSON output.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Fixed image input directory for numeric image names.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.22,
        help="OpenCLIP cosine threshold for accepting candidate words.",
    )
    return parser


def main() -> None:
    """Run Experiment A from the command line."""
    _ensure_local_venv_python()
    args = _build_parser().parse_args()
    if args.image:
        image_path = _resolve_image_path(args.image, input_dir=args.input_dir)
        run_experiment_a(
            str(image_path),
            output_dir=args.output_dir,
            clip_threshold=args.clip_threshold,
        )
        return

    if args.batch:
        image_paths = _iter_images(args.batch)
    else:
        try:
            image_path = _resolve_image_path("1", input_dir=args.input_dir)
        except FileNotFoundError:
            image_paths = _iter_images(args.input_dir)
        else:
            print(f"No arguments provided. Running default image: {image_path}")
            run_experiment_a(
                str(image_path),
                output_dir=args.output_dir,
                clip_threshold=args.clip_threshold,
            )
            return

    if not image_paths:
        raise ValueError(f"No PNG/JPG images found in input directory: {args.input_dir}")

    for image_path in image_paths:
        run_experiment_a(
            str(image_path),
            output_dir=args.output_dir,
            clip_threshold=args.clip_threshold,
        )


if __name__ == "__main__":
    main()
