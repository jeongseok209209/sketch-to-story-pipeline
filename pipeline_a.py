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
from generators import (
    generate_sequence_story_exaone_gguf,
    generate_story_en,
    generate_story_ko_exaone,
    generate_structured_plan_exaone_gguf,
    get_last_llama_runtime,
    translate_en_ko,
)
from utils import clear_vision_model_caches, log_stage, timed_step
from vision import recognize_with_steps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_INPUT_DIR = "inputs"
DEFAULT_SEQUENCE_DIR = "inputs"
BASE_DIR = Path(__file__).resolve().parent


def _ensure_local_venv_python() -> None:
    """Re-run this script with the project virtualenv Python when available."""
    # 프로젝트 안에 .venv가 있으면 의존성 충돌을 줄이기 위해 그 Python으로 재실행합니다.
    venv_python = BASE_DIR / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if current_python == target_python:
        return

    # 현재 프로세스를 가상환경 Python 프로세스로 교체하고 기존 CLI 인자를 그대로 넘깁니다.
    print(f"Switching to project virtualenv: {target_python}")
    os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with Korean text preserved."""
    # 결과 디렉터리가 없을 수 있으므로 저장 직전에 생성합니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    """Write a plain UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _save_step_records(stage_dir: Path, steps: dict[str, Any]) -> None:
    """Save each stage record to an individual JSON file."""
    # 같은 이미지로 재실행할 때 이전 단계 JSON이 섞이지 않도록 비웁니다.
    if stage_dir.exists():
        for old_json in stage_dir.glob("*.json"):
            old_json.unlink()
    # 단계별 파일을 따로 저장해 실패 지점이나 중간 산출물을 쉽게 확인하게 합니다.
    for name, payload in steps.items():
        _write_json(stage_dir / f"{name}.json", payload)


def run_experiment_a(
    image_path: str,
    output_dir: str = "results/A",
    clip_threshold: float = 0.22,
    story_backend: str = "gpt2_nllb",
    story_max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Run the full Experiment A pipeline and save the run record as JSON."""
    # 상대 경로 출력 디렉터리는 experiment_a 폴더 기준으로 맞춰 실행 위치에 덜 흔들리게 합니다.
    image = Path(image_path)
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = out_dir / f"{image.stem}_steps"

    # 1단계: 이미지에서 object/scene/mood 등 이야기 생성을 위한 vision 정보를 추출합니다.
    vision, steps = recognize_with_steps(str(image), clip_threshold=clip_threshold)
    if story_backend == "exaone":
        # EXAONE은 vision JSON을 받아 한국어 동화를 직접 생성합니다.
        token_budget = story_max_new_tokens or 60
        clear_vision_model_caches()
        story_en = ""
        story_final = generate_story_ko_exaone(vision, max_new_tokens=token_budget)
        steps["07_exaone_story_ko"] = {
            "step": 7,
            "name": "EXAONE 한국어 동화 직접 생성",
            "output": "story_final",
            "max_new_tokens": token_budget,
            "story_final": story_final,
        }
    elif story_backend == "gpt2_nllb":
        # 2단계: vision 정보를 바탕으로 먼저 영어 동화를 생성합니다.
        token_budget = story_max_new_tokens or 200
        story_en = generate_story_en(vision, max_new_tokens=token_budget)
        steps["07_gpt2_story_en"] = {
            "step": 7,
            "name": "GPT-2 영문 동화 생성",
            "output": "story_en",
            "max_new_tokens": token_budget,
            "story_en": story_en,
        }
        # 3단계: 생성된 영어 이야기를 최종 제출용 한국어 이야기로 번역합니다.
        story_final = translate_en_ko(story_en)
        steps["08_nllb_translation_ko"] = {
            "step": 8,
            "name": "NLLB 영한 번역",
            "output": "story_final",
            "story_final": story_final,
        }

    else:
        raise ValueError(f"Unsupported Experiment A story backend: {story_backend}")

    with timed_step(10, "evaluation and JSON save"):
        # object 반영률과 기본 분량 지표를 계산해 실험 결과 비교에 사용합니다.
        metrics = evaluate(
            vision,
            story_final,
            translate_objects=story_backend == "gpt2_nllb",
        )
        evaluation_key = "09_evaluation"
        steps[evaluation_key] = {
            "step": 9,
            "name": "평가 지표 계산",
            "output": "metrics",
            "metrics": metrics,
        }
        # 전체 실행 기록은 한 파일에 모으고, steps 폴더에는 단계별 JSON도 함께 저장합니다.
        run_record = {
            "image_id": image.name,
            "experiment": "A",
            "story_backend": story_backend,
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
        log_stage(f"saved JSON: {output_path}", step=10, model="output")
        log_stage(f"saved step records: {stage_dir}", step=10, model="output")

    return run_record


def _build_structured_for_backend(
    vision: dict[str, Any],
    story_backend: str,
    story_max_new_tokens: int | None,
) -> tuple[dict[str, Any], dict[str, Any], str, int, str]:
    """Build structured and plan JSON for one scene with EXAONE GGUF."""
    if story_backend != "exaone_gguf_structured":
        raise ValueError(f"Experiment B requires EXAONE GGUF backend, got: {story_backend}")
    token_budget = story_max_new_tokens or 700
    clear_vision_model_caches()
    structured_json, plan_json, raw_response = generate_structured_plan_exaone_gguf(
        vision,
        max_new_tokens=token_budget,
    )
    return structured_json, plan_json, raw_response, token_budget, "EXAONE GGUF ???/??"

def run_sequence_story(
    image_dir: str = DEFAULT_SEQUENCE_DIR,
    output_dir: str = "results/sequence",
    clip_threshold: float = 0.22,
    story_backend: str = "exaone_gguf_structured",
    story_max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Run ordered images such as 01.png, 02.png as one connected story."""
    image_paths = _iter_sequence_images(image_dir)
    if not image_paths:
        raise ValueError(f"No PNG/JPG images found in sequence directory: {image_dir}")

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = BASE_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = out_dir / "sequence_steps"

    scene_records: list[dict[str, Any]] = []
    sequence_steps: dict[str, Any] = {}
    for index, image_path in enumerate(image_paths, start=1):
        log_stage(f"scene {index:02d}: {image_path.name}", step=f"B-{index:02d}", model="BLIP/OpenCLIP")
        vision, vision_steps = recognize_with_steps(str(image_path), clip_threshold=clip_threshold)
        structured_json, plan_json, raw_response, token_budget, source = _build_structured_for_backend(
            vision,
            story_backend,
            story_max_new_tokens,
        )
        scene_record = {
            "scene_index": index,
            "image_id": image_path.name,
            "vision": vision,
            "structured_json": structured_json,
            "plan_json": plan_json,
            "structuring_source": source,
            "max_new_tokens": token_budget,
            "exaone_raw_response": raw_response,
        }
        scene_records.append(scene_record)
        sequence_steps[f"{index:02d}_{image_path.stem}"] = {
            "image_id": image_path.name,
            "vision_steps": vision_steps,
            "scene_record": scene_record,
        }

    story_final, sequence_raw_response = generate_sequence_story_exaone_gguf(
        scene_records,
        max_new_tokens=story_max_new_tokens
        or min(3600, max(1200, len(scene_records) * 360)),
    )
    sequence_llama_runtime = get_last_llama_runtime()
    finalizer = "EXAONE GGUF ?? ?? ??"
    output_path = out_dir / "sequence_story.json"
    text_output_path = out_dir / "sequence_story.txt"
    run_record = {
        "experiment": "sequence",
        "story_backend": story_backend,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_order": [path.name for path in image_paths],
        "step_output_dir": str(stage_dir),
        "story_text_output": str(text_output_path),
        "scenes": scene_records,
        "story_final": story_final,
        "finalizer": finalizer,
        "sequence_exaone_raw_response": sequence_raw_response,
        "sequence_llama_runtime": sequence_llama_runtime,
    }

    _save_step_records(stage_dir, sequence_steps)
    _write_json(stage_dir / "run_record.json", run_record)
    _write_json(output_path, run_record)
    _write_text(text_output_path, story_final)
    log_stage(f"saved JSON: {output_path}", step="B-save", model="output")
    log_stage(f"saved story text: {text_output_path}", step="B-save", model="output")
    log_stage(f"saved step records: {stage_dir}", step="B-save", model="output")
    return run_record


def _resolve_image_path(image_arg: str, input_dir: str = DEFAULT_INPUT_DIR) -> Path:
    """Resolve an image path or numeric image name from the fixed input directory."""
    # 사용자가 완전한 파일 경로를 넘긴 경우에는 그대로 사용합니다.
    image = Path(image_arg)
    if image.exists():
        return image

    # 숫자만 입력해도 input/1.png 같은 고정 입력 폴더의 파일로 해석합니다.
    fixed_dir = Path(input_dir)
    if not fixed_dir.is_absolute():
        fixed_dir = BASE_DIR / fixed_dir
    if image.suffix.lower() in IMAGE_EXTENSIONS:
        candidate = fixed_dir / image.name
        if candidate.exists():
            return candidate
    else:
        # 확장자가 없으면 지원하는 이미지 확장자를 차례대로 붙여 탐색합니다.
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
    # batch 실행을 위해 지정 디렉터리의 지원 이미지 파일만 정렬해서 반환합니다.
    root = Path(directory)
    if not root.is_absolute():
        root = BASE_DIR / root
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _iter_sequence_images(directory: str) -> list[Path]:
    """Return ordered sequence images, preferring 01.png over duplicate 1.png."""
    paths = _iter_images(directory)
    numbered: dict[int, Path] = {}
    others: list[Path] = []
    for path in paths:
        if path.stem.isdigit():
            number = int(path.stem)
            current = numbered.get(number)
            if current is None or len(path.stem) > len(current.stem):
                numbered[number] = path
        else:
            others.append(path)
    return [numbered[key] for key in sorted(numbered)] + sorted(others)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    # --image와 --batch는 동시에 의미가 충돌하므로 argparse의 mutually exclusive group으로 묶습니다.
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
    source.add_argument(
        "--sequence-dir",
        nargs="?",
        const=DEFAULT_SEQUENCE_DIR,
        help="Run ordered images such as 01.png, 02.png as one connected story.",
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
    parser.add_argument(
        "--story-backend",
        choices=(
            "gpt2_nllb",
            "exaone",
        ),
        default="gpt2_nllb",
        help="Story generator to use: GPT-2 + NLLB or direct EXAONE.",
    )
    parser.add_argument(
        "--story-max-new-tokens",
        type=int,
        help=(
            "Override story generation token budget. Defaults: 200 for gpt2_nllb, "
            "60 for exaone."
        ),
    )
    return parser


def main() -> None:
    """Run Experiment A from the command line."""
    # CLI 진입 시 먼저 가상환경을 보장한 뒤 사용자 인자를 해석합니다.
    _ensure_local_venv_python()
    args = _build_parser().parse_args()
    if args.image:
        # 단일 이미지 모드: 입력 하나만 처리하고 종료합니다.
        image_path = _resolve_image_path(args.image, input_dir=args.input_dir)
        run_experiment_a(
            str(image_path),
            output_dir=args.output_dir,
            clip_threshold=args.clip_threshold,
            story_backend=args.story_backend,
            story_max_new_tokens=args.story_max_new_tokens,
        )
        return

    if args.sequence_dir:
        run_sequence_story(
            image_dir=args.sequence_dir,
            output_dir=args.output_dir,
            clip_threshold=args.clip_threshold,
            story_backend=args.story_backend,
            story_max_new_tokens=args.story_max_new_tokens,
        )
        return

    if args.batch:
        # batch 모드: 지정 폴더의 모든 이미지를 처리합니다.
        image_paths = _iter_images(args.batch)
    else:
        try:
            # 인자가 없으면 데모용 기본 이미지 1을 우선 실행합니다.
            image_path = _resolve_image_path("1", input_dir=args.input_dir)
        except FileNotFoundError:
            # 기본 이미지가 없으면 input_dir 전체를 batch처럼 처리합니다.
            image_paths = _iter_images(args.input_dir)
        else:
            print(f"No arguments provided. Running default image: {image_path}")
            run_experiment_a(
                str(image_path),
                output_dir=args.output_dir,
                clip_threshold=args.clip_threshold,
                story_backend=args.story_backend,
                story_max_new_tokens=args.story_max_new_tokens,
            )
            return

    if not image_paths:
        raise ValueError(f"No PNG/JPG images found in input directory: {args.input_dir}")

    # batch 대상 이미지를 하나씩 독립 실행해 각 이미지별 결과 JSON을 만듭니다.
    for image_path in image_paths:
        run_experiment_a(
            str(image_path),
            output_dir=args.output_dir,
            clip_threshold=args.clip_threshold,
            story_backend=args.story_backend,
            story_max_new_tokens=args.story_max_new_tokens,
        )


if __name__ == "__main__":
    main()
