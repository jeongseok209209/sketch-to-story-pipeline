"""Unified command runner for experiments A/B/C/D/E/F/G/H/I."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
STORY_CAPTION_FILENAME = "caption.txt"
EVALUATION_DIR = DEFAULT_OUTPUT_ROOT / "evaluations"
EVALUATION_MAPPING_FILE = EVALUATION_DIR / "blind_mapping.json"
EVALUATION_RECORDS_FILE = EVALUATION_DIR / "evaluation_records.jsonl"
EVALUATION_SUMMARY_FILE = EVALUATION_DIR / "evaluation_summary.json"


def _add_story_input_args(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_SEQUENCE), help=help_text)
    parser.add_argument(
        "--story",
        help=(
            "Story folder to use under --input-dir. Accepts a list number, a numeric prefix "
            "such as 1, or the folder name."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sketch-to-story experiments by selecting a, b, c, d, e, f, g, h, i, all, or all-evaluate."
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    a_parser = subparsers.add_parser("a", help="Run Experiment A on one image or a batch.")
    a_parser.add_argument("--image", default="1", help="Image path or number. Defaults to 1.")
    a_parser.add_argument("--batch", action="store_true", help="Run all images in --input-dir.")
    _add_story_input_args(a_parser, help_text="Input image directory or story root.")
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
    _add_story_input_args(b_parser, help_text="Ordered image directory or story root.")
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

    for name in ("c", "d", "e", "f", "g", "h", "i"):
        sub = subparsers.add_parser(name, help=f"Run Experiment {name.upper()}.")
        _add_story_input_args(sub, help_text="Ordered image directory or story root.")
        sub.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")

    all_parser = subparsers.add_parser("all", help="Run A, B, C, D, E, F, G, H, and I in order.")
    _add_story_input_args(all_parser, help_text="Ordered image directory or story root.")
    all_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    all_parser.add_argument("--clip-threshold", type=float, default=0.22)
    all_parser.add_argument("--a-backend", choices=("gpt2_nllb", "exaone"), default="gpt2_nllb")
    all_parser.add_argument("--b-backend", choices=("exaone_gguf_structured",), default="exaone_gguf_structured")

    all_evaluate_parser = subparsers.add_parser(
        "all-evaluate",
        help="Run A-I, rebuild blind evaluation from successful results, then launch the dashboard.",
    )
    _add_story_input_args(all_evaluate_parser, help_text="Ordered image directory or story root.")
    all_evaluate_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root directory.")
    all_evaluate_parser.add_argument("--clip-threshold", type=float, default=0.22)
    all_evaluate_parser.add_argument("--a-backend", choices=("gpt2_nllb", "exaone"), default="gpt2_nllb")
    all_evaluate_parser.add_argument(
        "--b-backend",
        choices=("exaone_gguf_structured",),
        default="exaone_gguf_structured",
    )
    all_evaluate_parser.add_argument("--port", type=int, default=8501, help="Streamlit server port.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run the blind evaluation dashboard.")
    evaluate_parser.add_argument("--port", type=int, default=8501, help="Streamlit server port.")

    return parser


def _resolve_workspace_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return BASE_DIR / value


def _contains_images(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    return any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in directory.iterdir())


def _story_sort_key(path: Path) -> tuple[int, int | str, str]:
    match = re.match(r"^\s*(\d+)", path.name)
    if match:
        return (0, int(match.group(1)), path.name.casefold())
    return (1, path.name.casefold(), path.name.casefold())


def _story_folders(input_root: Path) -> list[Path]:
    if not input_root.exists() or not input_root.is_dir():
        return []
    return sorted(
        (path for path in input_root.iterdir() if path.is_dir() and _contains_images(path)),
        key=_story_sort_key,
    )


def _match_story_folder(input_root: Path, story: str) -> Path:
    folders = _story_folders(input_root)
    story = story.strip()
    if not story:
        raise ValueError("Story folder selection cannot be empty.")
    if story in {".", "0"} and _contains_images(input_root):
        return input_root
    if story.isdigit():
        index = int(story)
        if 1 <= index <= len(folders):
            return folders[index - 1]
        for folder in folders:
            prefix = folder.name.split(".", 1)[0].strip()
            if prefix == story:
                return folder
    normalized = story.casefold()
    exact_matches = [folder for folder in folders if folder.name.casefold() == normalized]
    if exact_matches:
        return exact_matches[0]
    partial_matches = [folder for folder in folders if normalized in folder.name.casefold()]
    if len(partial_matches) == 1:
        return partial_matches[0]
    available = ", ".join(folder.name for folder in folders) or "(none)"
    raise ValueError(f"Story folder not found: {story}. Available story folders: {available}")


def _prompt_story_folder(input_root: Path, folders: list[Path]) -> Path:
    use_root_images = _contains_images(input_root)
    print()
    print("사용할 이야기 폴더를 선택하세요.")
    if use_root_images:
        print("0. inputs 바로 아래 이미지 사용")
    for index, folder in enumerate(folders, start=1):
        print(f"{index}. {folder.name}")

    while True:
        choice = input("번호 또는 폴더명 입력: ").strip()
        if not choice:
            choice = "1" if folders else "0"
        try:
            return _match_story_folder(input_root, choice)
        except ValueError as exc:
            print(exc)


def _resolve_story_input(args: argparse.Namespace) -> None:
    if not hasattr(args, "input_dir"):
        return
    input_root = _resolve_workspace_path(args.input_dir)
    story = getattr(args, "story", None)
    folders = _story_folders(input_root)

    if story:
        selected = _match_story_folder(input_root, story)
    elif folders:
        if sys.stdin.isatty():
            selected = _prompt_story_folder(input_root, folders)
        elif not _contains_images(input_root) and len(folders) == 1:
            selected = folders[0]
        elif _contains_images(input_root):
            selected = input_root
        else:
            names = ", ".join(folder.name for folder in folders)
            raise SystemExit(f"Multiple story folders found. Re-run with --story. Available: {names}")
    else:
        selected = input_root

    args.input_dir = str(selected)
    args.selected_story = selected.name if selected != input_root else ""
    if selected != input_root:
        log_stage(f"selected story folder: {selected}", step="input", model="input")


def _iter_images(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _a_scene_summary(vision: dict[str, Any]) -> str:
    parts = []
    for key in ("raw_caption", "who", "actions", "scene", "mood"):
        value = str(vision.get(key, "")).strip()
        if value:
            parts.append(value)
    return " / ".join(parts)


def _a_record_sort_key(record: dict[str, Any]) -> tuple[int, int | str]:
    stem = Path(str(record.get("image_id", ""))).stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem.casefold())


def _write_a_standard_result(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    if not records:
        raise ValueError("Experiment A produced no records to normalize.")
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = _resolve_workspace_path(out_dir)
    ordered = sorted(records, key=_a_record_sort_key)
    story_sentences = [str(record.get("story_final", "")).strip() for record in ordered]
    scenes = [
        {
            "scene_index": index,
            "image_id": record.get("image_id", ""),
            "image_path": record.get("image_path", ""),
            "scene_summary": _a_scene_summary(record.get("vision") or {}),
            "vision": record.get("vision") or {},
            "metrics": record.get("metrics") or {},
        }
        for index, record in enumerate(ordered, start=1)
    ]
    body = "\n\n".join(sentence for sentence in story_sentences if sentence)
    aggregate_metrics = {
        "object_coverage_average": (
            sum(float((record.get("metrics") or {}).get("object_coverage", 0.0)) for record in ordered)
            / len(ordered)
        ),
        "char_count_total": sum(int((record.get("metrics") or {}).get("char_count", 0)) for record in ordered),
        "scene_count": len(ordered),
    }
    result = {
        "experiment": "A",
        "vision_model": "BLIP/BLIP-VQA/OpenCLIP",
        "llm_model": f"{GPT2_MODEL} + {NLLB_MODEL}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_order": [str(record.get("image_id", "")) for record in ordered],
        "scenes": scenes,
        "prompt_strategy": "single_image_baseline_normalized",
        "parsed_result": {
            "image_records": ordered,
            "aggregate_metrics": aggregate_metrics,
        },
        "json_repair_used": False,
        "story": {
            "title": "Experiment A Baseline",
            "body": body,
            "scene_sentences": story_sentences,
            "grounding_notes": [],
        },
        "structure": {
            "mode": "single_image_baseline_aggregate",
            "scene_count": len(ordered),
        },
        "plan": {
            "method": "BLIP/OpenCLIP per-image baseline -> normalized D-aligned story fields",
            "scene_order": [str(record.get("image_id", "")) for record in ordered],
        },
        "metrics": aggregate_metrics,
        "validation_policy": "d_aligned_story_fields",
        "experiment_method": "Experiment_A",
    }
    _write_standard_story_files(
        out_dir,
        "Experiment_A",
        result,
        result_filename="experiment_a_result.json",
        story_filename="experiment_a_story.txt",
        html_filename="experiment_a_story.html",
    )
    return result


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
        return _write_a_standard_result(records, args.output_dir)

    image_arg = Path(args.image)
    if image_arg.exists():
        image_path = image_arg
    else:
        image_path = None
        for extension in sorted(IMAGE_EXTENSIONS):
            candidate = Path(args.input_dir) / f"{args.image}{extension}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            image_path = Path(args.input_dir) / f"{args.image}.png"
    record = run_experiment_a(
        str(image_path),
        output_dir=args.output_dir,
        clip_threshold=args.clip_threshold,
        story_backend=args.story_backend,
        story_max_new_tokens=args.story_max_new_tokens,
    )
    return _write_a_standard_result([record], args.output_dir)


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
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        set_step_context(experiment="EVALUATE", phase="dashboard")
        log_stage("evaluation dashboard stopped by user", step="evaluate", event="stopped")


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
    if args.experiment in {"h", "i"}:
        caption_path = Path(args.input_dir) / STORY_CAPTION_FILENAME
        if not caption_path.exists():
            raise FileNotFoundError(f"Experiment {args.experiment.upper()} requires {caption_path}")
        if not caption_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Experiment {args.experiment.upper()} requires a non-empty caption file: {caption_path}")

    if args.experiment in {"all", "all-evaluate"}:
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
    elif args.experiment in {"c", "d", "e", "f", "g", "h", "i"}:
        _preflight_qwen_model()
        _preflight_exaone_gguf()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _html_escape(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _file_url(path: str | Path) -> str:
    return "file:///" + str(path).replace("\\", "/")


def _write_standard_story_files(
    output_dir: Path,
    experiment_name: str,
    record: dict[str, Any],
    *,
    result_filename: str,
    story_filename: str,
    html_filename: str,
) -> None:
    story = record["story"]
    _write_json(output_dir / result_filename, record)
    (output_dir / story_filename).write_text(
        f"[title]\n{story['title']}\n\n[story]\n{story['body']}\n",
        encoding="utf-8",
    )

    scenes = record.get("scenes") or []
    scene_cards = []
    for scene, sentence in zip(scenes, story.get("scene_sentences") or []):
        image_path = scene.get("image_path") or scene.get("image_id") or ""
        scene_cards.append(
            f"""
            <article class="scene">
              <div class="image-frame"><img src="{_html_escape(_file_url(image_path))}" alt="{_html_escape(scene.get('image_id', ''))}"></div>
              <div class="text">
                <p class="no">{_html_escape(scene.get('scene_index', ''))}</p>
                <p class="sentence">{_html_escape(sentence)}</p>
                <p class="summary">{_html_escape(scene.get('scene_summary', ''))}</p>
              </div>
            </article>
            """
        )
    story_paragraphs = "\n".join(
        f"<p>{_html_escape(part)}</p>" for part in str(story.get("body", "")).split("\n\n") if part.strip()
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html_escape(experiment_name)} - {_html_escape(story.get('title', ''))}</title>
<style>
body {{ margin:0; font-family:"Malgun Gothic",system-ui,sans-serif; background:#fff8e8; color:#2a211a; line-height:1.7; }}
header {{ padding:34px clamp(18px,5vw,70px); background:#fff1cf; border-bottom:1px solid #ddcfbd; }}
h1 {{ margin:0; font-size:clamp(28px,5vw,54px); letter-spacing:0; }}
main {{ max-width:1120px; margin:0 auto; padding:28px clamp(14px,3vw,36px) 60px; }}
.meta {{ color:#5f574f; }}
.book {{ background:#fffdf7; border:1px solid #ddcfbd; border-radius:8px; padding:22px; }}
.book p {{ font-size:18px; margin:0 0 12px; word-break:keep-all; }}
.scene {{ display:grid; grid-template-columns:minmax(220px,38%) 1fr; gap:22px; align-items:center; margin:18px 0; padding:18px; background:#fffdf7; border:1px solid #ddcfbd; border-radius:8px; }}
.image-frame {{ aspect-ratio:4/3; border:1px solid #ddcfbd; border-radius:8px; background:white; overflow:hidden; }}
.image-frame img {{ width:100%; height:100%; object-fit:contain; display:block; }}
.no {{ margin:0 0 8px; color:#964b3f; font-weight:700; }}
.sentence {{ margin:0; font-size:clamp(17px,2vw,23px); word-break:keep-all; }}
.summary {{ margin:12px 0 0; color:#74695f; font-size:14px; }}
@media (max-width:760px) {{ .scene {{ grid-template-columns:1fr; }} .image-frame {{ aspect-ratio:1/1; }} }}
</style>
</head>
<body>
<header>
<p class="meta">{_html_escape(experiment_name)} · vision: {_html_escape(record.get('vision_model', ''))} · llm: {_html_escape(record.get('llm_model', ''))}</p>
<h1>{_html_escape(story.get('title', ''))}</h1>
</header>
<main>
<section class="book"><h2>Story</h2>{story_paragraphs}</section>
<section><h2>Scenes</h2>{"".join(scene_cards)}</section>
</main>
</body>
</html>"""
    (output_dir / html_filename).write_text(html, encoding="utf-8")


def _relative_to_base(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _result_files_for_success(experiment: str, output_dir: Path) -> list[Path]:
    if not output_dir.is_absolute():
        output_dir = _resolve_workspace_path(output_dir)
    if experiment == "A":
        result_path = output_dir / "experiment_a_result.json"
        if result_path.exists():
            return [result_path]
        return sorted(output_dir.glob("*_experiment_a.json"))
    if experiment == "B":
        result_path = output_dir / "experiment_b_result.json"
        if result_path.exists():
            return [result_path]
        result_path = output_dir / "sequence_story.json"
        return [result_path] if result_path.exists() else []
    result_path = output_dir / f"experiment_{experiment.lower()}_result.json"
    return [result_path] if result_path.exists() else []


def _reset_evaluation_for_successful_all_run(output_root: Path) -> dict[str, dict[str, Any]]:
    summary_path = output_root / "all_run_summary.json"
    if not summary_path.is_absolute():
        summary_path = _resolve_workspace_path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"All-run summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mapping_items: list[dict[str, Any]] = []
    for item in summary.get("experiments", []):
        experiment = str(item.get("experiment", "")).upper()
        if item.get("status") != "success":
            continue
        output_dir = Path(str(item.get("output_dir") or output_root / experiment))
        result_files = _result_files_for_success(experiment, output_dir)
        if experiment == "A" and len(result_files) > 1:
            mapping_items.append(
                {
                    "experiment": experiment,
                    "result_files": [_relative_to_base(result_file) for result_file in result_files],
                }
            )
            continue
        for result_file in result_files:
            mapping_items.append(
                {
                    "experiment": experiment,
                    "result_file": _relative_to_base(result_file),
                }
            )

    random.SystemRandom().shuffle(mapping_items)
    mapping = {
        f"case_{index:03d}": item
        for index, item in enumerate(mapping_items, start=1)
    }

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(EVALUATION_MAPPING_FILE, mapping)
    for path in (EVALUATION_RECORDS_FILE, EVALUATION_SUMMARY_FILE):
        if path.exists():
            path.unlink()
    log_stage(
        f"rebuilt blind evaluation mapping from successful all-run results: {len(mapping)} cases",
        step="evaluate",
        model="output",
    )
    return mapping


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
    experiments = ["C", "D", "E", "F", "G", "H", "I"]
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


def _run_all_evaluate(args: argparse.Namespace) -> None:
    _run_all(args)
    _reset_evaluation_for_successful_all_run(Path(args.output_root))
    _run_evaluate(args)


def main() -> None:
    ensure_runtime_ready()
    args = _build_parser().parse_args()
    _resolve_story_input(args)
    _preflight_for_experiment(args)
    if args.experiment == "a":
        set_step_context(experiment="A", phase="generation")
        log_stage(f"start Experiment A backend={args.story_backend}", step="A")
        _run_a(args)
    elif args.experiment == "b":
        set_step_context(experiment="B", phase="generation")
        log_stage(f"start Experiment B backend={args.story_backend}", step="B")
        _run_b(args)
    elif args.experiment in {"c", "d", "e", "f", "g", "h", "i"}:
        set_step_context(experiment=args.experiment.upper(), phase="generation")
        log_stage(f"start Experiment {args.experiment.upper()}", step=args.experiment.upper())
        run_selected_experiments(
            experiments=[args.experiment],
            input_dir=args.input_dir,
            output_root=args.output_root,
        )
    elif args.experiment == "all":
        _run_all(args)
    elif args.experiment == "all-evaluate":
        _run_all_evaluate(args)
    elif args.experiment == "evaluate":
        _run_evaluate(args)


if __name__ == "__main__":
    main()
