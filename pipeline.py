"""[담당 3 · 파이프라인] 4-커맨드 CLI + doctor(환경 점검·설치) + C~J 통합 러너 + 출력 작성."""

from __future__ import annotations


# ╔══ pipeline/outputs.py ══╗


import argparse
import json
import random
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    GPT2_MODEL,
    IMAGE_EXTENSIONS,
    NLLB_MODEL,
    PIPELINE_VERSION,
    PROJECT_ROOT,
    STORY_CAPTION_FILENAME,
)
from common import file_url as _file_url
from common import html_escape as _html_escape
from common import write_json as _write_json
from common import log_stage
from common import llama_runtime_status, torch_runtime_status

BASE_DIR = PROJECT_ROOT
DEFAULT_INPUT_SEQUENCE = PROJECT_ROOT / "inputs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EVALUATION_DIR = DEFAULT_OUTPUT_ROOT / "evaluations"
EVALUATION_MAPPING_FILE = EVALUATION_DIR / "blind_mapping.json"
EVALUATION_RECORDS_FILE = EVALUATION_DIR / "evaluation_records.jsonl"
EVALUATION_SUMMARY_FILE = EVALUATION_DIR / "evaluation_summary.json"


# ─────────────────────────────────────────────────────────────────────────────
# 아래 본문은 기존 run.py의 이야기 선택/출력 작성 구역에서 이동한 코드.
# ─────────────────────────────────────────────────────────────────────────────
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_failure_metadata() -> dict[str, Any]:
    torch_status = torch_runtime_status()
    llama_status = llama_runtime_status()
    return {
        "torch_cuda_available": torch_status["torch_cuda_available"],
        "torch_device_name": torch_status["torch_device_name"],
        "torch_detail": torch_status["torch_detail"],
        "llama_mode": llama_status["llama_mode"],
        "llama_gpu_layers": llama_status["llama_gpu_layers"],
    }


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
        "runtime": _runtime_failure_metadata(),
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

# ╔══ pipeline/doctor.py ══╗


import argparse
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

from common import (
    BLIP_CAPTION_MODEL,
    BLIP_VQA_MODEL,
    DEFAULT_EXAONE_GGUF_PATH,
    GPT2_MODEL,
    LOCAL_HF_MODEL_DIR,
    NLLB_MODEL,
    PIPELINE_VERSION,
    QWEN25_VL_MODEL,
    STORY_CAPTION_FILENAME,
)
from common import (
    ensure_exaone_gguf_model,
    ensure_huggingface_model_snapshots,
    ensure_openclip_pretrained,
)
from common import llama_runtime_status, torch_runtime_status

# (import-name, pip-name) — doctor가 점검하는 런타임 의존성
_REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("open_clip", "open_clip_torch"),
    ("PIL", "Pillow"),
    ("sentencepiece", "sentencepiece"),
    ("google.protobuf", "protobuf"),
    ("huggingface_hub", "huggingface_hub"),
    ("qwen_vl_utils", "qwen-vl-utils"),
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("llama_cpp", "llama-cpp-python"),
]


def _check_line(label: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "WARN"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"


def _package_ok(import_name: str, pip_name: str) -> tuple[bool, str]:
    import importlib.util

    try:
        version = metadata.version(pip_name)
    except metadata.PackageNotFoundError:
        version = None
    importable = importlib.util.find_spec(import_name.split(".")[0]) is not None
    if importable:
        return True, version or "installed"
    return False, "not installed"


def _check_packages() -> bool:
    all_ok = True
    print("\nPython packages")
    for import_name, pip_name in _REQUIRED_PACKAGES:
        ok, detail = _package_ok(import_name, pip_name)
        all_ok = all_ok and ok
        _check_line(pip_name, ok, detail)
    return all_ok


def _check_compute() -> None:
    print("\nCompute")
    torch_status = torch_runtime_status()
    llama_status = llama_runtime_status()
    _check_line(
        "NVIDIA GPU",
        bool(torch_status["nvidia_gpu_detected"]),
        "found" if torch_status["nvidia_gpu_detected"] else "not found; CPU mode supported",
    )
    _check_line("PyTorch CUDA", bool(torch_status["torch_cuda_available"]), torch_status["torch_detail"])
    _check_line(
        "EXAONE GGUF mode",
        True,
        f"{llama_status['llama_mode']} (gpu_layers={llama_status['llama_gpu_layers']}; set LLAMA_GPU_LAYERS for GPU)",
    )


def _check_inputs(input_root: Path, story: str | None) -> bool:
    print("\nInputs")
    _check_line("input root", input_root.exists() and input_root.is_dir(), str(input_root))
    if not input_root.exists() or not input_root.is_dir():
        return False
    folders = _story_folders(input_root)
    _check_line("story folders", bool(folders), f"{len(folders)} folder(s)")
    for index, folder in enumerate(folders, start=1):
        images = _iter_images(folder)
        caption = folder / STORY_CAPTION_FILENAME
        caption_note = (
            "caption.txt yes"
            if caption.exists() and caption.read_text(encoding="utf-8").strip()
            else "caption.txt no"
        )
        _check_line(f"story {index}: {folder.name}", bool(images), f"{len(images)} image(s), {caption_note}")
    if story:
        try:
            selected = _match_story_folder(input_root, story)
        except ValueError as exc:
            _check_line("selected story", False, str(exc))
            return False
        _check_line("selected story", True, f"{selected} ({len(_iter_images(selected))} image(s))")
    return bool(folders)


def _check_local_assets() -> None:
    print("\nLocal model assets")
    hf_ready = LOCAL_HF_MODEL_DIR.exists() and any(LOCAL_HF_MODEL_DIR.iterdir())
    _check_line(
        "Hugging Face cache",
        hf_ready,
        str(LOCAL_HF_MODEL_DIR) if hf_ready else "will download on `storypipe doctor`",
    )
    import os

    gguf_path = Path(os.environ.get("EXAONE_GGUF_MODEL_PATH") or DEFAULT_EXAONE_GGUF_PATH).expanduser()
    _check_line(
        "EXAONE GGUF file",
        gguf_path.exists(),
        str(gguf_path) if gguf_path.exists() else "will download on `storypipe doctor`",
    )


def _preflight_models() -> None:
    print("\nDownloading / verifying models (first run can take a while; ~20GB)...")
    ensure_huggingface_model_snapshots([BLIP_CAPTION_MODEL, BLIP_VQA_MODEL, GPT2_MODEL, NLLB_MODEL])
    ensure_openclip_pretrained()
    ensure_huggingface_model_snapshots([QWEN25_VL_MODEL])
    ensure_exaone_gguf_model()


def _smoke_exaone() -> bool:
    from story_runtime import _run_exaone_gguf_prompt, ensure_exaone_gguf_runtime

    ensure_exaone_gguf_runtime()
    output = _run_exaone_gguf_prompt("한 문장으로 인사해 주세요.", max_new_tokens=24)
    ok = bool(output.strip())
    _check_line("EXAONE smoke inference", ok, output.strip()[:60] or "(empty output)")
    return ok


def run_doctor(args: argparse.Namespace) -> None:
    check_only = bool(getattr(args, "check_only", False))
    print(f"storypipe doctor ({PIPELINE_VERSION})")
    print(f"OS: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")

    free_bytes = shutil.disk_usage(BASE_DIR).free
    _check_line("free disk space", free_bytes >= 30 * 1024**3, f"{_format_bytes(free_bytes)} available; 30GB+ recommended")
    py_ok = (3, 10) <= sys.version_info[:2] <= (3, 12)
    _check_line("Python 3.10-3.12", py_ok, sys.version.split()[0])

    packages_ok = _check_packages()
    _check_compute()
    inputs_ok = _check_inputs(
        _resolve_workspace_path(getattr(args, "input_dir", str(DEFAULT_INPUT_SEQUENCE))),
        getattr(args, "story", None),
    )
    _check_local_assets()

    if check_only:
        print("\nResult")
        if packages_ok and inputs_ok:
            print("OK: setup looks ready. Run `storypipe doctor` (without --check-only) to download models.")
        else:
            print("WARN: fix the items above. Missing packages? Run `pip install -r requirements.txt`.")
            raise SystemExit(1)
        return

    if not packages_ok:
        print("\nMissing Python packages. Install them first:")
        print("  pip install -r requirements.txt")
        raise SystemExit(1)

    _preflight_models()
    smoke_ok = _smoke_exaone()

    print("\nResult")
    if inputs_ok and smoke_ok:
        print("OK: environment ready. Try:  storypipe demo")
    else:
        print("WARN: models are present but some checks failed; review the items above.")
        raise SystemExit(1)

# ╔══ pipeline/runner.py ══╗


import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    INPUT_DIR,
    OUTPUT_ROOT,
    VISION_MODEL_ID,
)
from common import experiment_dirs as _experiment_dirs
from common import file_url as _file_url
from common import html_escape as _html_escape
from common import log_stage, set_step_context
from story_experiments import (
    build_experiment_c,
    build_experiment_d,
    build_experiment_e,
    build_experiment_f,
    build_experiment_g,
    build_experiment_h,
    build_experiment_i,
    build_experiment_j,
)
from vision import (
    _read_story_caption,
    prepare_qwen_collage_for_experiment,
    prepare_qwen_scenes_for_experiment,
)

LLM_MODEL_NOTE = "EXAONE GGUF via llama.cpp"


# ─────────────────────────────────────────────────────────────────────────────
# 아래 본문은 기존 run_experiments_cd_qwen3b.py의 출력/오케스트레이션 구역에서 이동한 코드.
# ─────────────────────────────────────────────────────────────────────────────
def write_outputs(experiment_name: str, output_dir: Path, scenes: list[dict[str, Any]], result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": experiment_name,
        "vision_model": VISION_MODEL_ID,
        "llm_model": LLM_MODEL_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_order": [scene["image_id"] for scene in scenes],
        "scenes": scenes,
        **result,
    }
    (output_dir / f"{experiment_name.lower()}_result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    story = result["story"]
    (output_dir / f"{experiment_name.lower()}_story.txt").write_text(
        f"[제목]\n{story['title']}\n\n[동화]\n{story['body']}\n",
        encoding="utf-8",
    )
    scene_cards = []
    for scene, sentence in zip(scenes, story["scene_sentences"]):
        image_path = Path(str(scene.get("image_path") or INPUT_DIR / scene["image_id"]))
        scene_cards.append(
            f"""
            <article class="scene">
              <div class="image-frame"><img src="{_html_escape(_file_url(image_path))}" alt="{_html_escape(scene['image_id'])}"></div>
              <div class="text">
                <p class="no">{scene['scene_index']}번째 그림</p>
                <p class="label">EXAONE 장면 문장</p>
                <p class="sentence">{_html_escape(sentence)}</p>
                <p class="summary-label">Qwen 시각 요약</p>
                <p class="summary">{_html_escape(scene['scene_summary'])}</p>
              </div>
            </article>
            """
        )
    story_paragraphs = "\n".join(f"<p>{_html_escape(part)}</p>" for part in story["body"].split("\n\n"))
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html_escape(experiment_name)} - {_html_escape(story['title'])}</title>
<style>
body {{ margin:0; font-family:"Malgun Gothic",system-ui,sans-serif; background:#fff8e8; color:#2a211a; line-height:1.7; }}
header {{ padding:38px clamp(18px,5vw,70px); background:#fff1cf; border-bottom:1px solid #ddcfbd; }}
h1 {{ margin:0; font-size:clamp(30px,6vw,60px); letter-spacing:0; }}
main {{ max-width:1180px; margin:0 auto; padding:28px clamp(14px,3vw,36px) 60px; }}
.meta {{ color:#5f574f; }}
section {{ margin-top:26px; }}
.book {{ background:#fffdf7; border:1px solid #ddcfbd; border-radius:8px; padding:22px; }}
.book p {{ font-size:18px; margin:0 0 12px; word-break:keep-all; }}
.scene {{ display:grid; grid-template-columns:minmax(230px,42%) 1fr; gap:22px; align-items:center; margin:18px 0; padding:18px; background:#fffdf7; border:1px solid #ddcfbd; border-radius:8px; }}
.image-frame {{ aspect-ratio:4/3; border:1px solid #ddcfbd; border-radius:8px; background:white; overflow:hidden; }}
.image-frame img {{ width:100%; height:100%; object-fit:contain; display:block; }}
.no {{ margin:0 0 8px; color:#964b3f; font-weight:700; }}
.label {{ margin:0 0 6px; color:#2f6652; font-size:13px; font-weight:700; }}
.sentence {{ margin:0; font-size:clamp(18px,2.1vw,24px); word-break:keep-all; }}
.summary-label {{ margin:16px 0 4px; color:#6f6257; font-size:12px; font-weight:700; }}
.summary {{ margin:12px 0 0; color:#74695f; font-size:14px; }}
@media (max-width:760px) {{ .scene {{ grid-template-columns:1fr; }} .image-frame {{ aspect-ratio:1/1; }} }}
</style>
</head>
<body>
<header>
<p class="meta">{_html_escape(experiment_name)} · vision: {_html_escape(VISION_MODEL_ID)} · llm: {_html_escape(LLM_MODEL_NOTE)}</p>
<h1>{_html_escape(story['title'])}</h1>
</header>
<main>
<section class="book"><h2>[동화]</h2>{story_paragraphs}</section>
<section><h2>그림 옆 EXAONE 장면 문장</h2>{"".join(scene_cards)}</section>
</main>
</body>
</html>"""
    (output_dir / f"{experiment_name.lower()}_story.html").write_text(html, encoding="utf-8")


def _experiment_builders() -> dict[str, tuple[str, Any]]:
    return {
        "c": ("Experiment_C", build_experiment_c),
        "d": ("Experiment_D", build_experiment_d),
        "e": ("Experiment_E", build_experiment_e),
        "f": ("Experiment_F", build_experiment_f),
        "g": ("Experiment_G", build_experiment_g),
        "h": ("Experiment_H", build_experiment_h),
        "i": ("Experiment_I", build_experiment_i),
        "j": ("Experiment_J", build_experiment_j),
    }


def run_experiment_with_scenes(
    experiment: str,
    scenes: list[dict[str, Any]],
    output_root: str | Path = OUTPUT_ROOT,
    story_caption: str | None = None,
    collage_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    key = experiment.lower()
    dirs = _experiment_dirs(output_root)
    builders = _experiment_builders()
    experiment_name, builder = builders[key]
    set_step_context(experiment=experiment_name, phase="generation")
    log_stage(f"building {experiment_name}", step=key.upper(), model="Qwen scenes + EXAONE GGUF")
    if key == "j":
        result = builder(scenes, story_caption or "", collage_analysis or {})
    elif key == "i":
        result = builder(scenes, story_caption or "", collage_analysis or {})
    elif key == "h":
        result = builder(scenes, story_caption or "")
    else:
        result = builder(scenes)
    write_outputs(experiment_name, dirs[key], scenes, result)
    log_stage(f"saved {key.upper()}: {dirs[key]}", step=key.upper(), model="output")
    return {"output_dir": str(dirs[key]), "result": result}


def run_selected_experiments(
    experiments: list[str] | tuple[str, ...] = ("c", "d", "e", "f", "g", "h", "i", "j"),
    input_dir: str | Path = INPUT_DIR,
    output_root: str | Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = Path(output_root)
    input_dir = Path(input_dir)
    selected = [experiment.lower() for experiment in experiments]
    if "all" in selected:
        selected = ["c", "d", "e", "f", "g", "h", "i", "j"]

    results: dict[str, Any] = {}
    for key in selected:
        story_caption = _read_story_caption(input_dir) if key in {"h", "i", "j"} else None
        collage_analysis = (
            prepare_qwen_collage_for_experiment(
                input_dir,
                output_root,
                story_caption=story_caption or "",
                experiment=key,
            )
            if key in {"i", "j"}
            else None
        )
        set_step_context(experiment=key.upper(), phase="vision")
        log_stage(f"start Experiment {key.upper()} Qwen scene generation", step="Qwen", event="start")
        scenes = prepare_qwen_scenes_for_experiment(
            key,
            input_dir=input_dir,
            output_root=output_root,
            story_caption=story_caption,
            collage_analysis=collage_analysis,
        )
        set_step_context(experiment=key.upper(), phase="vision")
        log_stage(f"Experiment {key.upper()} Qwen scene generation succeeded", step="Qwen", event="success")
        results[key] = run_experiment_with_scenes(
            key,
            scenes,
            output_root=output_root,
            story_caption=story_caption,
            collage_analysis=collage_analysis,
        )
    return results


# ╔══ pipeline/cli.py ══╗


import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from common import PIPELINE_VERSION, PROJECT_ROOT
from common import log_stage, set_step_context
from common import ensure_runtime_ready
from experiment_a import run_experiment_a, run_sequence_story

EXPERIMENT_KEYS = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
DASHBOARD_PATH = PROJECT_ROOT / "dashboard.py"


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
