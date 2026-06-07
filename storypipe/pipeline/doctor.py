"""[담당 3 · 파이프라인] `storypipe doctor` (모드 0) — 환경 점검 + 필요한 설치 + 피드백.

재현성의 관문. 채점자 컴퓨터에서 이 한 명령으로 (1) 환경/의존성 점검, (2) 오픈소스 모델 자동
다운로드(BLIP/VQA/GPT2/NLLB/OpenCLIP/Qwen + EXAONE GGUF), (3) EXAONE 스모크 추론까지 끝낸다.
``--check-only``를 주면 다운로드 없이 점검만 한다.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

from storypipe.common.config import (
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
from storypipe.common.models import (
    ensure_exaone_gguf_model,
    ensure_huggingface_model_snapshots,
    ensure_openclip_pretrained,
)
from storypipe.common.runtime import llama_runtime_status, torch_runtime_status
from storypipe.pipeline.outputs import (
    BASE_DIR,
    DEFAULT_INPUT_SEQUENCE,
    _iter_images,
    _match_story_folder,
    _resolve_workspace_path,
    _story_folders,
)

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
    from storypipe.story.exaone_runtime import _run_exaone_gguf_prompt, ensure_exaone_gguf_runtime

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
