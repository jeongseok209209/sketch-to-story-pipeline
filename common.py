"""[공유 토대] 설정·런타임·로깅·모델·이미지·IO·JSON 파싱 (3인 공유)."""

from __future__ import annotations


# ╔══ common/config.py ══╗


from pathlib import Path

# ── 모델 ID ────────────────────────────────────────────────────────────────
BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"
GPT2_MODEL = "gpt2-medium"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
OPENCLIP_MODEL = "ViT-H-14"
OPENCLIP_PRETRAINED = "laion2b_s32b_b79k"
OPENCLIP_HF_CACHE_REPO = "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
OPENCLIP_WEIGHTS_FILENAME = "open_clip_model.safetensors"
EXAONE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
EXAONE_GGUF_REPO_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"
EXAONE_GGUF_FILENAME = "EXAONE-4.0-1.2B-IQ4_XS.gguf"
QWEN25_VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
VISION_MODEL_ID = QWEN25_VL_MODEL

# 재현성: 비우면 최신(main). 정확한 revision 해시를 넣으면 그 버전으로 고정 다운로드.
MODEL_REVISIONS: dict[str, str] = {}

HF_PREFLIGHT_IGNORE_PATTERNS = (
    "onnx/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "tf_model.*",
    "flax_model.*",
    "rust_model.*",
)

# ── 경로 ──────────────────────────────────────────────────────────────────
# common.py는 저장소 루트에 있으므로 그 폴더가 PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
COMMON_OUTPUT_DIR = OUTPUT_ROOT / "qwen25_vl_3b_story"
SHARED_DIR = COMMON_OUTPUT_DIR / "scene_descriptions"
RESIZED_DIR = COMMON_OUTPUT_DIR / "_resized_input"
LOCAL_HF_MODEL_DIR = PROJECT_ROOT / ".local_models" / "huggingface"
DEFAULT_EXAONE_GGUF_PATH = str(PROJECT_ROOT / ".local_models" / "exaone" / EXAONE_GGUF_FILENAME)

# ── 입력 규칙 ──────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
STORY_CAPTION_FILENAME = "caption.txt"
COLLAGE_FILENAME = "collage_2x5_scene_order.png"

# ── 런타임 호환성(진단/안내 메시지용 참고값) ──────────────────────────────
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
PYTORCH_CUDA_PACKAGES = ("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0")
TRANSFORMERS_VERSION_SPEC = "transformers>=4.54.0,<5"
TORCH_MIN_SAFE_VERSION = (2, 6)

# ── 버전 태그 ──────────────────────────────────────────────────────────────
PIPELINE_VERSION = "2026.06.07-storypipe"
STEP_LOG_VERSION = "step-log-v1"


def experiment_dirs(output_root: Path) -> dict[str, Path]:
    """실험 키(c~j) → 출력 디렉터리 매핑. vision/pipeline이 공유하므로 common에 둔다."""
    return {key: output_root / key.upper() for key in ("c", "d", "e", "f", "g", "h", "i", "j")}


def hf_revision(model_id: str) -> str | None:
    """모델 ID에 고정된 revision을 반환(없으면 None=최신)."""
    import os

    env_key = "HF_REVISION_" + model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return os.environ.get(env_key) or MODEL_REVISIONS.get(model_id)

# ╔══ common/logging.py ══╗


import time
from contextlib import contextmanager
from typing import Any, Generator


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

# ╔══ common/runtime.py ══╗


import os
import shutil
import subprocess
from typing import Any



def get_device() -> Any:
    """CUDA가 가능하면 CUDA, 아니면 CPU를 반환한다."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def has_nvidia_gpu() -> bool:
    """이 머신에 NVIDIA GPU가 있어 보이는지 반환한다."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return True
        except (OSError, subprocess.SubprocessError):
            pass

    if os.name == "nt":
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
            return "nvidia" in completed.stdout.lower()
        except (OSError, subprocess.SubprocessError):
            return False
    return False


def _torch_cuda_status() -> tuple[bool, bool, str]:
    """(torch_importable, cuda_available, device_name_or_error)를 반환한다."""
    try:
        import torch
    except Exception as exc:
        return False, False, str(exc)

    try:
        if torch.cuda.is_available():
            return True, True, torch.cuda.get_device_name(0)
        return True, False, "torch installed without available CUDA"
    except Exception as exc:
        return True, False, str(exc)


def torch_runtime_status() -> dict[str, Any]:
    """preflight/실패 로그용 PyTorch CUDA 상태를 반환한다."""
    torch_importable, cuda_available, detail = _torch_cuda_status()
    return {
        "torch_importable": torch_importable,
        "torch_cuda_available": cuda_available,
        "torch_device_name": detail if cuda_available else "",
        "torch_detail": detail,
        "nvidia_gpu_detected": has_nvidia_gpu(),
    }


def configured_llama_gpu_layers() -> int:
    """요청된 llama.cpp GPU 레이어 수를 반환한다(기본 CPU=0)."""
    raw_value = os.environ.get("LLAMA_GPU_LAYERS")
    if raw_value is not None:
        try:
            return max(int(raw_value), 0)
        except ValueError:
            print(f"[llama] Ignoring invalid LLAMA_GPU_LAYERS={raw_value!r}; using CPU.")
    # 재현성 우선: 기본은 CPU. NVIDIA GPU가 있어도 명시적 opt-in(LLAMA_GPU_LAYERS)이 없으면 CPU.
    return 0


def llama_runtime_status() -> dict[str, Any]:
    """EXAONE llama.cpp GPU-offload 상태를 반환한다."""
    gpu_layers = configured_llama_gpu_layers()
    return {
        "llama_mode": "gpu" if gpu_layers > 0 else "cpu",
        "llama_gpu_layers": gpu_layers,
    }


def _parse_major_minor(version: str) -> tuple[int, int] | None:
    version_core = version.split("+", 1)[0]
    parts = version_core.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _torch_needs_security_upgrade() -> tuple[bool, str]:
    """Transformers는 torch.load 체크포인트 때문에 torch>=2.6이 필요하다."""
    try:
        import torch
    except Exception as exc:
        return True, f"torch is not importable: {exc}"

    version = getattr(torch, "__version__", "unknown")
    parsed = _parse_major_minor(version)
    if parsed is None:
        return True, f"torch version is unknown: {version}"
    if parsed < TORCH_MIN_SAFE_VERSION:
        return True, f"torch {version} is below required 2.6"
    return False, f"torch {version}"


def ensure_runtime_ready() -> None:
    """런타임 상태를 *검증·안내*만 한다(설치하지 않음). 설치는 ``storypipe doctor``로."""
    log_stage("runtime check start", step="runtime", model="Python/PyTorch/CUDA")
    llama_status = llama_runtime_status()
    print(
        "[runtime] EXAONE llama.cpp mode: "
        f"{llama_status['llama_mode']} (gpu_layers={llama_status['llama_gpu_layers']})"
    )
    _torch_importable, cuda_available, torch_detail = _torch_cuda_status()
    torch_needs_upgrade, torch_version_detail = _torch_needs_security_upgrade()
    print(f"[runtime] PyTorch: {torch_detail}")

    if torch_needs_upgrade:
        print(
            f"[runtime] WARN: {torch_version_detail}. "
            "Run `storypipe doctor` (or `python run.py doctor`) to fix dependencies."
        )
        return
    if cuda_available:
        print(f"[runtime] CUDA ready: {torch_detail}")
    elif has_nvidia_gpu():
        print(
            "[runtime] NVIDIA GPU detected but PyTorch CUDA unavailable; PyTorch models run on CPU. "
            "Install the CUDA build (see docs/ADVANCED.md) for GPU acceleration."
        )
    else:
        print("[runtime] CPU mode: no NVIDIA GPU detected (supported).")

# ╔══ common/models.py ══╗


import os
import shutil
from pathlib import Path



def ensure_exaone_gguf_model(model_path: str = "") -> str:
    """로컬 EXAONE GGUF 경로를 반환하고, 없으면 첫 사용 시 다운로드한다."""
    resolved_path = Path(
        model_path
        or os.environ.get("EXAONE_GGUF_MODEL_PATH", "")
        or DEFAULT_EXAONE_GGUF_PATH
    ).expanduser()
    if resolved_path.exists():
        return str(resolved_path)

    repo_id = os.environ.get("EXAONE_GGUF_REPO_ID", EXAONE_GGUF_REPO_ID)
    filename = os.environ.get("EXAONE_GGUF_FILENAME", resolved_path.name or EXAONE_GGUF_FILENAME)
    print(f"[runtime] EXAONE GGUF not found; downloading {repo_id}/{filename} ...")
    try:
        from huggingface_hub import hf_hub_download

        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=hf_revision(repo_id),
                local_dir=str(resolved_path.parent),
            )
        )
        if downloaded_path.resolve() != resolved_path.resolve():
            shutil.copy2(downloaded_path, resolved_path)
        print(f"[runtime] EXAONE GGUF ready: {resolved_path}")
        return str(resolved_path)
    except Exception as exc:
        raise FileNotFoundError(
            "EXAONE GGUF model file could not be prepared automatically. "
            f"Tried {repo_id}/{filename}. If the model requires Hugging Face login, run "
            "`huggingface-cli login` or set HF_TOKEN. You may also set EXAONE_GGUF_MODEL_PATH "
            f"to a local file, or place it at: {resolved_path}. Error: {exc}"
        ) from exc


def ensure_huggingface_model_snapshots(model_ids: list[str] | tuple[str, ...]) -> None:
    """PyTorch 파이프라인에 필요한 HF 파일만 내려받는다."""
    if not model_ids:
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required for model preflight: {exc}") from exc

    for model_id in dict.fromkeys(model_ids):
        log_stage(f"ensuring Hugging Face model: {model_id}", step="preflight", model=model_id)
        local_dir = LOCAL_HF_MODEL_DIR / model_id.replace("/", "--")
        try:
            snapshot_download(
                repo_id=model_id,
                revision=hf_revision(model_id),
                local_dir=str(local_dir),
                ignore_patterns=list(HF_PREFLIGHT_IGNORE_PATTERNS),
                max_workers=1,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to prepare Hugging Face model {model_id}: {exc}. "
                "If the model is gated, run `huggingface-cli login` or set HF_TOKEN."
            ) from exc


def local_huggingface_model_path(model_id: str) -> Path | str:
    """preflight가 준비해 둔 프로젝트-로컬 HF 모델 경로(없으면 모델 ID)를 반환한다."""
    local_dir = LOCAL_HF_MODEL_DIR / model_id.replace("/", "--")
    if local_dir.exists():
        return local_dir
    return model_id


def _local_files_only(source: Path | str) -> bool:
    return isinstance(source, Path)


def _openclip_pretrained_source() -> str:
    """OpenCLIP 가중치가 캐시돼 있으면 로컬 경로, 아니면 pretrained 태그를 반환한다."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_root = cache_root / OPENCLIP_HF_CACHE_REPO
    ref_path = repo_root / "refs" / "main"
    if ref_path.exists():
        snapshot_id = ref_path.read_text(encoding="utf-8").strip()
        snapshot_file = repo_root / "snapshots" / snapshot_id / OPENCLIP_WEIGHTS_FILENAME
        if snapshot_file.is_file():
            return str(snapshot_file)
    snapshots_root = repo_root / "snapshots"
    if snapshots_root.exists():
        for snapshot_dir in sorted((path for path in snapshots_root.iterdir() if path.is_dir()), reverse=True):
            snapshot_file = snapshot_dir / OPENCLIP_WEIGHTS_FILENAME
            if snapshot_file.is_file():
                return str(snapshot_file)
    return OPENCLIP_PRETRAINED


def ensure_openclip_pretrained() -> None:
    """생성 시작 전에 OpenCLIP pretrained 가중치를 다운로드/검증한다."""
    log_stage(
        f"ensuring OpenCLIP model: {OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}",
        step="preflight",
        model=f"{OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}",
    )
    try:
        import open_clip
        import torch

        pretrained_source = _openclip_pretrained_source()
        model, _unused, _preprocess = open_clip.create_model_and_transforms(
            OPENCLIP_MODEL,
            pretrained=pretrained_source,
            device="cpu",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to prepare OpenCLIP model {OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}: {exc}"
        ) from exc

# ╔══ common/images.py ══╗


from typing import Any


def load_and_normalize_image(image_path: str) -> Any:
    """이미지를 로드해 흰 배경 정규화 + 대비 보정한다(손그림 선을 또렷하게)."""
    from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(image_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to load image: {image_path}") from exc

    # 투명 배경 그림은 흰 배경 위에 합성해 RGB 입력으로 안정화한다.
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")

    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    return image


def resize_square(image: Any, size: int) -> Any:
    """중앙 crop 기반 정사각형 캔버스로 리사이즈한다."""
    from PIL import Image, ImageOps

    return ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)

# ╔══ common/io.py ══╗


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

# ╔══ common/jsonparse.py ══╗


import re


def balanced_json_object_candidates(cleaned: str) -> list[str]:
    """문자열 안의 균형 잡힌 ``{...}`` 후보들을 등장 순서대로 반환한다."""
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(cleaned[start : index + 1])
                start = None
    if not candidates:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            candidates.append(match.group(0))
    return candidates


def json_object_candidates(text: str) -> list[str]:
    """모델 텍스트에서 JSON 객체로 보이는 부분 문자열들을 반환한다(코드펜스 포함)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = balanced_json_object_candidates(cleaned)
    for start in [match.start() for match in re.finditer(r"\{", cleaned)][-80:]:
        suffix_candidates = balanced_json_object_candidates(cleaned[start:])
        if suffix_candidates:
            candidates.append(suffix_candidates[0])
    for match in re.finditer(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", text, flags=re.S):
        candidates.extend(balanced_json_object_candidates(match.group(1).strip()))
    return candidates
