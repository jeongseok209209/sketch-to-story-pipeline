"""Shared utilities for Experiment A."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Generator


BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"
GPT2_MODEL = "gpt2-medium"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
OPENCLIP_MODEL = "ViT-H-14"
OPENCLIP_PRETRAINED = "laion2b_s32b_b79k"
EXAONE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
EXAONE_GGUF_REPO_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"
EXAONE_GGUF_FILENAME = "EXAONE-4.0-1.2B-IQ4_XS.gguf"
QWEN25_VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
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
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_HF_MODEL_DIR = PROJECT_ROOT / ".local_models" / "huggingface"
DEFAULT_EXAONE_GGUF_PATH = str(
    PROJECT_ROOT / ".local_models" / "exaone" / EXAONE_GGUF_FILENAME
)
LLAMA_CLI_FILENAME = "llama-cli.exe" if os.name == "nt" else "llama-cli"
LLAMA_CLI_PATH = str(
    PROJECT_ROOT / ".local_tools" / "llama.cpp" / "build" / "bin" / LLAMA_CLI_FILENAME
)
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
PYTORCH_CUDA_PACKAGES = ("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0")
TRANSFORMERS_VERSION_SPEC = "transformers>=4.54.0,<5"
TORCH_MIN_SAFE_VERSION = (2, 6)
PIPELINE_VERSION = "2026.06.05-independent-exaone-gguf"
STEP_LOG_VERSION = "step-log-v1"
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
    """Set default metadata used by stage logs."""
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
    """Print one structured stage log line."""
    print(f"{_stage_prefix(step, experiment, model, phase, event)} {message}")


def get_device() -> Any:
    """Return CUDA when available, otherwise CPU."""
    import torch

    # GPU가 있으면 모델 추론 속도를 위해 CUDA를 우선 사용하고, 없으면 CPU로 동작합니다.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def has_nvidia_gpu() -> bool:
    """Return whether this machine appears to have an NVIDIA GPU."""
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
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object -ExpandProperty Name",
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
    """Return (torch_importable, cuda_available, device_name_or_error)."""
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


def _parse_major_minor(version: str) -> tuple[int, int] | None:
    """Parse the major/minor prefix from package versions such as 2.6.0+cu124."""
    version_core = version.split("+", 1)[0]
    parts = version_core.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _torch_needs_security_upgrade() -> tuple[bool, str]:
    """Transformers needs torch>=2.6 for torch.load-based model checkpoints."""
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


def _install_cuda_torch() -> bool:
    """Install CUDA-enabled PyTorch into the current Python environment."""
    index_url = os.environ.get("PYTORCH_CUDA_INDEX_URL", PYTORCH_CUDA_INDEX_URL)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        *PYTORCH_CUDA_PACKAGES,
        "--index-url",
        index_url,
    ]
    print("[runtime] Installing CUDA PyTorch for NVIDIA GPU support...")
    print("[runtime] " + " ".join(command))
    try:
        subprocess.run(command, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[runtime] CUDA PyTorch install failed; continuing with CPU if possible: {exc}")
        return False


def _install_cpu_torch() -> bool:
    """Install a torch version new enough for Transformers checkpoint loading."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch>=2.6",
    ]
    print("[runtime] Installing CPU PyTorch compatibility update...")
    print("[runtime] " + " ".join(command))
    try:
        subprocess.run(command, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[runtime] CPU PyTorch compatibility install failed: {exc}")
        return False


def _install_transformers_compat() -> bool:
    """Install a Transformers version compatible with the PyTorch runtime."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        TRANSFORMERS_VERSION_SPEC,
    ]
    print("[runtime] Installing Transformers compatibility pin...")
    print("[runtime] " + " ".join(command))
    try:
        subprocess.run(command, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[runtime] Transformers compatibility install failed: {exc}")
        return False


def _ensure_transformers_compat() -> tuple[bool, bool]:
    """Return (compatible, changed) for the Transformers BLIP dependency."""
    try:
        version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        print("[runtime] Transformers is not installed.")
        installed = _install_transformers_compat()
        return installed, installed

    major_text = version.split(".", 1)[0]
    try:
        major = int(major_text)
    except ValueError:
        print(f"[runtime] Transformers version is unusual ({version}); enforcing compatibility pin.")
        installed = _install_transformers_compat()
        return installed, installed

    if major < 5:
        return True, False

    print(f"[runtime] Transformers {version} is incompatible with this BLIP stack.")
    installed = _install_transformers_compat()
    return installed, installed


def ensure_runtime_ready() -> None:
    """Prepare the current runtime for CUDA when an NVIDIA GPU is available."""
    log_stage("runtime check start", step="runtime", model="Python/PyTorch/CUDA")
    _transformers_compatible, package_changed = _ensure_transformers_compat()
    nvidia_available = has_nvidia_gpu()
    _torch_importable, cuda_available, torch_detail = _torch_cuda_status()
    torch_needs_upgrade, torch_version_detail = _torch_needs_security_upgrade()

    if not _transformers_compatible:
        print("[runtime] Transformers compatibility is not ready; continuing may fail at model import.")

    if torch_needs_upgrade:
        print(f"[runtime] PyTorch compatibility update needed: {torch_version_detail}")
        if nvidia_available and _env_flag("AUTO_INSTALL_TORCH_CUDA", True):
            if _install_cuda_torch():
                package_changed = True
        elif nvidia_available:
            print("[runtime] AUTO_INSTALL_TORCH_CUDA=0; skipping CUDA PyTorch install.")
        elif _install_cpu_torch():
            package_changed = True

        _torch_importable, cuda_available, torch_detail = _torch_cuda_status()
        torch_needs_upgrade, torch_version_detail = _torch_needs_security_upgrade()

    if cuda_available and not package_changed and not torch_needs_upgrade:
        print(f"[runtime] CUDA ready: {torch_detail}")
        return

    if not nvidia_available and not package_changed and not torch_needs_upgrade:
        print("[runtime] CPU mode: no NVIDIA GPU detected.")
        return

    if nvidia_available and not cuda_available and not torch_needs_upgrade:
        print(f"[runtime] NVIDIA GPU detected, but CUDA PyTorch is not ready: {torch_detail}")
        if not _env_flag("AUTO_INSTALL_TORCH_CUDA", True):
            print("[runtime] AUTO_INSTALL_TORCH_CUDA=0; skipping CUDA PyTorch install.")
        elif _install_cuda_torch():
            package_changed = True

        _torch_importable, cuda_available, torch_detail = _torch_cuda_status()
        if cuda_available and not package_changed:
            print(f"[runtime] CUDA ready after install: {torch_detail}")
            return

    if package_changed and os.environ.get("SKETCH_STORY_RUNTIME_RESTARTED") != "1":
        print("[runtime] Restarting Python once so runtime package changes are loaded...")
        os.environ["SKETCH_STORY_RUNTIME_RESTARTED"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])

    if cuda_available:
        print(f"[runtime] CUDA ready: {torch_detail}")
    elif nvidia_available:
        print(f"[runtime] CUDA still unavailable after install; continuing on CPU: {torch_detail}")
    else:
        print("[runtime] CPU mode: no NVIDIA GPU detected.")


def ensure_exaone_gguf_model(model_path: str = "") -> str:
    """Return a local EXAONE GGUF path, downloading it on first use when missing."""
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
                local_dir=str(resolved_path.parent),
                local_dir_use_symlinks=False,
            )
        )
        if downloaded_path.resolve() != resolved_path.resolve():
            shutil.copy2(downloaded_path, resolved_path)
        print(f"[runtime] EXAONE GGUF ready: {resolved_path}")
        return str(resolved_path)
    except Exception as exc:
        raise FileNotFoundError(
            "EXAONE GGUF model file could not be prepared automatically. "
            f"Tried {repo_id}/{filename}. Set EXAONE_GGUF_MODEL_PATH or place it at: "
            f"{resolved_path}. Error: {exc}"
        ) from exc


def ensure_huggingface_model_snapshots(model_ids: list[str] | tuple[str, ...]) -> None:
    """Download only the Hugging Face files needed by the PyTorch pipeline."""
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
                local_dir=str(local_dir),
                ignore_patterns=list(HF_PREFLIGHT_IGNORE_PATTERNS),
                max_workers=1,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare Hugging Face model {model_id}: {exc}") from exc


def local_huggingface_model_path(model_id: str) -> Path | str:
    """Return the project-local HF model directory when preflight prepared it."""
    local_dir = LOCAL_HF_MODEL_DIR / model_id.replace("/", "--")
    if local_dir.exists():
        return local_dir
    return model_id


def _local_files_only(source: Path | str) -> bool:
    return isinstance(source, Path)


def ensure_openclip_pretrained() -> None:
    """Download/verify the OpenCLIP pretrained weights before generation starts."""
    log_stage(
        f"ensuring OpenCLIP model: {OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}",
        step="preflight",
        model=f"{OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}",
    )
    try:
        import open_clip
        import torch

        model, _unused, _preprocess = open_clip.create_model_and_transforms(
            OPENCLIP_MODEL,
            pretrained=OPENCLIP_PRETRAINED,
            device="cpu",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to prepare OpenCLIP model {OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}: {exc}"
        ) from exc


@contextmanager
def timed_step(
    step: int | str,
    label: str,
    experiment: str | None = None,
    model: str | None = None,
    phase: str | None = None,
) -> Generator[None, None, None]:
    """Print a numbered step log with elapsed time."""
    # 긴 모델 추론 단계가 많으므로 콘솔에서 진행 상황과 소요 시간을 확인하게 합니다.
    start = time.perf_counter()
    print(f"{_stage_prefix(step, experiment, model, phase, 'start')} {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{_stage_prefix(step, experiment, model, phase, 'done')} {label} ({elapsed:.2f}s)")


def load_and_normalize_image(image_path: str) -> Any:
    """Load an image, normalize white background, and apply contrast correction."""
    from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

    # 파일이 없거나 PIL이 읽을 수 없는 이미지면 pipeline에서 이해하기 쉬운 오류로 바꿉니다.
    try:
        image = Image.open(image_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to load image: {image_path}") from exc

    # 투명 배경 그림은 흰 배경 위에 합성해 RGB 모델 입력으로 안정화합니다.
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")

    # 손그림의 연한 선을 모델이 더 잘 보도록 대비를 조금 올립니다.
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    return image


def resize_square(image: Any, size: int) -> Any:
    """Resize and center-crop an image to a square canvas."""
    from PIL import Image, ImageOps

    # 모델별 고정 입력 크기에 맞추기 위해 중앙 crop 기반의 정사각형 이미지를 만듭니다.
    return ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)


@lru_cache(maxsize=1)
def get_caption_components() -> tuple[Any, Any]:
    """Load and cache BLIP captioning components once per process."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    # lru_cache 덕분에 첫 호출 이후에는 같은 processor/model 객체를 재사용합니다.
    device = get_device()
    model_source = local_huggingface_model_path(BLIP_CAPTION_MODEL)
    local_only = _local_files_only(model_source)
    processor = BlipProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = BlipForConditionalGeneration.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def get_vqa_components() -> tuple[Any, Any]:
    """Load and cache BLIP VQA components once per process."""
    from transformers import BlipForQuestionAnswering, BlipProcessor

    # 질문-응답 모델은 caption 모델과 별도로 로드해 슬롯별 장면 정보를 추출합니다.
    device = get_device()
    model_source = local_huggingface_model_path(BLIP_VQA_MODEL)
    local_only = _local_files_only(model_source)
    processor = BlipProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = BlipForQuestionAnswering.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def get_openclip_components() -> tuple[Any, Any, Any]:
    """Load and cache OpenCLIP model, preprocessing transform, and tokenizer."""
    import open_clip

    # OpenCLIP은 BLIP 결과 후보를 이미지와 다시 대조하는 검증 단계에 사용됩니다.
    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL,
        pretrained=OPENCLIP_PRETRAINED,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    model.eval()
    return model, preprocess, tokenizer


@lru_cache(maxsize=1)
def get_gpt2_components() -> tuple[Any, Any]:
    """Load and cache GPT-2 generation components once per process."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # GPT-2는 vision 결과를 영어 이야기 초안으로 확장하는 생성 모델입니다.
    device = get_device()
    model_source = local_huggingface_model_path(GPT2_MODEL)
    local_only = _local_files_only(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        # 일부 GPT-2 토크나이저에는 pad token이 없어 eos token을 대신 지정합니다.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


@lru_cache(maxsize=1)
def get_nllb_components() -> tuple[Any, Any]:
    """Load and cache NLLB translation components once per process."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # NLLB는 영어 이야기 초안을 한국어 최종 이야기로 바꾸는 번역 모델입니다.
    device = get_device()
    model_source = local_huggingface_model_path(NLLB_MODEL)
    local_only = _local_files_only(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, src_lang="eng_Latn", local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def get_exaone_components() -> tuple[Any, Any]:
    """Load and cache EXAONE Korean story generation components."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # EXAONE 4.0 1.2B는 한국어를 지원하는 비교적 작은 instruction-capable 모델입니다.
    device = get_device()
    model_source = local_huggingface_model_path(EXAONE_MODEL)
    local_only = _local_files_only(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        local_files_only=local_only,
    )
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


@lru_cache(maxsize=1)
def get_exaone_gguf_components(model_path: str = "") -> Any:
    """Load and cache the quantized EXAONE GGUF model through llama.cpp."""
    from llama_cpp import Llama

    resolved_path = Path(ensure_exaone_gguf_model(model_path))

    return Llama(
        model_path=str(resolved_path),
        n_ctx=2048,
        n_batch=256,
        n_threads=max((os.cpu_count() or 4) - 1, 2),
        verbose=False,
    )


def clear_vision_model_caches() -> None:
    """Release cached vision models before loading a large text generator."""
    import gc
    import torch

    # EXAONE을 같은 프로세스에서 올릴 때 BLIP/OpenCLIP 모델까지 남아 있으면
    # 로컬 CPU/RAM 환경에서 스와핑이 심해질 수 있어 vision 캐시를 비웁니다.
    get_caption_components.cache_clear()
    get_vqa_components.cache_clear()
    get_openclip_components.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
