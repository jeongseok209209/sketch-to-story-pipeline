"""[공통 토대] 디바이스/런타임 감지.

재현성을 위해 런타임 중 패키지를 몰래 설치하지 않는다(과거 --force-reinstall이 환경 파손의
주원인이었음). 여기서는 *검증·안내*만 하고, 실제 설치는 ``storypipe doctor``로 일원화한다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from storypipe.common.config import TORCH_MIN_SAFE_VERSION
from storypipe.common.logging import log_stage


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
