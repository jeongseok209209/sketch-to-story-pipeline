"""[공통 토대] Hugging Face / GGUF 모델 자산 확보.

오픈소스 모델은 첫 사용 시(또는 ``storypipe doctor`` 실행 시) 자동 다운로드된다.
``config.hf_revision``로 정확한 revision을 고정해 재현성을 높일 수 있다.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from storypipe.common.config import (
    DEFAULT_EXAONE_GGUF_PATH,
    EXAONE_GGUF_FILENAME,
    EXAONE_GGUF_REPO_ID,
    HF_PREFLIGHT_IGNORE_PATTERNS,
    LOCAL_HF_MODEL_DIR,
    OPENCLIP_HF_CACHE_REPO,
    OPENCLIP_MODEL,
    OPENCLIP_PRETRAINED,
    OPENCLIP_WEIGHTS_FILENAME,
    hf_revision,
)
from storypipe.common.logging import log_stage


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
