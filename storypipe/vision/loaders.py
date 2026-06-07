"""[담당 1 · 비전] 비전 모델 로더.

BLIP(caption/VQA), OpenCLIP, Qwen2.5-VL 로딩을 한곳에 모은다.
Qwen 로더는 과거 monster 파일 두 곳에 중복돼 있던 로딩 코드를 단일화한 것이다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from storypipe.common.config import (
    BLIP_CAPTION_MODEL,
    BLIP_VQA_MODEL,
    GPT2_MODEL,
    NLLB_MODEL,
    OPENCLIP_MODEL,
    OPENCLIP_PRETRAINED,
    VISION_MODEL_ID,
)
from storypipe.common.logging import log_model_device, log_stage, timed_step
from storypipe.common.models import (
    _local_files_only,
    _openclip_pretrained_source,
    local_huggingface_model_path,
)
from storypipe.common.runtime import get_device

# GPT2/NLLB 상수는 story 로더가 쓰지만, 위치 일관성을 위해 config에서만 정의한다.
_ = (GPT2_MODEL, NLLB_MODEL)


@lru_cache(maxsize=1)
def get_caption_components() -> tuple[Any, Any]:
    """BLIP captioning 구성요소를 1회 로드/캐시한다."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = get_device()
    model_source = local_huggingface_model_path(BLIP_CAPTION_MODEL)
    local_only = _local_files_only(model_source)
    processor = BlipProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = BlipForConditionalGeneration.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    log_model_device(BLIP_CAPTION_MODEL, device)
    return processor, model


@lru_cache(maxsize=1)
def get_vqa_components() -> tuple[Any, Any]:
    """BLIP VQA 구성요소를 1회 로드/캐시한다."""
    from transformers import BlipForQuestionAnswering, BlipProcessor

    device = get_device()
    model_source = local_huggingface_model_path(BLIP_VQA_MODEL)
    local_only = _local_files_only(model_source)
    processor = BlipProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = BlipForQuestionAnswering.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    log_model_device(BLIP_VQA_MODEL, device)
    return processor, model


@lru_cache(maxsize=1)
def get_openclip_components() -> tuple[Any, Any, Any]:
    """OpenCLIP 모델/전처리/토크나이저를 1회 로드/캐시한다."""
    import open_clip

    device = get_device()
    pretrained_source = _openclip_pretrained_source()
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL,
        pretrained=pretrained_source,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    model.eval()
    log_model_device(f"OpenCLIP {OPENCLIP_MODEL}", device)
    return model, preprocess, tokenizer


def clear_vision_model_caches() -> None:
    """큰 텍스트 생성기를 올리기 전에 비전 모델 캐시를 비운다(CPU/RAM 절약)."""
    import gc

    import torch

    get_caption_components.cache_clear()
    get_vqa_components.cache_clear()
    get_openclip_components.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _snapshot_dir(model_cache: Path | str) -> Path | str:
    """로컬 HF 모델 디렉터리에서 실제 스냅샷 폴더를 찾는다."""
    if not isinstance(model_cache, Path):
        return model_cache
    if model_cache.exists() and (model_cache / "config.json").exists():
        return model_cache
    snapshots = model_cache / "snapshots"
    if snapshots.exists():
        dirs = sorted([path for path in snapshots.iterdir() if path.is_dir()])
        if dirs:
            return dirs[-1]
    return VISION_MODEL_ID


def load_qwen_model(max_pixels: int) -> tuple[Any, Any, str]:
    """Qwen2.5-VL 모델/프로세서/디바이스를 로드한다(CUDA 실패 시 CPU 폴백).

    과거 monster의 장면 추출(_ensure_scenes)과 콜라주 분석(_run_qwen_collage_analysis)에
    중복돼 있던 로딩 로직을 단일화한 것이다. ``max_pixels``로 프로세서 입력 해상도를 조절한다.
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_source = _snapshot_dir(local_huggingface_model_path(VISION_MODEL_ID))
    log_stage(f"loading vision model from {model_source}", step="Qwen-load", model=VISION_MODEL_ID)
    local_only = isinstance(model_source, Path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with timed_step("Qwen-load", "Qwen2.5-VL model load", model=VISION_MODEL_ID):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float16 if device == "cuda" else "auto",
            local_files_only=local_only,
        )
        if device == "cuda":
            try:
                model = model.to("cuda")
            except Exception as exc:
                device = "cpu"
                log_stage(
                    f"Qwen CUDA move failed; reloading on CPU: {exc}",
                    step="Qwen-device",
                    model=VISION_MODEL_ID,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_source,
                    torch_dtype="auto",
                    local_files_only=local_only,
                )
        model.eval()
        log_model_device(VISION_MODEL_ID, device, phase="vision")
        processor = AutoProcessor.from_pretrained(
            model_source,
            local_files_only=local_only,
            max_pixels=max_pixels,
        )
    return model, processor, device
