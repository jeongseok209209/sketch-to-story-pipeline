"""Shared utilities for Experiment A."""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator


BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"
GPT2_MODEL = "gpt2-medium"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
OPENCLIP_MODEL = "ViT-H-14"
OPENCLIP_PRETRAINED = "laion2b_s32b_b79k"


def get_device() -> Any:
    """Return CUDA when available, otherwise CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def timed_step(step: int, label: str) -> Generator[None, None, None]:
    """Print a numbered step log with elapsed time."""
    start = time.perf_counter()
    print(f"[{step}] {label} ...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{step}] {label} done ({elapsed:.2f}s)")


def load_and_normalize_image(image_path: str) -> Any:
    """Load an image, normalize white background, and apply contrast correction."""
    from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(image_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to load image: {image_path}") from exc

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
    """Resize and center-crop an image to a square canvas."""
    from PIL import Image, ImageOps

    return ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)


@lru_cache(maxsize=1)
def get_caption_components() -> tuple[Any, Any]:
    """Load and cache BLIP captioning components once per process."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = get_device()
    processor = BlipProcessor.from_pretrained(BLIP_CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_CAPTION_MODEL)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def get_vqa_components() -> tuple[Any, Any]:
    """Load and cache BLIP VQA components once per process."""
    from transformers import BlipForQuestionAnswering, BlipProcessor

    device = get_device()
    processor = BlipProcessor.from_pretrained(BLIP_VQA_MODEL)
    model = BlipForQuestionAnswering.from_pretrained(BLIP_VQA_MODEL)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def get_openclip_components() -> tuple[Any, Any, Any]:
    """Load and cache OpenCLIP model, preprocessing transform, and tokenizer."""
    import open_clip

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

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


@lru_cache(maxsize=1)
def get_nllb_components() -> tuple[Any, Any]:
    """Load and cache NLLB translation components once per process."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.to(device)
    model.eval()
    return tokenizer, model
