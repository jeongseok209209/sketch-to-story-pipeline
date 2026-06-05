"""Shared utilities for Experiment A."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator


BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"
GPT2_MODEL = "gpt2-medium"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
OPENCLIP_MODEL = "ViT-H-14"
OPENCLIP_PRETRAINED = "laion2b_s32b_b79k"
EXAONE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXAONE_GGUF_PATH = str(
    PROJECT_ROOT / ".local_models" / "exaone" / "EXAONE-4.0-1.2B-IQ4_XS.gguf"
)
LLAMA_CLI_FILENAME = "llama-cli.exe" if os.name == "nt" else "llama-cli"
LLAMA_CLI_PATH = str(
    PROJECT_ROOT / ".local_tools" / "llama.cpp" / "build" / "bin" / LLAMA_CLI_FILENAME
)


def get_device() -> Any:
    """Return CUDA when available, otherwise CPU."""
    import torch

    # GPU가 있으면 모델 추론 속도를 위해 CUDA를 우선 사용하고, 없으면 CPU로 동작합니다.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def timed_step(step: int, label: str) -> Generator[None, None, None]:
    """Print a numbered step log with elapsed time."""
    # 긴 모델 추론 단계가 많으므로 콘솔에서 진행 상황과 소요 시간을 확인하게 합니다.
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
    processor = BlipProcessor.from_pretrained(BLIP_CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_CAPTION_MODEL)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def get_vqa_components() -> tuple[Any, Any]:
    """Load and cache BLIP VQA components once per process."""
    from transformers import BlipForQuestionAnswering, BlipProcessor

    # 질문-응답 모델은 caption 모델과 별도로 로드해 슬롯별 장면 정보를 추출합니다.
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
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GPT2_MODEL)
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
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
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
    tokenizer = AutoTokenizer.from_pretrained(EXAONE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        EXAONE_MODEL,
        dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
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

    resolved_path = Path(
        model_path
        or os.environ.get("EXAONE_GGUF_MODEL_PATH", "")
        or DEFAULT_EXAONE_GGUF_PATH
    ).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(
            "EXAONE GGUF model file not found. Set EXAONE_GGUF_MODEL_PATH "
            f"or download it to: {resolved_path}"
        )

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
