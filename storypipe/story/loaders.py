"""[담당 2 · 스토리] 언어 모델 로더.

GPT-2 / NLLB(실험 A 베이스라인), EXAONE(HF), EXAONE GGUF(llama-cpp-python) 로딩.
과거 utils.get_exaone_gguf_components(미사용 llama-cpp-python 경로)를 정식 경로로 승격했다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from storypipe.common.config import EXAONE_MODEL, GPT2_MODEL, NLLB_MODEL
from storypipe.common.logging import log_model_device
from storypipe.common.models import (
    _local_files_only,
    ensure_exaone_gguf_model,
    local_huggingface_model_path,
)
from storypipe.common.runtime import configured_llama_gpu_layers, get_device


@lru_cache(maxsize=1)
def get_gpt2_components() -> tuple[Any, Any]:
    """GPT-2 생성 구성요소를 1회 로드/캐시한다(실험 A 영어 초안)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    model_source = local_huggingface_model_path(GPT2_MODEL)
    local_only = _local_files_only(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    log_model_device(GPT2_MODEL, device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


@lru_cache(maxsize=1)
def get_nllb_components() -> tuple[Any, Any]:
    """NLLB 번역 구성요소를 1회 로드/캐시한다(영→한)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = get_device()
    model_source = local_huggingface_model_path(NLLB_MODEL)
    local_only = _local_files_only(model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, src_lang="eng_Latn", local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    log_model_device(NLLB_MODEL, device)
    return tokenizer, model


@lru_cache(maxsize=1)
def get_exaone_components() -> tuple[Any, Any]:
    """EXAONE(HF transformers) 한국어 생성 구성요소를 1회 로드/캐시한다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    log_model_device(EXAONE_MODEL, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


@lru_cache(maxsize=1)
def get_exaone_gguf_components(model_path: str = "") -> Any:
    """양자화 EXAONE GGUF를 llama-cpp-python으로 1회 로드/캐시한다.

    재현성: 기본 CPU(n_gpu_layers=0). NVIDIA GPU에서 가속하려면 ``LLAMA_GPU_LAYERS`` 지정.
    n_ctx는 ``EXAONE_N_CTX``(기본 8192)로 시퀀스 스토리(8192 컨텍스트)까지 수용한다.
    """
    from llama_cpp import Llama

    resolved_path = Path(ensure_exaone_gguf_model(model_path))
    n_ctx = int(os.environ.get("EXAONE_N_CTX", "8192"))
    n_gpu_layers = configured_llama_gpu_layers()
    return Llama(
        model_path=str(resolved_path),
        n_ctx=n_ctx,
        n_batch=256,
        n_threads=max((os.cpu_count() or 4) - 1, 2),
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
