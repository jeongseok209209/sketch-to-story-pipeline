"""[담당 1 · 비전] BLIP/OpenCLIP 인식(실험 A) + Qwen2.5-VL 장면/콜라주 추출(C~J) + 비전 로더."""

from __future__ import annotations


# ╔══ vision/loaders.py ══╗


from functools import lru_cache
from pathlib import Path
from typing import Any

from common import (
    BLIP_CAPTION_MODEL,
    BLIP_VQA_MODEL,
    GPT2_MODEL,
    NLLB_MODEL,
    OPENCLIP_MODEL,
    OPENCLIP_PRETRAINED,
    VISION_MODEL_ID,
)
from common import log_model_device, log_stage, timed_step
from common import (
    _local_files_only,
    _openclip_pretrained_source,
    local_huggingface_model_path,
)
from common import get_device

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

# ╔══ vision/blip_clip.py ══╗


import re
from pathlib import Path
from typing import Any

from common import load_and_normalize_image, resize_square
from common import timed_step
from common import get_device


VQA_QUESTIONS = {
    # BLIP-VQA가 이미지에서 이야기 생성에 필요한 기본 장면 정보를 뽑도록 묻는 질문들입니다.
    "who": "Who is in this picture?",
    "actions": "What is happening in this picture?",
    "scene": "Where is this scene?",
    "mood": "What is the mood of this picture?",
}


def _decode_first(processor: Any, output_ids: Any) -> str:
    """Decode the first generated sequence and strip whitespace."""
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def _caption_image(image: Any) -> str:
    """Run BLIP captioning with the required prefix."""
    import torch

    # BLIP caption 모델은 비교적 무거우므로 utils에서 한 번만 로드해 재사용합니다.
    processor, model = get_caption_components()
    device = get_device()

    # 손그림이라는 맥락을 prefix로 넣어 모델이 사진보다 그림 설명에 맞춰 caption을 만들게 합니다.
    inputs = processor(
        images=image,
        text="a children's drawing of",
        return_tensors="pt",
    ).to(device)

    # 추론 전용 실행으로 그래디언트 계산을 끄고, beam search로 안정적인 캡션을 얻습니다.
    with torch.inference_mode():
        output = model.generate(**inputs, num_beams=5, max_length=50)
    caption = _decode_first(processor, output)
    if not caption:
        raise ValueError("BLIP captioning returned an empty result.")
    return caption


def _answer_vqa(image: Any) -> dict[str, str]:
    """Ask the required BLIP-VQA questions and collect short answers."""
    import torch

    # VQA 모델도 캐시된 구성요소를 사용해 질문마다 다시 로드하지 않습니다.
    processor, model = get_vqa_components()
    device = get_device()
    answers: dict[str, str] = {}

    # 같은 이미지에 여러 질문을 던져 주인공, 행동, 장소, 분위기를 슬롯 형태로 수집합니다.
    for key, question in VQA_QUESTIONS.items():
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=20)
        answers[key] = _decode_first(processor, output)

    return answers


def _extract_candidates(text: str) -> list[str]:
    """Extract unique lowercase alphabetic candidate words from model text."""
    # caption과 VQA 답변을 합친 문장에서 OpenCLIP으로 검증할 후보 단어만 추립니다.
    words = re.findall(r"[a-zA-Z]+", text.lower())
    candidates = list(set(words))
    if not candidates:
        raise ValueError("No candidate words were extracted from BLIP outputs.")
    return candidates


def _score_candidates(image: Any, candidates: list[str]) -> dict[str, float]:
    """Score candidate words with OpenCLIP cosine similarity."""
    import torch

    # OpenCLIP은 이미지 특징과 텍스트 프롬프트 특징을 같은 임베딩 공간에서 비교합니다.
    model, preprocess, tokenizer = get_openclip_components()
    device = get_device()

    # 각 후보 단어를 "아이의 그림" 프롬프트로 감싸 caption 후보와 이미지의 유사도를 측정합니다.
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    prompts = [f"a child's drawing of a {word}" for word in candidates]
    text_tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        # 코사인 유사도를 계산하기 위해 이미지/텍스트 특징 벡터를 정규화합니다.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)

    return {
        word: float(score.detach().cpu().item())
        for word, score in zip(candidates, similarities, strict=True)
    }


def _recognize_impl(
    image_path: str,
    clip_threshold: float = 0.22,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recognize sketch concepts and collect stage-by-stage records."""
    # steps에는 리포트/디버깅용으로 각 단계의 입력, 설정, 출력 요약을 저장합니다.
    steps: dict[str, Any] = {}
    source = Path(image_path)

    with timed_step(1, "image input", model="input"):
        # 투명 배경이나 낮은 대비의 손그림도 모델이 보기 좋도록 공통 전처리를 적용합니다.
        image = load_and_normalize_image(image_path)
        steps["01_image_input"] = {
            "step": 1,
            "name": "손그림 이미지 입력",
            "image_id": source.name,
            "image_path": str(source.resolve()),
            "image_mode": image.mode,
            "image_size": list(image.size),
        }

    with timed_step(2, "preprocessing for BLIP and OpenCLIP", model="BLIP/OpenCLIP"):
        # BLIP과 OpenCLIP은 권장 입력 크기가 달라 모델별 사각 이미지로 따로 맞춥니다.
        blip_image = resize_square(image, 384)
        clip_image = resize_square(image, 224)
        steps["02_preprocessing"] = {
            "step": 2,
            "name": "전처리",
            "blip_size": list(blip_image.size),
            "openclip_size": list(clip_image.size),
            "white_background_normalization": True,
            "contrast_correction": "ImageOps.autocontrast + contrast 1.15",
        }

    with timed_step(3, "BLIP captioning", model="Salesforce/blip-image-captioning-large"):
        raw_caption = _caption_image(blip_image)
        steps["03_blip_captioning"] = {
            "step": 3,
            "name": "BLIP captioning",
            "output": "자유 캡션",
            "prefix": "a children's drawing of",
            "num_beams": 5,
            "max_length": 50,
            "raw_caption": raw_caption,
        }

    with timed_step(4, "BLIP-VQA", model="Salesforce/blip-vqa-base"):
        answers = _answer_vqa(blip_image)
        steps["04_blip_vqa"] = {
            "step": 4,
            "name": "BLIP-VQA 슬롯 질의",
            "output": "{who, action, scene, mood}",
            "questions": VQA_QUESTIONS,
            "answers": answers,
        }

    with timed_step(5, "candidate word extraction", model="BLIP text outputs"):
        # BLIP이 만든 자유 캡션과 VQA 답변을 모두 합쳐 후보 개념의 재료로 사용합니다.
        combined = " ".join([raw_caption, *answers.values()])
        candidates = _extract_candidates(combined)

    with timed_step(6, "OpenCLIP candidate scoring", model="OpenCLIP ViT-H-14/laion2b_s32b_b79k"):
        # 추출된 후보가 실제 이미지와 잘 맞는지 OpenCLIP으로 한 번 더 검증합니다.
        candidate_scores = _score_candidates(clip_image, candidates)
        steps["05_openclip_concept_scoring"] = {
            "step": 5,
            "name": "OpenCLIP 개념 점수화",
            "output": "{concept: similarity}",
            "source_text": combined,
            "candidate_extraction_regex": r"[a-zA-Z]+",
            "candidate_words": candidates,
            "prompt_template": "a child's drawing of a {word}",
            "clip_threshold": clip_threshold,
            "scores": candidate_scores,
        }

    with timed_step(7, "vision_json creation", model="BLIP/OpenCLIP merged vision_json"):
        # threshold 이상인 후보만 최종 object로 채택해 이후 이야기 생성에 넘깁니다.
        object_scores = {
            word: score
            for word, score in candidate_scores.items()
            if score >= clip_threshold
        }
        objects = list(object_scores.keys())
        # object가 충분히 잡혔는지에 따라 결과 신뢰도를 단순 분류합니다.
        confidence = "high" if len(objects) >= 3 else "low"

    # downstream 단계가 모델 세부사항을 몰라도 사용할 수 있는 통합 vision JSON입니다.
    vision = {
        "objects": objects,
        "object_scores": object_scores,
        "who": answers.get("who", ""),
        "actions": answers.get("actions", ""),
        "scene": answers.get("scene", ""),
        "mood": answers.get("mood", ""),
        "raw_caption": raw_caption,
        "confidence": confidence,
    }
    steps["06_vision_json"] = {
        "step": 6,
        "name": "교차 검증 + 통합",
        "output": "vision_json",
        "vision": vision,
    }
    return vision, steps


def recognize(image_path: str, clip_threshold: float = 0.22) -> dict[str, Any]:
    """Recognize sketch concepts and scene hints from an input PNG/JPG image."""
    vision, _ = _recognize_impl(image_path, clip_threshold=clip_threshold)
    return vision


def recognize_with_steps(
    image_path: str,
    clip_threshold: float = 0.22,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recognize sketch concepts and return both vision JSON and stage records."""
    return _recognize_impl(image_path, clip_threshold=clip_threshold)

# ╔══ vision/qwen_scenes.py ══╗


import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from common import (
    COLLAGE_FILENAME,
    COMMON_OUTPUT_DIR,
    IMAGE_EXTENSIONS,
    INPUT_DIR,
    OUTPUT_ROOT,
    SHARED_DIR,
    STORY_CAPTION_FILENAME,
    VISION_MODEL_ID,
    experiment_dirs as _experiment_dirs,
)
from common import RESIZED_DIR as _DEFAULT_RESIZED_DIR
from common import log_stage, set_step_context, timed_step

# Qwen 입력 해상도(개별 장면은 작게, 콜라주는 덜 공격적으로 축소)
QWEN_IMAGE_MAX_SIDE = 384
QWEN_MAX_PIXELS = QWEN_IMAGE_MAX_SIDE * QWEN_IMAGE_MAX_SIDE
QWEN_COLLAGE_MAX_SIDE = 1600
QWEN_COLLAGE_MAX_PIXELS = QWEN_COLLAGE_MAX_SIDE * QWEN_COLLAGE_MAX_SIDE

# _ensure_scenes가 실험별 경로로 교체하는 모듈 전역(_prepare_image가 읽음)
RESIZED_DIR = _DEFAULT_RESIZED_DIR


# ─────────────────────────────────────────────────────────────────────────────
# 아래 본문은 기존 run_experiments_cd_qwen3b.py의 Qwen 비전 구역에서 이동한 코드.
# ─────────────────────────────────────────────────────────────────────────────
def _iter_images(directory: Path) -> list[Path]:
    numbered: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem.isdigit():
                numbered[int(path.stem)] = path
    return [numbered[key] for key in sorted(numbered)]


def _read_story_caption(input_dir: Path) -> str:
    caption_path = input_dir / STORY_CAPTION_FILENAME
    if not caption_path.exists():
        raise FileNotFoundError(
            f"Experiment H/I/J requires {STORY_CAPTION_FILENAME} in the selected story folder: {caption_path}"
        )
    caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"Experiment H/I/J requires a non-empty story caption: {caption_path}")
    return caption


def _resolve_collage_path(input_dir: Path) -> Path:
    candidates = [
        input_dir / COLLAGE_FILENAME,
        input_dir / "collages" / input_dir.name / COLLAGE_FILENAME,
        input_dir.parent / "collages" / input_dir.name / COLLAGE_FILENAME,
    ]
    for collage_path in candidates:
        if collage_path.exists():
            return collage_path
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Experiment J requires a story collage inside the input tree. "
        f"Expected: {expected}"
    )


def _prepare_image(image_path: Path) -> Path:
    """Resize large input drawings to reduce Qwen CPU inference time."""
    from PIL import Image

    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    source_key = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    target = RESIZED_DIR / f"{image_path.stem}_{source_key}.jpg"
    if target.exists() and target.stat().st_mtime >= image_path.stat().st_mtime:
        return target
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((QWEN_IMAGE_MAX_SIDE, QWEN_IMAGE_MAX_SIDE))
        image.save(target, format="JPEG", quality=90)
    return target


def _prepare_collage_image(image_path: Path, output_dir: Path) -> Path:
    """Resize the story collage less aggressively than individual scene images."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    source_key = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
    target = output_dir / f"{image_path.stem}_{source_key}.jpg"
    if target.exists() and target.stat().st_mtime >= image_path.stat().st_mtime:
        return target
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((QWEN_COLLAGE_MAX_SIDE, QWEN_COLLAGE_MAX_SIDE))
        image.save(target, format="JPEG", quality=92)
    return target


def _prompt(index: int) -> str:
    return (
        "당신은 아이 손그림을 동화 장면으로 읽는 시각 인식 모델입니다.\n"
        "반드시 한국어 JSON만 출력하세요. 마크다운과 영어 설명은 쓰지 마세요.\n"
        "확실하지 않은 대상은 '~처럼 보임'이라고 적고 억지로 단정하지 마세요.\n"
        "그림 속 인물, 동물, 사물, 위치, 색감, 표정, 분위기를 근거로 설명하세요.\n"
        f"이 이미지는 전체 이야기의 {index}번째 그림입니다.\n\n"
        "{\n"
        '  "scene_summary": "그림에 보이는 내용을 아이도 이해할 수 있게 2문장으로 설명",\n'
        '  "characters": ["사람/동물/말하는 사물"],\n'
        '  "objects": ["중요한 사물"],\n'
        '  "setting": "장소 또는 배경",\n'
        '  "mood": "밝음/쓸쓸함/신비로움/조심스러움 등",\n'
        '  "emotion": "주인공이 느낄 법한 감정",\n'
        '  "story_role": "이 그림이 이야기에서 맡을 역할",\n'
        '  "uncertain": "확실하지 않은 부분"\n'
        "}"
    )


def _prompt_e_visual_cot(index: int) -> str:
    return (
        "당신은 아이 손그림을 동화 장면으로 읽는 시각 인식 모델입니다.\n"
        "반드시 한국어 JSON만 출력하세요. 마크다운과 영어 설명은 쓰지 마세요.\n"
        "먼저 내부적으로 인물/동물/사물, 색/위치/행동, 확실한 단서와 불확실한 단서를 점검하세요.\n"
        "단, 이 점검 과정은 출력하지 말고 최종 JSON 필드에만 반영하세요.\n"
        "확실하지 않은 대상은 '~처럼 보임'이라고 적고 억지로 단정하지 마세요.\n"
        "동물의 종류가 확실하지 않으면 토끼, 새, 뱀처럼 단정하지 말고 '동물'이라고 쓰세요.\n"
        "그림 속 인물, 동물, 사물, 위치, 색감, 표정, 분위기를 근거로 설명하세요.\n"
        "scene_summary는 보이는 내용만 짧게 1~2문장으로 설명하세요.\n"
        "최종 출력은 한국어 JSON 객체 하나만 출력하세요.\n"
        f"이 이미지는 전체 이야기의 {index}번째 그림입니다.\n\n"
        "{\n"
        '  "scene_summary": "그림에 보이는 내용을 아이도 이해할 수 있게 2문장으로 설명",\n'
        '  "characters": ["사람/동물/말하는 사물"],\n'
        '  "objects": ["중요한 사물"],\n'
        '  "setting": "장소 또는 배경",\n'
        '  "mood": "밝음/쓸쓸함/신비로움/조심스러움 등",\n'
        '  "emotion": "주인공이 느낄 법한 감정",\n'
        '  "story_role": "이 그림이 이야기에서 맡을 역할",\n'
        '  "uncertain": "확실하지 않은 부분"\n'
        "}"
    )


def _prompt_f_fairy_tale_image_analyst(index: int) -> str:
    return (
        "당신은 아이 손그림을 동화 장면으로 해석하는 동화 그림 분석가입니다.\n"
        "반드시 한국어 JSON만 출력하세요. 마크다운과 영어 설명은 쓰지 마세요.\n"
        "그림을 동화의 재료로 읽되, 실제로 보이지 않는 사건이나 관계는 꾸며내지 마세요.\n"
        "먼저 내부적으로 인물/동물/사물, 색/위치/행동, 표정/분위기, 확실한 단서와 불확실한 단서를 점검하세요.\n"
        "단, 이 점검 과정은 출력하지 말고 최종 JSON 필드에만 반영하세요.\n"
        "확실하지 않은 대상은 '~처럼 보임'이라고 적고 억지로 단정하지 마세요.\n"
        "동물의 종류가 확실하지 않으면 토끼, 새, 뱀처럼 단정하지 말고 '동물'이라고 쓰세요.\n"
        "scene_summary는 보이는 내용을 동화 그림 분석가답게 짧게 1~2문장으로 설명하세요.\n"
        "최종 출력은 한국어 JSON 객체 하나만 출력하세요.\n"
        f"이 이미지는 전체 이야기의 {index}번째 그림입니다.\n\n"
        "{\n"
        '  "scene_summary": "그림에 보이는 내용을 아이도 이해할 수 있게 2문장으로 설명",\n'
        '  "characters": ["사람/동물/말하는 사물"],\n'
        '  "objects": ["중요한 사물"],\n'
        '  "setting": "장소 또는 배경",\n'
        '  "mood": "밝음/쓸쓸함/신비로움/조심스러움 등",\n'
        '  "emotion": "주인공이 느낄 법한 감정",\n'
        '  "story_role": "이 그림이 이야기에서 맡을 역할",\n'
        '  "uncertain": "확실하지 않은 부분"\n'
        "}"
    )


def _prompt_g_cot_persona(index: int) -> str:
    return (
        "당신은 아이 손그림을 동화 장면으로 해석하는 동화 그림 분석가입니다.\n"
        "반드시 한국어 JSON만 출력하세요. 마크다운과 영어 설명은 쓰지 마세요.\n"
        "그림을 동화의 재료로 읽되, 실제로 보이지 않는 사건이나 관계는 꾸며내지 마세요.\n"
        "먼저 내부적으로 인물/동물/사물, 색/위치/행동, 표정/분위기, 확실한 단서와 불확실한 단서를 점검하세요.\n"
        "단, 이 점검 과정은 출력하지 말고 최종 JSON 필드에만 반영하세요.\n"
        "확실하지 않은 대상은 '~처럼 보임'이라고 적고 억지로 단정하지 마세요.\n"
        "동물의 종류가 확실하지 않으면 토끼, 새, 뱀처럼 단정하지 말고 '동물'이라고 쓰세요.\n"
        "scene_summary는 보이는 내용을 동화 그림 분석가답게 짧게 1~2문장으로 설명하세요.\n"
        "최종 출력은 한국어 JSON 객체 하나만 출력하세요.\n"
        f"이 이미지는 전체 이야기의 {index}번째 그림입니다.\n\n"
        "{\n"
        '  "scene_summary": "그림에 보이는 내용을 아이도 이해할 수 있게 2문장으로 설명",\n'
        '  "characters": ["사람/동물/말하는 사물"],\n'
        '  "objects": ["중요한 사물"],\n'
        '  "setting": "장소 또는 배경",\n'
        '  "mood": "밝음/쓸쓸함/신비로움/조심스러움 등",\n'
        '  "emotion": "주인공이 느낄 법한 감정",\n'
        '  "story_role": "이 그림이 이야기에서 맡을 역할",\n'
        '  "uncertain": "확실하지 않은 부분"\n'
        "}"
    )


def _prompt_j_collage(story_caption: str = "") -> str:
    caption_section = ""
    if story_caption.strip():
        caption_section = (
            "Story caption text is provided as weak whole-story context. "
            "Use it only to understand the intended broad flow, and keep visible collage evidence primary.\n"
            f"story_caption:\n{story_caption.strip()}\n\n"
        )
    return (
        "Experiment J collage analysis: You are a careful visual story analyst for a 2x5 collage of child drawings.\n"
        "The collage contains Scene 1 through Scene 10 labels. Read the panels in numeric order only.\n"
        "Analyze the whole sequence at once, but do not invent details that are not visible.\n"
        "Use the collage as a broad continuity map. Individual scene JSON will still be the main evidence later.\n"
        f"{caption_section}"
        "Output exactly one Korean JSON object only. Do not add markdown, explanations, or code fences.\n"
        "Required shape:\n"
        "{\n"
        '  "overall_story_arc": "10장 전체가 어떻게 시작-전개-마무리로 이어지는지 3문장",\n'
        '  "scene_order_summary": ["1번부터 10번까지 각 장면을 한 문장씩 요약"],\n'
        '  "recurring_characters": ["반복 등장하는 인물/동물"],\n'
        '  "visual_continuity": ["장면 사이에 이어지는 시각 단서"],\n'
        '  "turning_points": ["이야기의 변화 지점"],\n'
        '  "ending_read": "마지막 장면이 어떤 정서로 끝나는지",\n'
        '  "uncertainty_notes": ["작게 보이거나 확실하지 않은 부분"]\n'
        "}"
    )


def _j_scene_prior_context(
    index: int,
    story_caption: str,
    collage_analysis: dict[str, Any] | None,
) -> str:
    scene_order_summary: list[Any] = []
    if collage_analysis and isinstance(collage_analysis.get("scene_order_summary"), list):
        scene_order_summary = collage_analysis["scene_order_summary"]
    current_scene_hint = ""
    if 1 <= index <= len(scene_order_summary):
        current_scene_hint = str(scene_order_summary[index - 1]).strip()
    context = {
        "priority": "weak_reference_only",
        "rule": (
            "Use this only to clarify scene order, story_role, and smooth connection to nearby scenes. "
            "Do not use it as evidence for visual facts."
        ),
        "not_allowed_for": ["characters", "objects", "setting", "visible_actions"],
        "caption_hint": story_caption.strip()[:300],
        "overall_arc_hint": "",
        "current_scene_collage_hint": current_scene_hint,
        "recurring_character_hints": [],
        "continuity_hints": [],
        "uncertainty_notes": [],
    }
    if collage_analysis:
        context["overall_arc_hint"] = str(collage_analysis.get("overall_story_arc", "")).strip()
        recurring = collage_analysis.get("recurring_characters")
        if isinstance(recurring, list):
            context["recurring_character_hints"] = [str(item).strip() for item in recurring[:3] if str(item).strip()]
        continuity = collage_analysis.get("visual_continuity")
        if isinstance(continuity, list):
            context["continuity_hints"] = [str(item).strip() for item in continuity[:2] if str(item).strip()]
        uncertainty = collage_analysis.get("uncertainty_notes")
        if isinstance(uncertainty, list):
            context["uncertainty_notes"] = [str(item).strip() for item in uncertainty[:2] if str(item).strip()]
    return json.dumps(context, ensure_ascii=False, indent=2)


def _prompt_j_scene_with_prior_context(
    index: int,
    story_caption: str,
    collage_analysis: dict[str, Any] | None,
) -> str:
    return (
        "Experiment J per-scene image analysis with weak prior context.\n"
        "Before analyzing this single image, you reviewed the story caption and the 2x5 collage of all 10 scenes.\n"
        "The prior context is weak reference material, not visual evidence and not a plot checklist.\n"
        "Use it only to clarify the whole-story flow, the scene's rough role, and smooth continuity with nearby scenes.\n"
        "It may lightly influence story_role or uncertainty notes, but it must not decide the concrete visual content.\n"
        "Do not add an object, character, action, setting, emotion, or story_role only because it appears in the prior context.\n"
        "For characters, objects, setting, and visible actions, use only what is visible in the current image.\n"
        "Do not copy wording from the prior context into the JSON fields.\n"
        "Priority order: current image pixels > visible current-scene clues > uncertainty marking > weak prior context.\n"
        "If prior context and the current image conflict, trust the current image and mark uncertain details as uncertain.\n"
        f"current_scene_index: {index}\n"
        f"weak_story_flow_hint:\n{_j_scene_prior_context(index, story_caption, collage_analysis)}\n\n"
        f"{_prompt_g_cot_persona(index)}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"scene_summary": text.strip(), "raw_parse_error": True}
    return value if isinstance(value, dict) else {"scene_summary": text.strip()}


def _listify(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [part.strip() for part in re.split(r"[,/，、]", str(value)) if part.strip()]


def _normalize_scene(index: int, image_path: Path, payload: dict[str, Any], raw: str) -> dict[str, Any]:
    return {
        "scene_index": index,
        "image_id": image_path.name,
        "image_path": str(image_path.resolve()),
        "scene_summary": str(payload.get("scene_summary", "")).strip(),
        "characters": _listify(payload.get("characters")),
        "objects": _listify(payload.get("objects")),
        "setting": str(payload.get("setting", "")).strip(),
        "mood": str(payload.get("mood", "")).strip(),
        "emotion": str(payload.get("emotion", "")).strip(),
        "story_role": str(payload.get("story_role", "")).strip(),
        "uncertain": str(payload.get("uncertain", "")).strip(),
        "raw_response": raw.strip(),
    }


def _normalize_collage_analysis(collage_path: Path, payload: dict[str, Any], raw: str) -> dict[str, Any]:
    scene_summaries = payload.get("scene_order_summary")
    if not isinstance(scene_summaries, list):
        scene_summaries = _listify(scene_summaries)
    return {
        "collage_image_id": collage_path.name,
        "collage_image_path": str(collage_path.resolve()),
        "overall_story_arc": str(payload.get("overall_story_arc", "")).strip(),
        "scene_order_summary": [str(item).strip() for item in scene_summaries if str(item).strip()],
        "recurring_characters": _listify(payload.get("recurring_characters")),
        "visual_continuity": _listify(payload.get("visual_continuity")),
        "turning_points": _listify(payload.get("turning_points")),
        "ending_read": str(payload.get("ending_read", "")).strip(),
        "uncertainty_notes": _listify(payload.get("uncertainty_notes")),
        "raw_response": raw.strip(),
    }


def _run_qwen_scene(
    model: Any,
    processor: Any,
    image_path: Path,
    index: int,
    prompt_builder: Callable[[int], str] = _prompt,
    max_new_tokens: int = 220,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    qwen_image_path = _prepare_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(qwen_image_path.resolve())},
                {"type": "text", "text": prompt_builder(index)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _normalize_scene(index, image_path, _extract_json(raw), raw)


def _run_qwen_collage_analysis(
    image_path: Path,
    output_dir: Path,
    story_caption: str = "",
    max_new_tokens: int = 520,
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    # 모델 로딩은 vision/loaders.py로 단일화(과거 중복 코드 제거).
    model, processor, device = load_qwen_model(QWEN_COLLAGE_MAX_PIXELS)

    prepared_path = _prepare_collage_image(image_path, output_dir / "_resized_collage")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(prepared_path.resolve())},
                {"type": "text", "text": _prompt_j_collage(story_caption)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to("cuda")
    with timed_step("J-collage", "Qwen collage sequence analysis", model=VISION_MODEL_ID):
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    analysis = _normalize_collage_analysis(image_path, _extract_json(raw), raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qwen25_vl_3b_collage_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "collage_raw.txt").write_text(raw, encoding="utf-8")
    return analysis


def _ensure_scenes(
    input_dir: Path = INPUT_DIR,
    common_output_dir: Path = COMMON_OUTPUT_DIR,
    shared_dir: Path = SHARED_DIR,
    resized_dir: Path = RESIZED_DIR,
    prompt_builder: Callable[[int], str] = _prompt,
    qwen_max_new_tokens: int = 220,
) -> list[dict[str, Any]]:
    global RESIZED_DIR

    RESIZED_DIR = resized_dir
    images = _iter_images(input_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    scenes_path = shared_dir / "scenes.json"
    scenes: list[dict[str, Any]] = []

    if not images:
        raise ValueError(f"No PNG/JPG images found in input directory: {input_dir}")

    # 모델 로딩은 vision/loaders.py로 단일화(과거 중복 코드 제거).
    model, processor, device = load_qwen_model(QWEN_MAX_PIXELS)
    for index, image_path in enumerate(images, start=1):
        log_stage(f"scene {index}: {image_path.name}", step=f"Qwen-{index:02d}", model=VISION_MODEL_ID)
        with timed_step(f"Qwen-{index:02d}", "Qwen scene JSON recognition", model=VISION_MODEL_ID):
            scene = _run_qwen_scene(
                model,
                processor,
                image_path,
                index,
                prompt_builder=prompt_builder,
                max_new_tokens=qwen_max_new_tokens,
                device=device,
            )
        scenes.append(scene)
        scenes.sort(key=lambda item: int(item["scene_index"]))
        scenes_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
        (shared_dir / f"{index:02d}_{image_path.stem}_raw.txt").write_text(
            scene["raw_response"],
            encoding="utf-8",
        )
        log_stage(scene["scene_summary"][:160], step=f"Qwen-{index:02d}", model=VISION_MODEL_ID, event="summary")

    scenes.sort(key=lambda item: int(item["scene_index"]))
    common_output_dir.mkdir(parents=True, exist_ok=True)
    (common_output_dir / "qwen25_vl_3b_scene_descriptions.json").write_text(
        json.dumps(scenes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scenes


def _try_reuse_scene_cache(
    source_common_output_dir: Path,
    input_dir: Path,
    common_output_dir: Path,
    shared_dir: Path,
) -> list[dict[str, Any]] | None:
    source_path = source_common_output_dir / "qwen25_vl_3b_scene_descriptions.json"
    if not source_path.exists():
        return None

    source_scenes = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_scenes, list):
        return None

    image_by_name = {image.name: image for image in _iter_images(input_dir)}
    if len(source_scenes) != len(image_by_name):
        return None

    scenes: list[dict[str, Any]] = []
    for scene in source_scenes:
        if not isinstance(scene, dict):
            return None
        image_id = str(scene.get("image_id") or "")
        image_path = image_by_name.get(image_id)
        if image_path is None:
            return None
        patched = dict(scene)
        patched["image_path"] = str(image_path.resolve())
        scenes.append(patched)

    scenes.sort(key=lambda item: int(item["scene_index"]))
    common_output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)
    (common_output_dir / "qwen25_vl_3b_scene_descriptions.json").write_text(
        json.dumps(scenes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (shared_dir / "scenes.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")

    source_shared_dir = source_common_output_dir / "scene_descriptions"
    if source_shared_dir.exists():
        for source_raw in sorted(source_shared_dir.glob("*_raw.txt")):
            shutil.copy2(source_raw, shared_dir / source_raw.name)
    log_stage(
        f"reused scene analysis cache from {source_path}",
        step="Qwen-cache",
        model=VISION_MODEL_ID,
        event="cache-hit",
    )
    return scenes


def prepare_qwen_scenes_for_experiment(
    experiment: str,
    input_dir: str | Path = INPUT_DIR,
    output_root: str | Path = OUTPUT_ROOT,
    story_caption: str | None = None,
    collage_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    input_dir = Path(input_dir)
    key = experiment.lower()
    common_output_dir = _experiment_dirs(output_root)[key] / "qwen25_vl_3b_story"
    shared_dir = common_output_dir / "scene_descriptions"
    resized_dir = common_output_dir / "_resized_input"

    if key == "e":
        prompt_builder = _prompt_e_visual_cot
        qwen_max_new_tokens = 240
    elif key == "f":
        prompt_builder = _prompt_f_fairy_tale_image_analyst
        qwen_max_new_tokens = 240
    elif key == "j":
        prompt_builder = lambda index: _prompt_j_scene_with_prior_context(
            index,
            story_caption or "",
            collage_analysis or {},
        )
        qwen_max_new_tokens = 260
    elif key in {"g", "h", "i", "j"}:
        prompt_builder = _prompt_g_cot_persona
        qwen_max_new_tokens = 240
    else:
        prompt_builder = _prompt
        qwen_max_new_tokens = 220
    return _ensure_scenes(
        input_dir=input_dir,
        common_output_dir=common_output_dir,
        shared_dir=shared_dir,
        resized_dir=resized_dir,
        prompt_builder=prompt_builder,
        qwen_max_new_tokens=qwen_max_new_tokens,
    )


def prepare_qwen_collage_for_experiment(
    input_dir: str | Path,
    output_root: str | Path = OUTPUT_ROOT,
    story_caption: str = "",
    experiment: str = "j",
) -> dict[str, Any]:
    output_root = Path(output_root)
    input_dir = Path(input_dir)
    key = experiment.lower()
    experiment_label = f"Experiment {key.upper()}"
    collage_path = _resolve_collage_path(input_dir)
    common_output_dir = _experiment_dirs(output_root)[key] / "qwen25_vl_3b_story" / "collage_analysis"
    log_stage(f"start {experiment_label} collage analysis: {collage_path}", step=f"{key.upper()}-collage", event="start")
    analysis = _run_qwen_collage_analysis(collage_path, common_output_dir, story_caption=story_caption)
    log_stage(f"{experiment_label} collage analysis succeeded", step=f"{key.upper()}-collage", event="success")
    return analysis
