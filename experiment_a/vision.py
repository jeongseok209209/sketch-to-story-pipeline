"""Vision recognition stages for Experiment A."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from utils import (
    get_caption_components,
    get_device,
    get_openclip_components,
    get_vqa_components,
    load_and_normalize_image,
    resize_square,
    timed_step,
)


VQA_QUESTIONS = {
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

    processor, model = get_caption_components()
    device = get_device()
    inputs = processor(
        images=image,
        text="a children's drawing of",
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, num_beams=5, max_length=50)
    caption = _decode_first(processor, output)
    if not caption:
        raise ValueError("BLIP captioning returned an empty result.")
    return caption


def _answer_vqa(image: Any) -> dict[str, str]:
    """Ask the required BLIP-VQA questions and collect short answers."""
    import torch

    processor, model = get_vqa_components()
    device = get_device()
    answers: dict[str, str] = {}

    for key, question in VQA_QUESTIONS.items():
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=20)
        answers[key] = _decode_first(processor, output)

    return answers


def _extract_candidates(text: str) -> list[str]:
    """Extract unique lowercase alphabetic candidate words from model text."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    candidates = list(set(words))
    if not candidates:
        raise ValueError("No candidate words were extracted from BLIP outputs.")
    return candidates


def _score_candidates(image: Any, candidates: list[str]) -> dict[str, float]:
    """Score candidate words with OpenCLIP cosine similarity."""
    import torch

    model, preprocess, tokenizer = get_openclip_components()
    device = get_device()

    image_tensor = preprocess(image).unsqueeze(0).to(device)
    prompts = [f"a child's drawing of a {word}" for word in candidates]
    text_tokens = tokenizer(prompts).to(device)

    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
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
    steps: dict[str, Any] = {}
    source = Path(image_path)

    with timed_step(1, "image input"):
        image = load_and_normalize_image(image_path)
        steps["01_image_input"] = {
            "step": 1,
            "name": "손그림 이미지 입력",
            "image_id": source.name,
            "image_path": str(source.resolve()),
            "image_mode": image.mode,
            "image_size": list(image.size),
        }

    with timed_step(2, "preprocessing for BLIP and OpenCLIP"):
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

    with timed_step(3, "BLIP captioning"):
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

    with timed_step(4, "BLIP-VQA"):
        answers = _answer_vqa(blip_image)
        steps["04_blip_vqa"] = {
            "step": 4,
            "name": "BLIP-VQA 슬롯 질의",
            "output": "{who, action, scene, mood}",
            "questions": VQA_QUESTIONS,
            "answers": answers,
        }

    with timed_step(5, "candidate word extraction"):
        combined = " ".join([raw_caption, *answers.values()])
        candidates = _extract_candidates(combined)

    with timed_step(6, "OpenCLIP candidate scoring"):
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

    with timed_step(7, "vision_json creation"):
        object_scores = {
            word: score
            for word, score in candidate_scores.items()
            if score >= clip_threshold
        }
        objects = list(object_scores.keys())
        confidence = "high" if len(objects) >= 3 else "low"

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
