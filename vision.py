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

    with timed_step(1, "image input"):
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

    with timed_step(2, "preprocessing for BLIP and OpenCLIP"):
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
        # BLIP이 만든 자유 캡션과 VQA 답변을 모두 합쳐 후보 개념의 재료로 사용합니다.
        combined = " ".join([raw_caption, *answers.values()])
        candidates = _extract_candidates(combined)

    with timed_step(6, "OpenCLIP candidate scoring"):
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

    with timed_step(7, "vision_json creation"):
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
