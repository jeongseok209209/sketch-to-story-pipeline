"""Run independent C/D/E/F/G/H/I/J experiments with Qwen vision and EXAONE GGUF writing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from generators import _run_exaone_gguf_prompt, get_last_llama_runtime
from utils import (
    DEFAULT_EXAONE_GGUF_PATH,
    LLAMA_CLI_PATH,
    QWEN25_VL_MODEL,
    local_huggingface_model_path,
    ensure_exaone_gguf_model,
    log_model_device,
    log_stage,
    set_step_context,
    timed_step,
)


VISION_MODEL_ID = QWEN25_VL_MODEL
LLM_MODEL_NOTE = "EXAONE GGUF via llama.cpp"
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_ROOT = BASE_DIR / "outputs"
COMMON_OUTPUT_DIR = OUTPUT_ROOT / "qwen25_vl_3b_story"
SHARED_DIR = COMMON_OUTPUT_DIR / "scene_descriptions"
RESIZED_DIR = COMMON_OUTPUT_DIR / "_resized_input"
QWEN3B_LOCAL_DIR = local_huggingface_model_path(VISION_MODEL_ID)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
QWEN_IMAGE_MAX_SIDE = 384
QWEN_MAX_PIXELS = QWEN_IMAGE_MAX_SIDE * QWEN_IMAGE_MAX_SIDE
QWEN_COLLAGE_MAX_SIDE = 1600
QWEN_COLLAGE_MAX_PIXELS = QWEN_COLLAGE_MAX_SIDE * QWEN_COLLAGE_MAX_SIDE
STORY_CAPTION_FILENAME = "caption.txt"
COLLAGE_FILENAME = "collage_2x5_scene_order.png"
H_REFINEMENT_MAX_NEW_TOKENS = 450
I_REFINEMENT_MAX_NEW_TOKENS = 500
I_ENDING_MAX_NEW_TOKENS = 500
I_CLEANUP_MAX_NEW_TOKENS = 300
I_QUALITY_GATES = (
    "repetition",
    "english",
    "meta_language",
    "ending",
)


def _snapshot_dir(model_cache: Path | str) -> Path | str:
    if not isinstance(model_cache, Path):
        return model_cache
    if isinstance(model_cache, Path) and model_cache.exists() and (model_cache / "config.json").exists():
        return model_cache
    snapshots = model_cache / "snapshots"
    if snapshots.exists():
        dirs = sorted([path for path in snapshots.iterdir() if path.is_dir()])
        if dirs:
            return dirs[-1]
    return VISION_MODEL_ID


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


def _balanced_json_object_candidates(cleaned: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(cleaned[start : index + 1])
                start = None
    if not candidates:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            candidates.append(match.group(0))
    return candidates


def _json_object_candidates(text: str) -> list[str]:
    """Return balanced JSON-object-looking substrings from model text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = _balanced_json_object_candidates(cleaned)
    for start in [match.start() for match in re.finditer(r"\{", cleaned)][-80:]:
        suffix_candidates = _balanced_json_object_candidates(cleaned[start:])
        if suffix_candidates:
            candidates.append(suffix_candidates[0])
    for match in re.finditer(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", text, flags=re.S):
        candidates.extend(_balanced_json_object_candidates(match.group(1).strip()))
    return candidates


def _last_jsonish_fragment(text: str, limit: int = 6000) -> str:
    fenced_blocks = [
        match.group(1).strip()
        for match in re.finditer(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", text, flags=re.S)
        if match.group(1).strip()
    ]
    if fenced_blocks:
        return fenced_blocks[-1][-limit:]
    candidates = _json_object_candidates(text)
    if candidates:
        return candidates[-1][-limit:]
    cleaned = text.strip()
    return cleaned[-limit:] if len(cleaned) > limit else cleaned


def _extract_required_json(text: str) -> dict[str, Any]:
    candidates = _json_object_candidates(text)
    if not candidates:
        raise ValueError("EXAONE response did not contain a JSON object.")
    last_error: Exception | None = None
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, dict):
            story = value.get("story") if isinstance(value.get("story"), dict) else value
            if any(key in story for key in ("title", "body", "scene_sentences")):
                return value
            last_error = ValueError("EXAONE JSON response was not a story object.")
            continue
        last_error = ValueError("EXAONE JSON response was not an object.")
    if last_error:
        raise last_error
    raise ValueError("EXAONE response did not contain a JSON object.")


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
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_source = _snapshot_dir(local_huggingface_model_path(VISION_MODEL_ID))
    log_stage(f"loading collage vision model from {model_source}", step="J-collage-load", model=VISION_MODEL_ID)
    local_only = isinstance(model_source, Path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with timed_step("J-collage-load", "Qwen2.5-VL collage model load", model=VISION_MODEL_ID):
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
                log_stage(f"Qwen CUDA move failed; reloading on CPU: {exc}", step="J-collage-device", model=VISION_MODEL_ID)
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
            max_pixels=QWEN_COLLAGE_MAX_PIXELS,
        )

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


def _compact_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_index": scene.get("scene_index"),
        "image_id": scene.get("image_id"),
        "scene_summary": scene.get("scene_summary", ""),
        "characters": scene.get("characters", []),
        "objects": scene.get("objects", []),
        "setting": scene.get("setting", ""),
        "mood": scene.get("mood", ""),
        "emotion": scene.get("emotion", ""),
        "story_role": scene.get("story_role", ""),
        "uncertain": scene.get("uncertain", ""),
    }


def _scene_windows(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_index = {int(scene["scene_index"]): _compact_scene(scene) for scene in scenes}
    windows = []
    for scene in scenes:
        index = int(scene["scene_index"])
        windows.append(
            {
                "target_scene_index": index,
                "previous_scene": by_index.get(index - 1),
                "current_scene": by_index[index],
                "next_scene": by_index.get(index + 1),
            }
        )
    return windows


def _ensure_exaone_gguf_available() -> None:
    model_path = Path(
        ensure_exaone_gguf_model(os.environ.get("EXAONE_GGUF_MODEL_PATH") or DEFAULT_EXAONE_GGUF_PATH)
    )
    llama_cli = Path(os.environ.get("LLAMA_CLI_PATH") or LLAMA_CLI_PATH).expanduser()
    if not llama_cli.exists() and os.environ.get("AUTO_INSTALL_LLAMA_CPP", "").strip().lower() not in {
        "",
        "1",
        "true",
        "yes",
        "on",
    }:
        raise FileNotFoundError(
            "llama.cpp CLI not found. Set LLAMA_CLI_PATH or build it at: "
            f"{llama_cli}"
        )


def _has_korean_text(value: str, minimum_chars: int = 4) -> bool:
    return len(re.findall(r"[\uac00-\ud7a3]", value)) >= minimum_chars


def _global_story_invalid_reasons(value: str, *, field_name: str) -> list[str]:
    reasons: list[str] = []
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = cleaned.lower()
    if not cleaned:
        reasons.append("empty")
    if not _has_korean_text(cleaned, minimum_chars=4):
        reasons.append("not_korean_prose")
    if _looks_like_placeholder(cleaned):
        reasons.append("placeholder")
    schema_markers = (
        "one non-empty korean",
        "actual korean story",
        "actual korean prose",
        "write real korean prose",
        "for this scene",
        "story.scene_sentences",
        "scene_sentences",
        "story_sentence",
        "grounding_notes",
        "json",
        "objects",
        "characters",
        "setting",
        "qwen",
        "\uadf8\ub9bc \uadfc\uac70\ub97c \uc5b4\ub5bb\uac8c \ubc18\uc601",
        "\uac01 \uadf8\ub9bc\uc758 objects",
    )
    if any(marker in lowered for marker in schema_markers):
        reasons.append("schema_or_meta_language")
    if field_name == "scene_sentences" and _has_meta_language(cleaned):
        reasons.append("meta_language")
    return reasons


def _validate_global_story_text(value: str, *, field_name: str, scene_index: int | None = None) -> None:
    reasons = _global_story_invalid_reasons(value, field_name=field_name)
    if reasons:
        location = f" scene {scene_index}" if scene_index is not None else ""
        raise ValueError(
            f"exaone_output_invalid: EXAONE story.{field_name}{location} is not valid story prose "
            f"({', '.join(sorted(set(reasons)))}): {value[:160]!r}"
        )


def _story_from_payload(payload: dict[str, Any], scene_count: int) -> dict[str, Any]:
    story = payload.get("story") if isinstance(payload.get("story"), dict) else payload
    title = str(story.get("title") or "").strip()
    if not title:
        raise ValueError("EXAONE story.title is required.")
    if _looks_like_placeholder(title) or not _has_korean_text(title, minimum_chars=2):
        raise ValueError(f"exaone_output_invalid: EXAONE story.title is not usable: {title[:160]!r}")
    body = str(story.get("body") or "").strip()
    if not body:
        raise ValueError("EXAONE story.body is required.")
    _validate_global_story_text(body, field_name="body")
    scene_sentences = story.get("scene_sentences")
    if not isinstance(scene_sentences, list):
        raise ValueError("EXAONE story.scene_sentences must be a list.")
    scene_sentences = [str(sentence).strip() for sentence in scene_sentences if str(sentence).strip()]
    if len(scene_sentences) != scene_count:
        raise ValueError(
            f"EXAONE returned {len(scene_sentences)} scene_sentences for {scene_count} scenes."
        )
    for index, sentence in enumerate(scene_sentences, start=1):
        _validate_global_story_text(sentence, field_name="scene_sentences", scene_index=index)
    grounding_notes = story.get("grounding_notes")
    return {
        "title": title,
        "body": body,
        "scene_sentences": scene_sentences,
        "grounding_notes": grounding_notes if grounding_notes is not None else [],
    }


def _global_story_json_schema(scene_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "structure": {},
            "plan": {},
            "story": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "body": {"type": "string", "minLength": 1},
                    "scene_sentences": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": scene_count,
                        "maxItems": scene_count,
                    },
                    "grounding_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["title", "body", "scene_sentences"],
                "additionalProperties": True,
            },
        },
        "required": ["story"],
        "additionalProperties": True,
    }


def _run_exaone_experiment(
    experiment_name: str,
    prompt_strategy: str,
    prompt: str,
    scenes: list[dict[str, Any]],
    max_new_tokens: int = 2200,
    context_size: int = 8192,
) -> dict[str, Any]:
    _ensure_exaone_gguf_available()
    json_schema = _global_story_json_schema(len(scenes))
    with timed_step(
        "EXAONE",
        f"{experiment_name} EXAONE GGUF generation",
        experiment=experiment_name,
        model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
    ):
        raw_response = _run_exaone_gguf_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            timeout=300,
            context_size=context_size,
            json_schema=json_schema,
        )
    llama_runtime = get_last_llama_runtime()
    json_repair_used = False
    try:
        payload = _extract_required_json(raw_response)
        story = _story_from_payload(payload, len(scenes))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        repair_prompt = _build_json_repair_prompt(raw_response, len(scenes), scenes)
        with timed_step(
            "EXAONE-repair",
            f"{experiment_name} EXAONE JSON repair",
            experiment=experiment_name,
            model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
        ):
            repair_response = _run_exaone_gguf_prompt(
                repair_prompt,
                max_new_tokens=max_new_tokens,
                timeout=300,
                context_size=context_size,
                json_schema=json_schema,
            )
        json_repair_used = True
        try:
            payload = _extract_required_json(repair_response)
            story = _story_from_payload(payload, len(scenes))
        except (json.JSONDecodeError, ValueError, TypeError) as repair_exc:
            raise RuntimeError(
                "json_repair_failed: EXAONE did not return valid D-aligned story JSON, and JSON repair also failed. "
                "Required fields are story.title, story.body, and one story.scene_sentences item per input scene. "
                f"initial_error={exc}; repair_error={repair_exc}; "
                f"cleaned_response_head={raw_response[:800]!r}; repair_response_head={repair_response[:800]!r}; "
                f"llama_runtime={get_last_llama_runtime()!r}"
            ) from repair_exc
        raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
    return {
        "prompt_strategy": prompt_strategy,
        "exaone_prompt": prompt,
        "exaone_raw_response": raw_response,
        "llama_runtime": llama_runtime,
        "parsed_result": payload,
        "json_repair_used": json_repair_used,
        "story": story,
        "structure": payload.get("structure", {}),
        "plan": payload.get("plan", {}),
        "experiment_method": experiment_name,
    }


def _build_e_scene_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene.get("scene_index", 0))
    return (
        "실험 E: 아래 Qwen 장면 JSON 하나만 보고, 해당 그림에 맞는 한국어 동화 문단을 작성하세요.\n"
        "이전 장면이나 다음 장면은 추측하지 마세요. 현재 장면에 보이는 단서만 사용하세요.\n"
        "story_sentence는 아이가 읽기 쉬운 한국어 3~5문장으로 쓰세요. 가능하면 정확히 3문장으로 쓰세요.\n"
        "\"해당하는 문단\", \"동화 문장\", \"...\" 같은 placeholder나 형식 설명 문구를 쓰지 마세요.\n"
        "반드시 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        f"출력 JSON은 scene_index와 story_sentence 두 키만 포함하고, scene_index 값은 {scene_index}이어야 합니다.\n\n"
        f"scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_e_scene_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    return (
        "You are a strict JSON repair tool. Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        "story_sentence must be real Korean fairy-tale prose with exactly 3 short sentences.\n"
        "If the original response has fewer than 3 sentences, expand it to exactly 3 sentences using only the scene context.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, or ....\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_f_scene_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene.get("scene_index", 0))
    return (
        "실험 F: 당신은 어린이 그림을 따뜻한 동화 문단으로 바꾸는 동화작가입니다.\n"
        "아래 Qwen 장면 JSON 하나만 보고, 해당 그림에 맞는 한국어 동화 문단을 작성하세요.\n"
        "이전 장면이나 다음 장면은 추측하지 마세요. 현재 장면에 보이는 단서만 사용하세요.\n"
        "동화작가답게 부드럽고 따뜻한 문체로 쓰되, 그림에 없는 사건이나 관계를 과하게 꾸며내지 마세요.\n"
        "story_sentence는 아이가 읽기 쉬운 한국어 3~5문장으로 쓰세요. 가능하면 정확히 3문장으로 쓰세요.\n"
        "\"해당하는 문단\", \"동화 문장\", \"...\" 같은 placeholder나 형식 설명 문구를 쓰지 마세요.\n"
        "반드시 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        f"출력 JSON은 scene_index와 story_sentence 두 키만 포함하고, scene_index 값은 {scene_index}이어야 합니다.\n\n"
        f"scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_f_scene_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    return (
        "You are a strict JSON repair tool and a warm Korean fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        "Keep story_sentence in a gentle fairy-tale writer style.\n"
        "If possible, make story_sentence exactly 3 short Korean sentences using only the scene context.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, or ....\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _g_scene_position_hint(scene_index: int) -> str:
    if scene_index == 1:
        return "이 장면은 이야기의 시작입니다. 인물, 장소, 분위기를 소개하는 느낌을 살리세요."
    if scene_index == 10:
        return "이 장면은 이야기의 마무리입니다. 따뜻한 결말과 여운을 주는 느낌을 살리세요."
    return "이 장면은 앞뒤 흐름 속에서 자연스럽게 이어지는 중간 장면입니다. 너무 구체적인 단계나 사건을 강제하지 마세요."


def _build_g_scene_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene.get("scene_index", 0))
    return (
        "실험 G: 당신은 어린이 그림을 따뜻한 동화 문단으로 바꾸는 동화작가입니다.\n"
        "아래 Qwen 장면 JSON 하나만 보고, 해당 그림에 맞는 한국어 동화 문단을 작성하세요.\n"
        "먼저 내부적으로 현재 그림의 핵심 시각 단서, 장면 번호의 이야기상 위치감, 문장 수, placeholder 여부를 점검하세요.\n"
        "단, 이 점검 과정은 출력하지 말고 최종 JSON에만 반영하세요.\n"
        f"장면 위치감: {_g_scene_position_hint(scene_index)}\n"
        "현재 장면에 보이는 단서만 사용하고, 그림에 없는 사건이나 관계를 과하게 꾸며내지 마세요.\n"
        "story_sentence는 아이가 읽기 쉬운 한국어 3~5문장으로 쓰세요. 가능하면 정확히 3문장으로 쓰세요.\n"
        "\"해당하는 문단\", \"동화 문장\", \"...\" 같은 placeholder나 형식 설명 문구를 쓰지 마세요.\n"
        "반드시 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        f"출력 JSON은 scene_index와 story_sentence 두 키만 포함하고, scene_index 값은 {scene_index}이어야 합니다.\n\n"
        f"scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_g_scene_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    return (
        "You are a strict JSON repair tool and a warm Korean fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"Scene position nuance: {_g_scene_position_hint(scene_index)}\n"
        "Keep story_sentence in a gentle fairy-tale writer style using only the scene context.\n"
        "If possible, make story_sentence exactly 3 short Korean sentences.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, or ....\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_g_refinement_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    return (
        "실험 G 2차 순차 개선: 당신은 어린이 동화를 다듬는 따뜻한 동화작가입니다.\n"
        "앞 장면의 최종 문장과 현재 장면의 1차 문장을 보고, 현재 장면 문장만 더 자연스럽게 개선하세요.\n"
        "먼저 내부적으로 앞 문장의 감정/상황, 현재 그림의 핵심 시각 단서, 자연스러운 연결, placeholder 여부를 점검하세요.\n"
        "단, 이 점검 과정은 출력하지 말고 최종 JSON에만 반영하세요.\n"
        f"장면 위치감: {_g_scene_position_hint(scene_index)}\n"
        "앞 문장과 자연스럽게 이어지게 하되, 현재 그림에 없는 사건이나 사물을 과하게 추가하지 마세요.\n"
        "현재 scene JSON의 시각 근거가 앞 문장보다 우선입니다.\n"
        "story_sentence는 아이가 읽기 쉬운 한국어 3~5문장으로 쓰세요. 가능하면 정확히 3문장으로 쓰세요.\n"
        "\"해당하는 문단\", \"동화 문장\", \"...\" 같은 placeholder나 형식 설명 문구를 쓰지 마세요.\n"
        "반드시 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        f"출력 JSON은 scene_index와 story_sentence 두 키만 포함하고, scene_index 값은 {scene_index}이어야 합니다.\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_initial_story_sentence:\n{initial_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_g_refinement_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    return (
        "You are a strict JSON repair tool and a warm Korean fairy-tale editor.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"Scene position nuance: {_g_scene_position_hint(scene_index)}\n"
        "The revised story_sentence should connect naturally after the previous final sentence while staying grounded in the current scene.\n"
        "If possible, make story_sentence exactly 3 short Korean sentences.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, or ....\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_INITIAL_STORY_SENTENCE:\n{initial_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _story_caption_from_scene(scene: dict[str, Any]) -> str:
    return str(scene.get("_story_caption") or "").strip()


def _i_scene_position_role(scene_index: int) -> str:
    if scene_index == 1:
        return (
            "opening: introduce the main presence, place, and mood. "
            "Do not rush into a large event."
        )
    if scene_index in (2, 3):
        return (
            "early story: show a small interest, action, meeting, or situation beginning. "
            "Keep it grounded in this drawing."
        )
    if 4 <= scene_index <= 6:
        return (
            "middle story: let something already noticed or started move, grow, or change a little. "
            "Do not force a specific plot."
        )
    if 7 <= scene_index <= 9:
        return (
            "late story: show a result, response, or movement toward closure. "
            "Avoid starting a brand-new problem."
        )
    if scene_index == 10:
        return (
            "ending: close the story warmly without starting a new event. "
            "Leave calm afterglow."
        )
    return "middle story: keep the scene connected but grounded in the current drawing."


def _weak_story_direction(story_caption: str) -> str:
    text = story_caption.strip()
    hints: list[str] = []
    if re.search(r"어두|무서|길\s*잃|외롭|슬프|걱정", text):
        hints.append("A gentle movement from uncertainty toward comfort may fit the whole story.")
    if re.search(r"도와|함께|친구|나누|구해|돌봐", text):
        hints.append("Kindness or togetherness may matter if the current drawing supports it.")
    if re.search(r"찾|발견|만나|얻", text):
        hints.append("A small discovery or meeting may feel meaningful if it is visible or naturally implied.")
    if re.search(r"자라|키우|만들|변하|커지", text):
        hints.append("A small change may gradually grow across the story if the drawings support it.")
    if not hints:
        hints.append("Keep a gentle, coherent picture-book arc across the ten scenes.")
    hints.append("Do not copy concrete nouns from the caption unless they are visible in the current scene JSON.")
    return " ".join(hints[:3])


def _build_h_scene_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene.get("scene_index", 0))
    return (
        "Experiment H: You are a warm Korean fairy-tale writer for children's drawings.\n"
        "Write one Korean fairy-tale paragraph for the current scene JSON only.\n"
        "Use only the current scene JSON and scene position nuance. Do not use any whole-story caption in this initial pass.\n"
        "Internally check visible clues, scene position, sentence count, and placeholders. Do not output these steps.\n"
        f"Scene position nuance: {_g_scene_position_hint(scene_index)}\n"
        "Scene 1 should feel like a beginning. Scene 10 should feel like a warm ending. Scenes 2-9 should read as middle scenes in order.\n"
        "Prioritize varied, concrete visual details from this scene so the paragraphs do not all repeat the same motif.\n"
        "story_sentence must be Korean fairy-tale prose with 3 to 5 short sentences, preferably exactly 3 sentences.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, story_sentence, or ....\n"
        "Output exactly one JSON object only. Do not add markdown, explanations, or code fences.\n"
        f"The JSON must contain only scene_index and story_sentence, and scene_index must be {scene_index}.\n\n"
        f"scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_h_scene_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    return (
        "You are a strict JSON repair tool and a warm Korean fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"Scene position nuance: {_g_scene_position_hint(scene_index)}\n"
        "Use only the current scene context and scene position nuance. Do not add whole-story caption details in this initial repair.\n"
        "If possible, make story_sentence exactly 3 short Korean fairy-tale sentences.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, story_sentence, or ....\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_h_refinement_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    return (
        "Experiment H second-pass sequential refinement: You are a warm Korean children's fairy-tale writer.\n"
        "Rewrite only the current scene paragraph as gentle Korean fairy-tale prose using the previous final paragraph, the current initial paragraph, the current scene JSON, and story_caption.\n"
        "Do not merely correct or summarize the paragraph; shape it like a short scene in a children's picture book.\n"
        "Use soft sensory details, natural emotional flow, and short rhythmic sentences that are easy for a child to read.\n"
        "Avoid report-like, explanatory, evaluative, or stiff wording.\n"
        "Use story_caption only as a weak direction check for the whole story, not as wording to copy into the paragraph.\n"
        "Do not repeat the main nouns or phrases from story_caption in every scene.\n"
        "Use a key word from story_caption only when it clearly fits the current scene's visible clues; otherwise prefer the current scene JSON.\n"
        "The current scene JSON and previous final paragraph are more important than story_caption.\n"
        "Internally check continuity from the previous final paragraph, visible clues, fairy-tale tone, scene position, weak caption consistency, and placeholders. Do not output these steps.\n"
        f"Scene position nuance: {_g_scene_position_hint(scene_index)}\n"
        "The revised paragraph should connect naturally after the previous final paragraph while staying grounded in the current scene.\n"
        "story_sentence must be warm Korean fairy-tale prose with 3 to 5 short sentences, preferably exactly 3 sentences.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, story_sentence, or ....\n"
        "Output exactly one JSON object only. Do not add markdown, explanations, or code fences.\n"
        f"The JSON must contain only scene_index and story_sentence, and scene_index must be {scene_index}.\n\n"
        f"story_caption:\n{story_caption}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_initial_story_sentence:\n{initial_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_h_refinement_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    return (
        "You are a strict JSON repair tool and a warm Korean children's fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"Scene position nuance: {_g_scene_position_hint(scene_index)}\n"
        "When repairing story_sentence, keep it in a gentle children's fairy-tale style, not a report or explanation.\n"
        "Use soft sensory details, natural emotional flow, and short rhythmic Korean sentences.\n"
        "Use story_caption only as a weak direction check for the whole story, not as wording to copy.\n"
        "Do not repeat the main nouns or phrases from story_caption in every scene.\n"
        "Use a key word from story_caption only when it clearly fits the current scene's visible clues; otherwise prefer the current scene context.\n"
        "The current scene context and previous final sentence are more important than story_caption.\n"
        "The revised story_sentence should connect naturally after the previous final sentence while staying grounded in the current scene.\n"
        "If possible, make story_sentence exactly 3 short, warm Korean fairy-tale sentences.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, story_sentence, or ....\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"STORY_CAPTION:\n{story_caption}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_INITIAL_STORY_SENTENCE:\n{initial_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _fairy_tale_speech_style_policy() -> str:
    return (
        "Fairy-tale speech style policy:\n"
        "- Write like a picture-book narrator gently reading aloud to a child.\n"
        "- Prefer soft Korean fairy-tale endings such as 했어요, 했답니다, 있었어요, 되었어요, 되었답니다, 였답니다.\n"
        "- Mix 했어요-style and 했답니다-style endings naturally; keep the rhythm warm and child-friendly.\n"
        "- avoid stiff report-style endings: do not end most sentences with ~다/~습니다, ~보입니다, ~나타납니다, or ~입니다.\n"
        "- Keep the same meaning and visual grounding, but convert stiff sentence endings into warm fairy-tale spoken endings.\n"
    )


def _build_i_caption_initial_scene_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene.get("scene_index", 0))
    weak_direction = str(scene.get("_weak_story_direction") or _weak_story_direction(_story_caption_from_scene(scene)))
    position_role = str(scene.get("_scene_position_role") or _i_scene_position_role(scene_index))
    ending_instruction = ""
    if scene_index == 10:
        ending_instruction = (
            "This is the last scene. Make the paragraph feel like a warm ending with gentle afterglow. "
            "Do not start a new event.\n"
        )
    return (
        "Experiment I initial scene writing: You are a warm Korean fairy-tale writer for children's drawings.\n"
        "Write one Korean fairy-tale paragraph for the current scene JSON.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Use scene_position_role only to control the story rhythm, not to force plot events.\n"
        "Use weak_story_direction only as a broad consistency hint.\n"
        "weak_story_direction is not scene content, not required vocabulary, and not a plot checklist.\n"
        "Do not use caption-derived nouns or expressions unless the current scene JSON directly supports them.\n"
        "A good paragraph may contain none of the caption words.\n"
        "Priority: current scene JSON > scene_position_role > weak_story_direction.\n"
        f"{ending_instruction}"
        "Use scene_summary, characters, objects, setting, and mood before story_role. Treat story_role as weak analysis only.\n"
        "Internally check visible clues, caption overuse, scene position, sentence count, and placeholders. Do not output these steps.\n"
        f"scene_position_role:\n{position_role}\n\n"
        f"weak_story_direction:\n{weak_direction}\n\n"
        "story_sentence must be Korean fairy-tale prose with 2 to 5 short sentences, preferably exactly 3 sentences.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, story_sentence, or ....\n"
        "Output exactly one JSON object only. Do not add markdown, explanations, or code fences.\n"
        f"The JSON must contain only scene_index and story_sentence, and scene_index must be {scene_index}.\n\n"
        f"scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_caption_initial_scene_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    position_role = str(scene.get("_scene_position_role") or _i_scene_position_role(scene_index))
    ending_instruction = ""
    if scene_index == 10:
        ending_instruction = "This is the last scene. Make story_sentence feel like a warm ending, not a new event.\n"
    return (
        "You are a strict JSON repair tool and a warm Korean fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"{ending_instruction}"
        "Do not use story_caption or any external caption in this repair.\n"
        f"scene_position_role: {position_role}\n"
        "Use the current scene context as the main evidence.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "If possible, make story_sentence exactly 3 short Korean fairy-tale sentences. At minimum, avoid a one-sentence result.\n"
        "Do not use placeholders such as 해당하는 문단, 동화 문장, story_sentence, or ....\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_sequential_refinement_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    position_role = str(scene.get("_scene_position_role") or _i_scene_position_role(scene_index))
    ending_instruction = ""
    if scene_index == 10:
        ending_instruction = (
            "This is the final scene. Close the story warmly with relief and afterglow. "
            "Do not start a new event. Do not simply list visible characters or objects.\n"
            "Do not copy or lightly paraphrase the previous final paragraph; write a distinct final paragraph grounded in scene 10.\n"
        )
    return (
        "Experiment I second-pass sequential refinement: You are a warm Korean children's fairy-tale writer.\n"
        "Rewrite only the current scene paragraph as gentle Korean fairy-tale prose.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Use the previous final paragraph, the current initial paragraph, the current scene JSON, and scene_position_role.\n"
        "Do not use story_caption in this second pass.\n"
        f"{ending_instruction}"
        "The previous final paragraph is only a bridge. Do not copy its expressions, motifs, sentence shape, or visual details.\n"
        "Connect naturally after it while staying grounded in the current drawing.\n"
        "Reflect at least one new visible clue from the current scene JSON that was not central in the previous final paragraph.\n"
        "Use scene_position_role only for rhythm and story function, not to force a plot event.\n"
        "Use soft sensory details, natural emotional flow, and short rhythmic sentences.\n"
        "Avoid report-like, explanatory, evaluative, or stiff wording.\n"
        "Do not add events or objects that are not supported by the current scene JSON.\n"
        "story_sentence must be Korean fairy-tale prose with 2 to 5 short sentences, preferably exactly 3 sentences.\n"
        "Output exactly one JSON object only. Do not add markdown, explanations, or code fences.\n"
        f"The JSON must contain only scene_index and story_sentence, and scene_index must be {scene_index}.\n\n"
        f"scene_position_role:\n{position_role}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_initial_story_sentence:\n{initial_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_sequential_refinement_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    position_role = str(scene.get("_scene_position_role") or _i_scene_position_role(scene_index))
    ending_instruction = ""
    if scene_index == 10:
        ending_instruction = (
            "This is the final scene. End warmly; do not start a new event. "
            "Do not copy or lightly paraphrase the previous final sentence.\n"
        )
    return (
        "You are a strict JSON repair tool and a warm Korean children's fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"{ending_instruction}"
        "Do not use story_caption in this repair.\n"
        f"scene_position_role: {position_role}\n"
        "The revised story_sentence should connect naturally after the previous final sentence while staying grounded in the current scene.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Do not copy the previous sentence's expressions, motifs, sentence shape, or visual details.\n"
        "Include at least one current-scene visual clue.\n"
        "If possible, make story_sentence exactly 3 short, warm Korean fairy-tale sentences. At minimum, avoid a one-sentence result.\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_INITIAL_STORY_SENTENCE:\n{initial_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_english_translation_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    english_terms = ", ".join(_english_words(current_sentence))
    return (
        "Experiment I English translation cleanup: You are a Korean fairy-tale text editor.\n"
        "Rewrite current_story_sentence by translating any English words or English phrases into natural Korean.\n"
        "Do not use story_caption in this cleanup.\n"
        "Keep the same story meaning, scene_index, and Korean fairy-tale tone as much as possible.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Use the current scene JSON only to choose natural Korean wording when needed.\n"
        f"English terms to remove: {english_terms}\n"
        "Output exactly one JSON object only. Do not add markdown, explanations, or code fences.\n"
        f"The JSON must contain only scene_index and story_sentence, and scene_index must be {scene_index}.\n\n"
        f"current_story_sentence:\n{current_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_english_translation_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    return (
        "You are a strict JSON repair tool and Korean fairy-tale text editor.\n"
        "Return one valid JSON object only. No markdown. No explanation.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        "story_sentence must be Korean fairy-tale prose with all English words translated into Korean.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Do not use story_caption in this repair.\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"CURRENT_STORY_SENTENCE:\n{current_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_english_term_translation_prompt(terms: list[str], sentence: str) -> str:
    return (
        "Experiment I English term translation: Translate only the listed English-containing terms/fragments into Korean.\n"
        "Do not decide whether English exists; the caller already detected these terms/fragments in Python.\n"
        "Return exactly one JSON object only. Do not add markdown, explanations, or code fences.\n"
        "The JSON object must have this shape:\n"
        "{\n"
        '  "translations": {\n'
        '    "EnglishTerm": "KoreanTranslation"\n'
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Translate each listed item 1:1 into Korean replacement text.\n"
        "- Values must not contain English letters.\n"
        "- Do not rewrite the sentence.\n"
        "- If an item mixes English with a Korean suffix, translate the whole listed item into one natural Korean replacement.\n\n"
        f"terms:\n{json.dumps(terms, ensure_ascii=False)}\n\n"
        f"sentence:\n{sentence}\n"
    )


def _build_i_english_term_translation_repair_prompt(
    raw_response: str,
    terms: list[str],
    sentence: str,
) -> str:
    return (
        "You are a strict JSON repair tool.\n"
        "Convert the model response into one valid JSON object only. No markdown. No explanation.\n"
        "The JSON object must contain a translations object mapping every listed item to Korean.\n"
        "Translation values must not contain English letters.\n\n"
        "Required shape:\n"
        "{\n"
        '  "translations": {\n'
        '    "EnglishTerm": "KoreanTranslation"\n'
        "  }\n"
        "}\n\n"
        f"terms:\n{json.dumps(terms, ensure_ascii=False)}\n\n"
        f"sentence:\n{sentence}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _english_translation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "object",
                "additionalProperties": {"type": "string", "minLength": 1},
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }


def _scene_story_json_schema(scene_index: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scene_index": {"type": "integer", "const": scene_index},
            "story_sentence": {"type": "string", "minLength": 1},
        },
        "required": ["scene_index", "story_sentence"],
        "additionalProperties": False,
    }


def _extract_i_english_translations(raw_response: str, terms: list[str]) -> dict[str, str]:
    term_keys = {term.lower(): term for term in terms}
    last_error: Exception | None = None
    for candidate in reversed(_json_object_candidates(raw_response)):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = ValueError("translation response was not a JSON object")
            continue
        raw_translations = payload.get("translations", payload)
        if not isinstance(raw_translations, dict):
            last_error = ValueError("translation response did not contain a translations object")
            continue
        translations: dict[str, str] = {}
        lowered_payload = {str(key).lower(): value for key, value in raw_translations.items()}
        for lower_term, original_term in term_keys.items():
            value = lowered_payload.get(lower_term)
            translated = str(value or "").strip()
            if not translated or _english_words(translated):
                continue
            translations[original_term] = translated
        if translations:
            return translations
        last_error = ValueError("translation response did not include usable Korean translations")
    if last_error:
        raise last_error
    raise ValueError("translation response did not contain a JSON object")


def _build_i_opening_refinement_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    return (
        "Experiment I opening refinement: You are a warm Korean children's fairy-tale writer.\n"
        "Rewrite the first scene as the opening paragraph of a picture-book fairy tale.\n"
        "Use the current scene JSON and current initial paragraph. Use story_caption only as a weak direction check.\n"
        "Introduce the character, place, and mood through story prose. Do not say this is the first scene or explain the role of the image.\n"
        "Use only Korean fairy-tale prose. Do not include English words, analysis labels, or prompt words.\n"
        "Do not copy story_caption wording, and do not force caption nouns unless the current scene visibly supports them.\n"
        "Use soft sensory details, natural emotion, and short rhythmic sentences for children.\n"
        "Output exactly one JSON object only with scene_index and story_sentence.\n"
        f"scene_index must be {scene_index}.\n\n"
        f"story_caption:\n{story_caption}\n\n"
        f"current_initial_story_sentence:\n{initial_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_middle_refinement_prompt(scene: dict[str, Any]) -> str:
    return _build_h_refinement_prompt(scene)


def _build_i_ending_refinement_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    return (
        "Experiment I ending refinement: You are a warm Korean children's fairy-tale writer.\n"
        "Rewrite the tenth scene as the final paragraph of the story.\n"
        "Use the previous final paragraph, current initial paragraph, current scene JSON, and story_caption.\n"
        "Do not start a new event. Do not simply list every visible character or object.\n"
        "Close the story with warmth, relief, togetherness, and a gentle afterglow while staying grounded in the current drawing.\n"
        "Use story_caption only as a weak direction check. Do not repeat its main nouns unless the current scene visibly supports them.\n"
        "Use only Korean fairy-tale prose. Do not include English words, analysis labels, or prompt words.\n"
        "Use soft sensory details, natural emotion, and short rhythmic sentences for children.\n"
        "Output exactly one JSON object only with scene_index and story_sentence.\n"
        f"scene_index must be {scene_index}.\n\n"
        f"story_caption:\n{story_caption}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_initial_story_sentence:\n{initial_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_refinement_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    ending_instruction = ""
    if scene_index == 10:
        ending_instruction = (
            "This is the final scene. End warmly with relief, togetherness, and afterglow. "
            "Do not start a new event or list objects.\n"
        )
    return (
        "You are a strict JSON repair tool and a warm Korean children's fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"{ending_instruction}"
        "Keep story_sentence as Korean fairy-tale prose for children.\n"
        "Remove English words, analysis labels, prompt words, and meta sentences such as '이 장면은'.\n"
        "Use story_caption only as weak direction. Do not copy or repeat its main nouns unless visible in the current scene.\n"
        "The current scene context and previous final sentence are more important than story_caption.\n"
        "If possible, make story_sentence exactly 3 short Korean sentences.\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"STORY_CAPTION:\n{story_caption}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_INITIAL_STORY_SENTENCE:\n{initial_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_context_style_refinement_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    position_instruction = (
        "If this is scene 1, make it feel like the beginning of the story through prose, not explanation.\n"
        if scene_index == 1
        else "Keep this paragraph flowing naturally from the previous final paragraph.\n"
    )
    return (
        "Experiment I combined context-style refinement: You are a warm Korean children's fairy-tale writer.\n"
        "Rewrite only the current scene paragraph.\n"
        "Do two jobs together: connect naturally after the previous final paragraph, and make the prose soft, rhythmic, and fairy-tale-like.\n"
        f"{_fairy_tale_speech_style_policy()}"
        f"{position_instruction}"
        "Use the current scene JSON and current initial paragraph as the main evidence.\n"
        "Use story_caption only as a weak whole-story direction check, not as wording to copy.\n"
        "Do not repeat story_caption's main nouns in every scene; use them only when the current drawing visibly supports them.\n"
        "Avoid report-like, explanatory, evaluative, or analysis-style wording.\n"
        "Use only Korean fairy-tale prose. Do not include English words, field names, analysis labels, or prompt words.\n"
        "Internally check continuity, visible clues, fairy-tale tone, caption overuse, English, meta language, and placeholders. Do not output these steps.\n"
        f"Scene position nuance: {_g_scene_position_hint(scene_index)}\n"
        "story_sentence must contain 3 to 5 short Korean sentences, preferably exactly 3 sentences.\n"
        "Output exactly one JSON object only with scene_index and story_sentence. Do not add markdown or explanations.\n"
        f"scene_index must be {scene_index}.\n\n"
        f"story_caption:\n{story_caption}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_initial_story_sentence:\n{initial_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_context_style_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    initial_sentence = str(scene.get("_g_initial_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    return (
        "You are a strict JSON repair tool and a warm Korean children's fairy-tale writer.\n"
        "Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        "Keep story_sentence connected to the previous final paragraph and written as gentle Korean fairy-tale prose.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Remove English words, analysis labels, field names, prompt words, and meta sentences.\n"
        "Use story_caption only as weak direction. Do not copy or repeat its main nouns unless visible in the current scene.\n"
        "The current scene context, current initial paragraph, and previous final paragraph are more important than story_caption.\n"
        "If possible, make story_sentence exactly 3 short Korean sentences.\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"STORY_CAPTION:\n{story_caption}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_INITIAL_STORY_SENTENCE:\n{initial_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_final_ending_prompt(scene: dict[str, Any]) -> str:
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    return (
        "Experiment I final ending refinement: You are a warm Korean children's fairy-tale writer.\n"
        "Rewrite only scene 10 as the final paragraph of the whole story.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Use the previous final paragraph, the current scene 10 paragraph, current scene JSON, and story_caption.\n"
        "Do not start a new event. Do not simply list visible characters or objects.\n"
        "Close the story with warmth, relief, togetherness, and a gentle afterglow while staying grounded in the current drawing.\n"
        "Use story_caption only as a weak direction check. Do not copy or repeat its main nouns unless the current drawing visibly supports them.\n"
        "Use only Korean fairy-tale prose. Do not include English words, field names, analysis labels, or prompt words.\n"
        "story_sentence must contain 3 to 5 short Korean sentences, preferably exactly 3 sentences.\n"
        "Output exactly one JSON object only with scene_index and story_sentence. Do not add markdown or explanations.\n"
        "scene_index must be 10.\n\n"
        f"story_caption:\n{story_caption}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_scene_10_story_sentence:\n{current_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_final_ending_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    previous_sentence = str(scene.get("_g_previous_sentence") or "").strip()
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    story_caption = _story_caption_from_scene(scene)
    return (
        "You are a strict JSON repair tool and a warm Korean children's fairy-tale writer.\n"
        "Return one valid JSON object only. No markdown. No explanation.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        "scene_index must be 10.\n"
        "This is the final paragraph. End warmly with relief, togetherness, and afterglow.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Do not start a new event or list objects.\n"
        "Remove English words, analysis labels, field names, prompt words, and meta sentences.\n"
        "Use story_caption only as weak direction. Do not copy or repeat its main nouns unless visible in the current scene.\n"
        "If possible, make story_sentence exactly 3 short Korean sentences.\n"
        "Required shape:\n"
        "{\n"
        '  "scene_index": 10,\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"STORY_CAPTION:\n{story_caption}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_SCENE_10_STORY_SENTENCE:\n{current_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_cleanup_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    story_caption = _story_caption_from_scene(scene)
    previous_sentence = str(scene.get("_i_previous_sentence") or "").strip()
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    reasons = scene.get("_i_quality_reasons") or []
    reason_text = ", ".join(str(reason) for reason in reasons)
    ending_instruction = ""
    if bool(scene.get("_i_ending_scene")):
        ending_instruction = (
            "This is scene 10, the ending. Close the story warmly. Do not start a new event. "
            "Do not list visible objects one by one. Give a gentle afterglow.\n"
        )
    return (
        "Experiment I quality cleanup: You are a strict Korean fairy-tale cleanup writer.\n"
        "Rewrite current_story_sentence into clean Korean children's fairy-tale prose.\n"
        f"{_fairy_tale_speech_style_policy()}"
        f"Cleanup reasons: {reason_text}\n"
        f"{ending_instruction}"
        "Remove all English words. Replace them with natural Korean words.\n"
        "Remove meta or analysis language such as '이 장면은', '그림에는', scene, mood, emotion, story_role, or story_sentence.\n"
        "Do not copy story_caption. Do not repeat caption key nouns unless they are directly supported by the current scene.\n"
        "The current scene JSON and previous final sentence are the main sources.\n"
        "Keep the paragraph warm, concrete, and easy for a child to read.\n"
        "Output exactly one JSON object only with scene_index and story_sentence.\n"
        f"scene_index must be {scene_index}.\n\n"
        f"story_caption:\n{story_caption}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_story_sentence:\n{current_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_cleanup_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    previous_sentence = str(scene.get("_i_previous_sentence") or "").strip()
    reasons = scene.get("_i_quality_reasons") or []
    reason_text = ", ".join(str(reason) for reason in reasons)
    ending_instruction = "If this is scene 10, make it a warm ending with afterglow.\n" if bool(scene.get("_i_ending_scene")) else ""
    return (
        "You are a strict JSON repair tool and Korean fairy-tale cleanup writer.\n"
        "Return one valid JSON object only. No markdown. No explanation.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"Cleanup reasons: {reason_text}\n"
        f"{ending_instruction}"
        "story_sentence must be Korean only, with no English words or meta/analysis language.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Do not repeat caption words unless they are visible in the current scene.\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_STORY_SENTENCE:\n{current_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_captionless_cleanup_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_i_previous_sentence") or "").strip()
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    reasons = scene.get("_i_quality_reasons") or []
    reason_text = ", ".join(str(reason) for reason in reasons)
    position_role = str(scene.get("_scene_position_role") or _i_scene_position_role(scene_index))
    ending_instruction = ""
    if bool(scene.get("_i_ending_scene")):
        ending_instruction = (
            "This is scene 10, the ending. Close the story warmly. Do not start a new event. "
            "Do not list visible objects one by one. Give a gentle afterglow.\n"
        )
    return (
        "Experiment I quality cleanup: You are a strict Korean fairy-tale cleanup writer.\n"
        "Rewrite current_story_sentence into clean Korean children's fairy-tale prose.\n"
        f"{_fairy_tale_speech_style_policy()}"
        f"Cleanup reasons: {reason_text}\n"
        f"{ending_instruction}"
        "Remove all English words. Replace them with natural Korean words.\n"
        "Remove meta or analysis language such as '이 장면은', '그림에는', scene, mood, emotion, story_role, or story_sentence.\n"
        "Do not introduce or repeat whole-story caption wording. This cleanup pass does not use story_caption.\n"
        "The current scene JSON and previous final sentence are the main sources.\n"
        "Do not copy the previous final sentence. Use at least one concrete visual clue from the current scene JSON.\n"
        "Use scene_position_role only for story rhythm, not as a plot requirement.\n"
        "Keep the paragraph warm, concrete, and easy for a child to read.\n"
        "story_sentence must contain 2 to 5 short Korean sentences, preferably exactly 3.\n"
        "Output exactly one JSON object only with scene_index and story_sentence.\n"
        f"scene_index must be {scene_index}.\n\n"
        f"scene_position_role:\n{position_role}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_story_sentence:\n{current_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_captionless_cleanup_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    previous_sentence = str(scene.get("_i_previous_sentence") or "").strip()
    reasons = scene.get("_i_quality_reasons") or []
    reason_text = ", ".join(str(reason) for reason in reasons)
    position_role = str(scene.get("_scene_position_role") or _i_scene_position_role(scene_index))
    ending_instruction = "If this is scene 10, make it a warm ending with afterglow.\n" if bool(scene.get("_i_ending_scene")) else ""
    return (
        "You are a strict JSON repair tool and Korean fairy-tale cleanup writer.\n"
        "Return one valid JSON object only. No markdown. No explanation.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        f"scene_index must be {scene_index}.\n"
        f"Cleanup reasons: {reason_text}\n"
        f"{ending_instruction}"
        "story_sentence must be Korean only, with no English words or meta/analysis language.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Do not introduce or repeat whole-story caption wording. This cleanup pass does not use story_caption.\n"
        f"scene_position_role: {position_role}\n"
        "Do not copy the previous final sentence. Use current-scene visual clues.\n"
        "Avoid a one-sentence result.\n"
        "Required shape:\n"
        "{\n"
        f'  "scene_index": {scene_index},\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_STORY_SENTENCE:\n{current_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _build_i_ending_cleanup_prompt(scene: dict[str, Any]) -> str:
    scene_index = int(scene["scene_index"])
    previous_sentence = str(scene.get("_i_previous_sentence") or "").strip()
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    reasons = scene.get("_i_quality_reasons") or []
    reason_text = ", ".join(str(reason) for reason in reasons)
    position_role = str(scene.get("_scene_position_role") or _i_scene_position_role(scene_index))
    return (
        "Experiment I ending cleanup: You are a warm Korean children's fairy-tale writer.\n"
        "Rewrite only scene 10 as the ending paragraph of the whole story.\n"
        f"{_fairy_tale_speech_style_policy()}"
        f"Ending cleanup reasons: {reason_text}\n"
        "Do not use story_caption in this ending cleanup.\n"
        "Do not start a new event, new goal, new discovery, or new problem.\n"
        "Close the flow from the previous final paragraph with warmth, comfort, and afterglow.\n"
        "Show that at least one person, place, or nearby being has become calm or safe.\n"
        "Do not list visible objects one by one. Use the current scene JSON as grounding.\n"
        "The last sentence should feel like a gentle closing line.\n"
        "story_sentence must contain 2 to 5 short Korean sentences, preferably exactly 3.\n"
        "Output exactly one JSON object only with scene_index and story_sentence.\n"
        "scene_index must be 10.\n\n"
        f"scene_position_role:\n{position_role}\n\n"
        f"previous_final_story_sentence:\n{previous_sentence}\n\n"
        f"current_scene_10_story_sentence:\n{current_sentence}\n\n"
        f"current_scene:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n"
    )


def _build_i_ending_cleanup_repair_prompt(raw_response: str, scene: dict[str, Any]) -> str:
    previous_sentence = str(scene.get("_i_previous_sentence") or "").strip()
    current_sentence = str(scene.get("_i_current_sentence") or "").strip()
    reasons = scene.get("_i_quality_reasons") or []
    reason_text = ", ".join(str(reason) for reason in reasons)
    return (
        "You are a strict JSON repair tool and Korean fairy-tale ending editor.\n"
        "Return one valid JSON object only. No markdown. No explanation.\n"
        "The JSON must contain exactly these keys: scene_index, story_sentence.\n"
        "scene_index must be 10.\n"
        f"Ending cleanup reasons: {reason_text}\n"
        "Do not use story_caption in this repair.\n"
        "story_sentence must close the story warmly without starting a new event.\n"
        f"{_fairy_tale_speech_style_policy()}"
        "Avoid a one-sentence result. Use Korean only, with no meta/analysis language.\n"
        "Required shape:\n"
        "{\n"
        '  "scene_index": 10,\n'
        '  "story_sentence": ""\n'
        "}\n\n"
        f"PREVIOUS_FINAL_STORY_SENTENCE:\n{previous_sentence}\n\n"
        f"CURRENT_SCENE_10_STORY_SENTENCE:\n{current_sentence}\n\n"
        f"CURRENT_SCENE_CONTEXT:\n{json.dumps(_compact_scene(scene), ensure_ascii=False, indent=2)}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _looks_like_placeholder(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    lowered = text.lower()
    robust_placeholder_patterns = (
        "\ud574\ub2f9\ud558\ub294 \ubb38\ub2e8",
        "\ub3d9\ud654 \ubb38\uc7a5",
        "\uc2e4\uc81c \ub3d9\ud654 \ubb38\ub2e8",
        "\ud55c\uad6d\uc5b4 \ub3d9\ud654 \uc81c\ubaa9",
        "\ube48 \ubb38\uc790\uc5f4",
        "one non-empty korean",
        "actual korean story",
        "actual korean prose",
        "write real korean prose",
        "for this scene",
        "non-empty korean",
        "story_sentence",
        "placeholder",
        "...",
    )
    if any(pattern in lowered for pattern in robust_placeholder_patterns):
        return True
    placeholder_patterns = (
        "해당하는 문단",
        "동화 문장",
        "실제 동화 문단",
        "짧은 한국어 동화 제목",
        "story_sentence",
        "placeholder",
        "...",
    )
    return any(pattern in text for pattern in placeholder_patterns)


def _english_words(value: str) -> list[str]:
    return re.findall(r"[A-Za-z]{2,}", value)


def _unique_english_words(value: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in _english_words(value):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _unique_english_replacement_units(value: str) -> list[str]:
    units: list[str] = []
    seen: set[str] = set()
    for unit in re.findall(r"[A-Za-z]{2,}(?:[\uac00-\ud7a3]+)?", value):
        key = unit.lower()
        if key in seen:
            continue
        seen.add(key)
        units.append(unit)
    return units


def _replace_english_terms(value: str, translations: dict[str, str]) -> str:
    result = value
    for term in sorted(translations, key=len, reverse=True):
        translated = translations[term].strip()
        if not translated:
            continue
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", flags=re.I)
        result = pattern.sub(translated, result)
    return result


def _has_meta_language(value: str) -> bool:
    lowered = value.lower()
    robust_meta_patterns = (
        "\uc774 \uc7a5\uba74\uc740",
        "\uadf8\ub9bc\uc5d0\ub294",
        "\uadf8\ub9bc\uc5d0\uc11c",
        "\ubcf4\uc785\ub2c8\ub2e4",
        "\uc124\uba85\ud569\ub2c8\ub2e4",
        "\ubb18\uc0ac\ud569\ub2c8\ub2e4",
        "\uadf8\ub9bc \uadfc\uac70",
    )
    if any(pattern in lowered for pattern in robust_meta_patterns):
        return True
    meta_patterns = (
        "이 장면은",
        "그림에는",
        "그림에서",
        "보입니다",
        "묘사합니다",
        "설명합니다",
        "scene",
        "mood",
        "emotion",
        "story_role",
        "story_sentence",
        "characters",
        "objects",
        "setting",
        "uncertain",
        "json",
        "qwen",
        "prompt",
    )
    return any(pattern in lowered for pattern in meta_patterns)


def _caption_keywords(story_caption: str) -> list[str]:
    stopwords = {
        "이야기",
        "작은",
        "아이",
        "아이가",
        "주인공",
        "그림",
        "동화",
        "대한",
        "하는",
        "있는",
        "없는",
        "되어",
        "된다",
        "준다",
        "주는",
    }
    keywords: list[str] = []
    for raw in re.findall(r"[가-힣]{2,}", story_caption):
        token = raw
        for suffix in ("에서", "으로", "에게", "들을", "까지", "부터", "처럼", "만큼"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
        for suffix in ("은", "는", "이", "가", "을", "를", "와", "과", "의", "로", "도", "만", "한"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
        if len(token) >= 2 and token not in stopwords and token not in keywords:
            keywords.append(token)
    return keywords[:8]


def _scene_supports_keyword(scene: dict[str, Any], keyword: str) -> bool:
    scene_text = json.dumps(_compact_scene(scene), ensure_ascii=False)
    return keyword in scene_text


def _caption_keywords_for_gate(story_caption: str) -> list[str]:
    stopwords = {
        "이야기",
        "작은",
        "아이",
        "아이가",
        "주인공",
        "그림",
        "동화",
        "숲",
        "숲속",
        "숲에서",
        "숲과",
        "길",
        "잃은",
        "어두운",
        "정성껏",
        "발견",
        "발견한",
        "키워",
        "밝혀",
        "주는",
        "한다",
        "하는",
    }
    keywords: list[str] = []
    for raw in re.findall(r"[가-힣]{2,}", story_caption):
        token = raw
        for suffix in ("에서", "으로", "에게", "들을", "까지", "부터", "처럼", "만큼"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
        for suffix in ("은", "는", "이", "가", "을", "를", "와", "과", "에", "로", "도", "만", "의"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
        if len(token) >= 2 and token not in stopwords and token not in keywords:
            keywords.append(token)
    return keywords[:6]


def _ending_quality_reasons_for_gate(value: str) -> list[str]:
    reasons: list[str] = []
    ending_cues = (
        "따뜻",
        "함께",
        "안심",
        "미소",
        "고마",
        "편안",
        "돌아",
        "집",
        "반짝",
        "웃음",
        "잠",
        "평화",
        "마지막",
        "여운",
    )
    new_event_cues = ("갑자기", "새로운", "처음", "발견", "시작", "나타", "낯선")
    if not any(cue in value for cue in ending_cues):
        reasons.append("ending")
    if any(cue in value for cue in new_event_cues):
        reasons.append("ending")
    animal_mentions = sum(value.count(name) for name in ("토끼", "강아지", "다람쥐", "고양이", "동물"))
    if animal_mentions >= 3 and not any(cue in value for cue in ("함께", "안심", "따뜻", "평화")):
        reasons.append("ending")
    return reasons


def _caption_repetition_reasons(
    value: str,
    scene: dict[str, Any],
    story_caption: str,
    caption_usage_counts: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    for keyword in _caption_keywords_for_gate(story_caption):
        if keyword not in value:
            continue
        if not _scene_supports_keyword(scene, keyword):
            reasons.append("caption_repetition")
            break
        if caption_usage_counts.get(keyword, 0) >= 2:
            reasons.append("caption_repetition")
            break
    return reasons


def _ending_quality_reasons(value: str) -> list[str]:
    reasons: list[str] = []
    ending_cues = (
        "따뜻",
        "함께",
        "안심",
        "미소",
        "고마",
        "편안",
        "돌아",
        "집",
        "반짝",
        "속삭",
        "잠잠",
        "평화",
        "마지막",
        "여운",
    )
    new_event_cues = ("갑자기", "새로운", "처음", "발견", "시작", "나타났", "떠났")
    if not any(cue in value for cue in ending_cues):
        reasons.append("ending")
    if any(cue in value for cue in new_event_cues):
        reasons.append("ending")
    animal_mentions = sum(value.count(name) for name in ("토끼", "강아지", "원숭이", "고양이", "동물"))
    if animal_mentions >= 3 and not any(cue in value for cue in ("함께", "안심", "따뜻", "평화")):
        reasons.append("ending")
    return list(dict.fromkeys(reasons))


def _quality_gate_reasons(
    value: str,
    scene: dict[str, Any],
    story_caption: str,
    caption_usage_counts: dict[str, int],
    *,
    check_ending: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if _english_words(value):
        reasons.append("english")
    if _has_meta_language(value) or _looks_like_placeholder(value):
        reasons.append("meta_language")
    reasons.extend(_caption_repetition_reasons(value, scene, story_caption, caption_usage_counts))
    if check_ending:
        reasons.extend(_ending_quality_reasons_for_gate(value))
    return list(dict.fromkeys(reasons))


def _similarity_tokens(value: str) -> set[str]:
    return set(re.findall(r"[가-힣A-Za-z0-9]{2,}", value.lower()))


def _i_repetition_reasons(value: str, previous_sentence: str) -> list[str]:
    if not previous_sentence:
        return []
    current = re.sub(r"\s+", " ", value).strip()
    previous = re.sub(r"\s+", " ", previous_sentence).strip()
    if not current or not previous:
        return []
    if current == previous:
        return ["repetition"]
    if len(current) > 24 and (current in previous or previous in current):
        return ["repetition"]
    current_tokens = _similarity_tokens(current)
    previous_tokens = _similarity_tokens(previous)
    if len(current_tokens) < 5 or len(previous_tokens) < 5:
        return []
    overlap = len(current_tokens & previous_tokens) / max(1, min(len(current_tokens), len(previous_tokens)))
    if overlap >= 0.7:
        return ["repetition"]
    return []


def _i_quality_gate_reasons(
    value: str,
    scene: dict[str, Any],
    story_caption: str,
    caption_usage_counts: dict[str, int],
    *,
    previous_sentence: str = "",
    check_ending: bool = False,
) -> list[str]:
    reasons: list[str] = []
    reasons.extend(_i_repetition_reasons(value, previous_sentence))
    if _english_words(value):
        reasons.append("english")
    if _has_meta_language(value) or _looks_like_placeholder(value):
        reasons.append("meta_language")
    if check_ending:
        reasons.extend(_ending_quality_reasons_for_gate(value))
    return list(dict.fromkeys(reasons))


def _record_caption_usage(value: str, story_caption: str, caption_usage_counts: dict[str, int]) -> None:
    for keyword in _caption_keywords_for_gate(story_caption):
        if keyword in value:
            caption_usage_counts[keyword] = caption_usage_counts.get(keyword, 0) + 1


def _caption_usage_without_sentence(
    caption_usage_counts: dict[str, int],
    value: str,
    story_caption: str,
) -> dict[str, int]:
    adjusted = dict(caption_usage_counts)
    for keyword in _caption_keywords_for_gate(story_caption):
        if keyword in value and adjusted.get(keyword, 0) > 0:
            adjusted[keyword] -= 1
    return adjusted


def _sentence_mark_count(value: str) -> int:
    return len(re.findall(r"[.!?。！？]", value))


def _scene_story_from_payload(
    payload: dict[str, Any],
    expected_index: int,
) -> dict[str, Any]:
    scene_index = int(payload.get("scene_index") or expected_index)
    if scene_index != expected_index:
        raise ValueError(f"EXAONE scene_index mismatch: expected {expected_index}, got {scene_index}")
    story_sentence = str(payload.get("story_sentence") or "").strip()
    if _looks_like_placeholder(story_sentence):
        raise ValueError(f"EXAONE scene {expected_index} returned placeholder or empty story_sentence.")
    return {"scene_index": scene_index, "story_sentence": story_sentence}


def _has_korean(value: str) -> bool:
    return bool(re.search(r"[가-힣]", value))


def _decode_json_string_fragment(raw: str) -> str:
    fragment = re.split(r"\s*```", raw, maxsplit=1)[0].strip()
    fragment = re.sub(r"[\s,}\]]+$", "", fragment).strip()
    if not fragment:
        return ""
    try:
        return json.loads(f'"{fragment}"')
    except json.JSONDecodeError:
        return fragment.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").strip()


def _partial_json_string_field(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
    if not match:
        return None
    chars: list[str] = []
    escaped = False
    closed = False
    for char in text[match.end() :]:
        if escaped:
            chars.append("\\" + char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            closed = True
            break
        else:
            chars.append(char)
    if escaped:
        chars.append("\\")
    raw = "".join(chars)
    if not closed:
        raw = re.split(r"\s*```", raw, maxsplit=1)[0]
    return _decode_json_string_fragment(raw)


def _partial_scene_story_payload(text: str, expected_index: int) -> dict[str, Any] | None:
    scene_index_matches = list(re.finditer(r'"scene_index"\s*:\s*(\d+)', text))
    for match in reversed(scene_index_matches):
        scene_index = int(match.group(1))
        if scene_index != expected_index:
            continue
        segment = text[match.start() :]
        story_sentence = _partial_json_string_field(segment, "story_sentence")
        if story_sentence and story_sentence.strip():
            return {"scene_index": scene_index, "story_sentence": story_sentence.strip()}
    return None


def _extract_scene_story_json(
    text: str,
    expected_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _json_object_candidates(text)
    if not candidates:
        payload = _partial_scene_story_payload(text, expected_index)
        if payload is None:
            raise ValueError("EXAONE response did not contain a JSON object.")
        parsed = _scene_story_from_payload(
            payload,
            expected_index,
        )
        return payload, parsed
    last_error: Exception | None = None
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = ValueError("EXAONE JSON response was not an object.")
            continue
        try:
            parsed = _scene_story_from_payload(
                payload,
                expected_index,
            )
        except (ValueError, TypeError) as exc:
            last_error = exc
            continue
        return payload, parsed
    partial_payload = _partial_scene_story_payload(text, expected_index)
    if partial_payload is not None:
        try:
            partial_parsed = _scene_story_from_payload(
                partial_payload,
                expected_index,
            )
        except (ValueError, TypeError) as exc:
            last_error = exc
        else:
            return partial_payload, partial_parsed
    if last_error:
        raise last_error
    raise ValueError("EXAONE response did not contain valid scene JSON.")


def _extract_title_json(text: str) -> tuple[dict[str, Any], str]:
    candidates = _json_object_candidates(text)
    if not candidates:
        raise ValueError("EXAONE response did not contain a JSON object.")
    last_error: Exception | None = None
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            last_error = ValueError("EXAONE JSON response was not an object.")
            continue
        title = str(payload.get("title") or "").strip()
        if title and not _looks_like_placeholder(title):
            return payload, title
        last_error = ValueError("EXAONE title JSON had no usable title.")
    if last_error:
        raise last_error
    raise ValueError("EXAONE response did not contain valid title JSON.")


def _run_exaone_scene_story(
    scene: dict[str, Any],
    *,
    experiment_name: str = "Experiment_E",
    scene_prompt_builder: Callable[[dict[str, Any]], str] = _build_e_scene_prompt,
    repair_prompt_builder: Callable[[str, dict[str, Any]], str] = _build_e_scene_repair_prompt,
    max_new_tokens: int = 350,
    step_prefix: str = "EXAONE",
    step_label: str = "scene",
) -> dict[str, Any]:
    scene_index = int(scene["scene_index"])
    prompt = scene_prompt_builder(scene)
    with timed_step(
        f"{step_prefix}-{scene_index:02d}",
        f"{experiment_name} {step_label} {scene_index} EXAONE GGUF generation",
        experiment=experiment_name,
        model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
    ):
        raw_response = _run_exaone_gguf_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            timeout=300,
            context_size=4096,
        )
    json_repair_used = False
    try:
        payload, parsed = _extract_scene_story_json(raw_response, scene_index)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        lenient_initial_payload: dict[str, Any] | None = None
        lenient_initial_parsed: dict[str, Any] | None = None
        try:
            lenient_initial_payload, lenient_initial_parsed = _extract_scene_story_json(
                raw_response,
                scene_index,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        repair_prompt = repair_prompt_builder(raw_response, scene)
        with timed_step(
            f"{step_prefix}-{scene_index:02d}-repair",
            f"{experiment_name} {step_label} {scene_index} EXAONE JSON repair",
            experiment=experiment_name,
            model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
        ):
            repair_response = _run_exaone_gguf_prompt(
                repair_prompt,
                max_new_tokens=max_new_tokens,
                timeout=300,
                context_size=4096,
            )
        json_repair_used = True
        try:
            payload, parsed = _extract_scene_story_json(
                repair_response,
                scene_index,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as repair_exc:
            if lenient_initial_payload is not None and lenient_initial_parsed is not None:
                payload = lenient_initial_payload
                parsed = lenient_initial_parsed
                raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
                return {
                    "scene_index": scene_index,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_result": parsed,
                    "json_repair_used": json_repair_used,
                    "llama_runtime": get_last_llama_runtime(),
                }
            raise RuntimeError(
                f"json_repair_failed: EXAONE scene {scene_index} did not return valid scene JSON, and JSON repair also failed. "
                f"initial_error={exc}; repair_error={repair_exc}; "
                f"cleaned_response_head={raw_response[:800]!r}; repair_response_head={repair_response[:800]!r}; "
                f"llama_runtime={get_last_llama_runtime()!r}"
            ) from repair_exc
        raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
    return {
        "scene_index": scene_index,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_result": parsed,
        "json_repair_used": json_repair_used,
        "llama_runtime": get_last_llama_runtime(),
    }


def _build_e_title_prompt(body: str) -> str:
    return (
        "아래 한국어 동화 전체 본문을 읽고 짧은 한국어 제목 하나를 만드세요.\n"
        "제목은 15자 이내가 좋고, 따뜻한 어린이 동화 느낌이어야 합니다.\n"
        "반드시 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        "{\n"
        '  "title": "짧은 한국어 동화 제목"\n'
        "}\n\n"
        f"story_body:\n{body}\n"
    )


def _build_f_title_prompt(body: str) -> str:
    return (
        "당신은 어린이 동화를 쓰는 따뜻한 동화작가입니다.\n"
        "아래 한국어 동화 전체 본문을 읽고 짧은 한국어 제목 하나를 만드세요.\n"
        "제목은 15자 이내가 좋고, 아이가 읽고 싶어지는 동화 느낌이어야 합니다.\n"
        "반드시 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        "{\n"
        '  "title": "짧은 한국어 동화 제목"\n'
        "}\n\n"
        f"story_body:\n{body}\n"
    )


def _build_g_title_prompt(body: str) -> str:
    return (
        "당신은 어린이 동화를 쓰는 따뜻한 동화작가입니다.\n"
        "아래 한국어 동화 전체 본문을 읽고 짧은 한국어 제목 하나를 만드세요.\n"
        "먼저 내부적으로 이야기의 중심 정서와 1번부터 10번까지의 흐름을 확인하세요.\n"
        "단, 이 점검 과정은 출력하지 말고 최종 JSON에만 반영하세요.\n"
        "제목은 15자 이내가 좋고, 아이가 읽고 싶어지는 동화 느낌이어야 합니다.\n"
        "반드시 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        "{\n"
        '  "title": "짧은 한국어 동화 제목"\n'
        "}\n\n"
        f"story_body:\n{body}\n"
    )


def _generate_e_title(
    body: str,
    *,
    experiment_name: str = "Experiment_E",
    title_prompt_builder: Callable[[str], str] = _build_e_title_prompt,
) -> dict[str, Any]:
    prompt = title_prompt_builder(body)
    with timed_step(
        "EXAONE-title",
        f"{experiment_name} EXAONE title generation",
        experiment=experiment_name,
        model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
    ):
        raw_response = _run_exaone_gguf_prompt(
            prompt,
            max_new_tokens=120,
            timeout=120,
            context_size=4096,
        )
    try:
        payload, title = _extract_title_json(raw_response)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "json_parse_failed: EXAONE title response did not contain a usable title JSON. "
            f"parse_error={exc}; cleaned_response_head={raw_response[:800]!r}; "
            f"llama_runtime={get_last_llama_runtime()!r}"
        ) from exc
    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_result": payload,
        "title": title,
        "llama_runtime": get_last_llama_runtime(),
    }


def _run_exaone_per_scene_experiment(
    scenes: list[dict[str, Any]],
    *,
    experiment_name: str = "Experiment_E",
    prompt_strategy: str = "per_scene_exaone_then_title",
    scene_prompt_builder: Callable[[dict[str, Any]], str] = _build_e_scene_prompt,
    repair_prompt_builder: Callable[[str, dict[str, Any]], str] = _build_e_scene_repair_prompt,
    title_prompt_builder: Callable[[str], str] = _build_e_title_prompt,
) -> dict[str, Any]:
    _ensure_exaone_gguf_available()
    ordered_scenes = sorted(scenes, key=lambda item: int(item["scene_index"]))
    scene_results = [
        _run_exaone_scene_story(
            scene,
            experiment_name=experiment_name,
            scene_prompt_builder=scene_prompt_builder,
            repair_prompt_builder=repair_prompt_builder,
        )
        for scene in ordered_scenes
    ]
    scene_sentences = [item["parsed_result"]["story_sentence"] for item in scene_results]
    body = "\n\n".join(scene_sentences)
    title_result = _generate_e_title(
        body,
        experiment_name=experiment_name,
        title_prompt_builder=title_prompt_builder,
    )
    json_repair_used = any(item["json_repair_used"] for item in scene_results)
    return {
        "prompt_strategy": prompt_strategy,
        "exaone_prompt": [item["prompt"] for item in scene_results],
        "exaone_raw_response": "\n\n".join(
            f"[scene {item['scene_index']}]\n{item['raw_response']}" for item in scene_results
        )
        + f"\n\n[title]\n{title_result['raw_response']}",
        "llama_runtime": title_result.get("llama_runtime") or get_last_llama_runtime(),
        "parsed_result": {
            "scene_results": [item["parsed_result"] for item in scene_results],
            "title_result": title_result["parsed_result"],
        },
        "json_repair_used": json_repair_used,
        "story": {
            "title": title_result["title"],
            "body": body,
            "scene_sentences": scene_sentences,
            "grounding_notes": [],
        },
        "structure": {
            "mode": "per_scene_exaone",
            "scene_count": len(ordered_scenes),
            "exaone_scene_calls": len(scene_results),
            "exaone_title_calls": 1,
        },
        "plan": {
            "method": "Qwen scene JSON -> EXAONE per-scene paragraphs -> code joins body -> EXAONE title",
            "scene_order": [scene["image_id"] for scene in ordered_scenes],
            "scene_max_new_tokens": 350,
            "title_max_new_tokens": 120,
        },
        "experiment_method": experiment_name,
    }


def _run_exaone_g_experiment(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    _ensure_exaone_gguf_available()
    experiment_name = "Experiment_G"
    ordered_scenes = sorted(scenes, key=lambda item: int(item["scene_index"]))
    initial_results = [
        _run_exaone_scene_story(
            scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_g_scene_prompt,
            repair_prompt_builder=_build_g_scene_repair_prompt,
            step_prefix="EXAONE-initial",
            step_label="initial scene",
        )
        for scene in ordered_scenes
    ]

    initial_by_index = {int(item["scene_index"]): item for item in initial_results}
    final_scene_results = [initial_results[0]["parsed_result"]]
    refined_results: list[dict[str, Any]] = []
    previous_final_sentence = initial_results[0]["parsed_result"]["story_sentence"]

    for scene in ordered_scenes[1:]:
        scene_index = int(scene["scene_index"])
        initial_sentence = initial_by_index[scene_index]["parsed_result"]["story_sentence"]
        refinement_scene = {
            **scene,
            "_g_previous_sentence": previous_final_sentence,
            "_g_initial_sentence": initial_sentence,
        }
        refined_result = _run_exaone_scene_story(
            refinement_scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_g_refinement_prompt,
            repair_prompt_builder=_build_g_refinement_repair_prompt,
            step_prefix="EXAONE-refine",
            step_label="refinement scene",
        )
        refined_results.append(refined_result)
        final_scene_results.append(refined_result["parsed_result"])
        previous_final_sentence = refined_result["parsed_result"]["story_sentence"]

    scene_sentences = [item["story_sentence"] for item in final_scene_results]
    body = "\n\n".join(scene_sentences)
    title_result = _generate_e_title(
        body,
        experiment_name=experiment_name,
        title_prompt_builder=_build_g_title_prompt,
    )
    json_repair_used = any(item["json_repair_used"] for item in initial_results + refined_results)
    return {
        "prompt_strategy": "cot_persona_scene_position_sequential_refinement",
        "exaone_prompt": {
            "initial_scene_prompts": [item["prompt"] for item in initial_results],
            "refinement_prompts": [item["prompt"] for item in refined_results],
            "title_prompt": title_result["prompt"],
        },
        "exaone_raw_response": "\n\n".join(
            f"[initial scene {item['scene_index']}]\n{item['raw_response']}" for item in initial_results
        )
        + "\n\n"
        + "\n\n".join(
            f"[refine scene {item['scene_index']}]\n{item['raw_response']}" for item in refined_results
        )
        + f"\n\n[title]\n{title_result['raw_response']}",
        "llama_runtime": title_result.get("llama_runtime") or get_last_llama_runtime(),
        "parsed_result": {
            "initial_scene_results": [item["parsed_result"] for item in initial_results],
            "refined_scene_results": [item["parsed_result"] for item in refined_results],
            "final_scene_results": final_scene_results,
            "title_result": title_result["parsed_result"],
        },
        "json_repair_used": json_repair_used,
        "story": {
            "title": title_result["title"],
            "body": body,
            "scene_sentences": scene_sentences,
            "grounding_notes": [],
        },
        "structure": {
            "mode": "per_scene_exaone_with_sequential_refinement",
            "scene_count": len(ordered_scenes),
            "exaone_initial_scene_calls": len(initial_results),
            "exaone_refinement_calls": len(refined_results),
            "exaone_title_calls": 1,
            "exaone_total_calls": len(initial_results) + len(refined_results) + 1,
        },
        "plan": {
            "method": (
                "Qwen scene JSON -> EXAONE initial per-scene paragraphs -> "
                "EXAONE sequential refinement with previous final sentence -> code joins body -> EXAONE title"
            ),
            "scene_order": [scene["image_id"] for scene in ordered_scenes],
            "scene_max_new_tokens": 350,
            "refinement_max_new_tokens": 350,
            "title_max_new_tokens": 120,
        },
        "experiment_method": experiment_name,
    }


def _run_exaone_h_experiment(scenes: list[dict[str, Any]], story_caption: str) -> dict[str, Any]:
    _ensure_exaone_gguf_available()
    experiment_name = "Experiment_H"
    story_caption = story_caption.strip()
    if not story_caption:
        raise ValueError("Experiment H requires a non-empty story caption.")
    ordered_scenes = [
        {**scene, "_story_caption": story_caption}
        for scene in sorted(scenes, key=lambda item: int(item["scene_index"]))
    ]
    initial_results = [
        _run_exaone_scene_story(
            scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_h_scene_prompt,
            repair_prompt_builder=_build_h_scene_repair_prompt,
            step_prefix="EXAONE-initial",
            step_label="initial scene",
        )
        for scene in ordered_scenes
    ]

    initial_by_index = {int(item["scene_index"]): item for item in initial_results}
    final_scene_results = [initial_results[0]["parsed_result"]]
    refined_results: list[dict[str, Any]] = []
    previous_final_sentence = initial_results[0]["parsed_result"]["story_sentence"]

    for scene in ordered_scenes[1:]:
        scene_index = int(scene["scene_index"])
        initial_sentence = initial_by_index[scene_index]["parsed_result"]["story_sentence"]
        refinement_scene = {
            **scene,
            "_g_previous_sentence": previous_final_sentence,
            "_g_initial_sentence": initial_sentence,
        }
        refined_result = _run_exaone_scene_story(
            refinement_scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_h_refinement_prompt,
            repair_prompt_builder=_build_h_refinement_repair_prompt,
            max_new_tokens=H_REFINEMENT_MAX_NEW_TOKENS,
            step_prefix="EXAONE-refine",
            step_label="refinement scene",
        )
        refined_results.append(refined_result)
        final_scene_results.append(refined_result["parsed_result"])
        previous_final_sentence = refined_result["parsed_result"]["story_sentence"]

    scene_sentences = [item["story_sentence"] for item in final_scene_results]
    body = "\n\n".join(scene_sentences)
    title_result = _generate_e_title(
        body,
        experiment_name=experiment_name,
        title_prompt_builder=_build_g_title_prompt,
    )
    json_repair_used = any(item["json_repair_used"] for item in initial_results + refined_results)
    return {
        "prompt_strategy": "weak_caption_refinement_only_fairy_tale_writer",
        "story_caption": story_caption,
        "exaone_prompt": {
            "story_caption": story_caption,
            "initial_scene_prompts": [item["prompt"] for item in initial_results],
            "refinement_prompts": [item["prompt"] for item in refined_results],
            "title_prompt": title_result["prompt"],
        },
        "exaone_raw_response": "\n\n".join(
            f"[initial scene {item['scene_index']}]\n{item['raw_response']}" for item in initial_results
        )
        + "\n\n"
        + "\n\n".join(
            f"[refine scene {item['scene_index']}]\n{item['raw_response']}" for item in refined_results
        )
        + f"\n\n[title]\n{title_result['raw_response']}",
        "llama_runtime": title_result.get("llama_runtime") or get_last_llama_runtime(),
        "parsed_result": {
            "story_caption": story_caption,
            "initial_scene_results": [item["parsed_result"] for item in initial_results],
            "refined_scene_results": [item["parsed_result"] for item in refined_results],
            "final_scene_results": final_scene_results,
            "title_result": title_result["parsed_result"],
        },
        "json_repair_used": json_repair_used,
        "story": {
            "title": title_result["title"],
            "body": body,
            "scene_sentences": scene_sentences,
            "grounding_notes": [],
        },
        "structure": {
            "mode": "weak_caption_refinement_only_sequential_refinement",
            "scene_count": len(ordered_scenes),
            "story_caption_used": True,
            "story_caption_stage": "refinement_only",
            "refinement_persona": "fairy_tale_writer",
            "exaone_initial_scene_calls": len(initial_results),
            "exaone_refinement_calls": len(refined_results),
            "exaone_title_calls": 1,
            "exaone_total_calls": len(initial_results) + len(refined_results) + 1,
        },
        "plan": {
            "method": (
                "Qwen scene JSON -> EXAONE initial per-scene paragraphs without story_caption -> "
                "EXAONE sequential refinement with previous final sentence, weak story_caption guidance, and fairy-tale writer style -> "
                "code joins body -> EXAONE title"
            ),
            "story_caption": story_caption,
            "story_caption_stage": "refinement_only",
            "scene_order": [scene["image_id"] for scene in ordered_scenes],
            "scene_max_new_tokens": 350,
            "refinement_max_new_tokens": H_REFINEMENT_MAX_NEW_TOKENS,
            "title_max_new_tokens": 120,
        },
        "experiment_method": experiment_name,
    }


def _maybe_run_i_cleanup(
    scene: dict[str, Any],
    parsed_result: dict[str, Any],
    *,
    experiment_name: str,
    stage: str,
    previous_final_sentence: str,
    story_caption: str,
    caption_usage_counts: dict[str, int],
    check_ending: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sentence = str(parsed_result.get("story_sentence") or "")
    reasons = _quality_gate_reasons(
        sentence,
        scene,
        story_caption,
        caption_usage_counts,
        check_ending=check_ending,
    )
    if not reasons:
        return parsed_result, None

    cleanup_scene = {
        **scene,
        "_story_caption": story_caption,
        "_i_previous_sentence": previous_final_sentence,
        "_i_current_sentence": sentence,
        "_i_quality_reasons": reasons,
        "_i_ending_scene": check_ending,
    }
    cleanup_result = _run_exaone_scene_story(
        cleanup_scene,
        experiment_name=experiment_name,
        scene_prompt_builder=_build_i_captionless_cleanup_prompt,
        repair_prompt_builder=_build_i_captionless_cleanup_repair_prompt,
        max_new_tokens=I_CLEANUP_MAX_NEW_TOKENS,
        step_prefix=f"EXAONE-cleanup-{stage}",
        step_label=f"{stage} cleanup",
    )
    cleaned = cleanup_result["parsed_result"]
    remaining_reasons = _quality_gate_reasons(
        str(cleaned.get("story_sentence") or ""),
        scene,
        story_caption,
        caption_usage_counts,
        check_ending=check_ending,
    )
    if remaining_reasons:
        raise RuntimeError(
            "exaone_output_invalid: EXAONE cleanup response did not satisfy quality gates. "
            f"scene_index={int(scene['scene_index'])}; stage={stage}; "
            f"initial_reasons={reasons!r}; remaining_reasons={remaining_reasons!r}; "
            f"cleaned_response_head={cleanup_result['raw_response'][:800]!r}; "
            f"llama_runtime={cleanup_result.get('llama_runtime')!r}"
        )
    cleanup_record = {
        "stage": stage,
        "scene_index": int(scene["scene_index"]),
        "reasons": reasons,
        "remaining_reasons": remaining_reasons,
        "prompt": cleanup_result["prompt"],
        "raw_response": cleanup_result["raw_response"],
        "parsed_result": cleaned,
        "json_repair_used": cleanup_result["json_repair_used"],
    }
    return cleaned, cleanup_record


def _run_i_english_term_translation(
    scene: dict[str, Any],
    parsed_result: dict[str, Any],
    *,
    experiment_name: str,
    stage: str,
    previous_final_sentence: str,
    story_caption: str,
    caption_usage_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sentence = str(parsed_result.get("story_sentence") or "")
    terms = _unique_english_replacement_units(sentence)
    if not terms:
        return parsed_result, None

    scene_index = int(scene["scene_index"])
    prompt = _build_i_english_term_translation_prompt(terms, sentence)
    raw_response = ""
    json_repair_used = False
    translations: dict[str, str] = {}
    translation_schema = _english_translation_json_schema()
    with timed_step(
        f"EXAONE-english-terms-{stage}-{scene_index:02d}",
        f"{experiment_name} {stage} scene {scene_index} English term translation",
        experiment=experiment_name,
        model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
    ):
        raw_response = _run_exaone_gguf_prompt(
            prompt,
            max_new_tokens=180,
            timeout=300,
            context_size=4096,
            json_schema=translation_schema,
        )
    try:
        translations = _extract_i_english_translations(raw_response, terms)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        repair_prompt = _build_i_english_term_translation_repair_prompt(raw_response, terms, sentence)
        with timed_step(
            f"EXAONE-english-terms-{stage}-{scene_index:02d}-repair",
            f"{experiment_name} {stage} scene {scene_index} English term translation JSON repair",
            experiment=experiment_name,
            model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
        ):
            repair_response = _run_exaone_gguf_prompt(
                repair_prompt,
                max_new_tokens=180,
                timeout=300,
                context_size=4096,
                json_schema=translation_schema,
            )
        json_repair_used = True
        raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
        try:
            translations = _extract_i_english_translations(repair_response, terms)
        except (json.JSONDecodeError, ValueError, TypeError) as repair_exc:
            return _run_i_english_sentence_rewrite(
                scene,
                parsed_result,
                experiment_name=experiment_name,
                stage=stage,
                previous_final_sentence=previous_final_sentence,
                story_caption=story_caption,
                caption_usage_counts=caption_usage_counts,
                previous_raw_response=raw_response,
                previous_error=(
                    "translation_json_repair_failed: "
                    f"terms={terms!r}; initial_error={exc}; repair_error={repair_exc}"
                ),
            )

    if not translations:
        return _run_i_english_sentence_rewrite(
            scene,
            parsed_result,
            experiment_name=experiment_name,
            stage=stage,
            previous_final_sentence=previous_final_sentence,
            story_caption=story_caption,
            caption_usage_counts=caption_usage_counts,
            previous_raw_response=raw_response,
            previous_error=f"translation_empty: terms={terms!r}",
        )

    translated_sentence = _replace_english_terms(sentence, translations)
    translated_result = {
        **parsed_result,
        "scene_index": scene_index,
        "story_sentence": translated_sentence,
    }
    remaining_reasons = _i_quality_gate_reasons(
        translated_sentence,
        scene,
        story_caption,
        caption_usage_counts,
        previous_sentence=previous_final_sentence,
        check_ending=False,
    )
    if remaining_reasons:
        if "english" in remaining_reasons:
            return _run_i_english_sentence_rewrite(
                scene,
                translated_result,
                experiment_name=experiment_name,
                stage=stage,
                previous_final_sentence=previous_final_sentence,
                story_caption=story_caption,
                caption_usage_counts=caption_usage_counts,
                previous_raw_response=raw_response,
                previous_error=f"term_translation_remaining_reasons={remaining_reasons!r}",
            )
        raise RuntimeError(
            "exaone_output_invalid: EXAONE English term translation did not satisfy quality gates. "
            f"scene_index={scene_index}; stage={stage}; remaining_reasons={remaining_reasons!r}; "
            f"cleaned_response_head={raw_response[:800]!r}; llama_runtime={get_last_llama_runtime()!r}"
        )
    record = {
        "stage": f"{stage}_english_translation",
        "scene_index": scene_index,
        "reasons": ["english"],
        "remaining_reasons": remaining_reasons,
        "english_terms": terms,
        "translations": translations,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_result": translated_result,
        "json_repair_used": json_repair_used,
    }
    return translated_result, record


def _run_i_english_sentence_rewrite(
    scene: dict[str, Any],
    parsed_result: dict[str, Any],
    *,
    experiment_name: str,
    stage: str,
    previous_final_sentence: str,
    story_caption: str,
    caption_usage_counts: dict[str, int],
    previous_raw_response: str = "",
    previous_error: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_index = int(scene["scene_index"])
    sentence_scene = {
        **scene,
        "_i_current_sentence": str(parsed_result.get("story_sentence") or ""),
    }
    prompt = _build_i_english_translation_prompt(sentence_scene)
    raw_response = ""
    json_repair_used = False
    with timed_step(
        f"EXAONE-english-rewrite-{stage}-{scene_index:02d}",
        f"{experiment_name} {stage} scene {scene_index} English sentence rewrite",
        experiment=experiment_name,
        model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
    ):
        raw_response = _run_exaone_gguf_prompt(
            prompt,
            max_new_tokens=260,
            timeout=300,
            context_size=4096,
            json_schema=_scene_story_json_schema(scene_index),
        )
    try:
        payload, rewritten = _extract_scene_story_json(raw_response, scene_index)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        repair_prompt = _build_i_english_translation_repair_prompt(raw_response, sentence_scene)
        with timed_step(
            f"EXAONE-english-rewrite-{stage}-{scene_index:02d}-repair",
            f"{experiment_name} {stage} scene {scene_index} English sentence rewrite JSON repair",
            experiment=experiment_name,
            model="EXAONE-4.0-1.2B-IQ4_XS.gguf",
        ):
            repair_response = _run_exaone_gguf_prompt(
                repair_prompt,
                max_new_tokens=260,
                timeout=300,
                context_size=4096,
                json_schema=_scene_story_json_schema(scene_index),
            )
        json_repair_used = True
        raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
        try:
            payload, rewritten = _extract_scene_story_json(repair_response, scene_index)
        except (json.JSONDecodeError, ValueError, TypeError) as repair_exc:
            raise RuntimeError(
                "json_repair_failed: EXAONE English sentence rewrite JSON repair failed. "
                f"scene_index={scene_index}; stage={stage}; initial_error={exc}; repair_error={repair_exc}; "
                f"previous_error={previous_error}; cleaned_response_head={raw_response[:800]!r}; "
                f"previous_translation_response_head={previous_raw_response[:800]!r}; "
                f"llama_runtime={get_last_llama_runtime()!r}"
            ) from repair_exc

    rewritten_result = {
        **parsed_result,
        "scene_index": scene_index,
        "story_sentence": rewritten["story_sentence"],
    }
    remaining_reasons = _i_quality_gate_reasons(
        rewritten_result["story_sentence"],
        scene,
        story_caption,
        caption_usage_counts,
        previous_sentence=previous_final_sentence,
        check_ending=False,
    )
    if remaining_reasons:
        raise RuntimeError(
            "exaone_output_invalid: EXAONE English sentence rewrite did not satisfy quality gates. "
            f"scene_index={scene_index}; stage={stage}; remaining_reasons={remaining_reasons!r}; "
            f"previous_error={previous_error}; cleaned_response_head={raw_response[:800]!r}; "
            f"previous_translation_response_head={previous_raw_response[:800]!r}; "
            f"llama_runtime={get_last_llama_runtime()!r}"
        )
    record = {
        "stage": f"{stage}_english_sentence_rewrite",
        "scene_index": scene_index,
        "reasons": ["english"],
        "remaining_reasons": remaining_reasons,
        "english_terms": _unique_english_replacement_units(str(parsed_result.get("story_sentence") or "")),
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_result": rewritten_result,
        "json_repair_used": json_repair_used,
        "previous_translation_response": previous_raw_response,
        "previous_translation_error": previous_error,
    }
    return rewritten_result, record


def _maybe_run_i_quality_cleanup(
    scene: dict[str, Any],
    parsed_result: dict[str, Any],
    *,
    experiment_name: str,
    stage: str,
    previous_final_sentence: str,
    story_caption: str,
    caption_usage_counts: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cleanup_records: list[dict[str, Any]] = []
    sentence = str(parsed_result.get("story_sentence") or "")
    reasons = _i_quality_gate_reasons(
        sentence,
        scene,
        story_caption,
        caption_usage_counts,
        previous_sentence=previous_final_sentence,
        check_ending=False,
    )
    if "english" in reasons:
        parsed_result, english_record = _run_i_english_term_translation(
            scene,
            parsed_result,
            experiment_name=experiment_name,
            stage=stage,
            previous_final_sentence=previous_final_sentence,
            story_caption=story_caption,
            caption_usage_counts=caption_usage_counts,
        )
        if english_record:
            cleanup_records.append(english_record)
        sentence = str(parsed_result.get("story_sentence") or "")
        reasons = _i_quality_gate_reasons(
            sentence,
            scene,
            story_caption,
            caption_usage_counts,
            previous_sentence=previous_final_sentence,
            check_ending=False,
        )
    if not reasons:
        return parsed_result, cleanup_records

    cleanup_scene = {
        **scene,
        "_i_previous_sentence": previous_final_sentence,
        "_i_current_sentence": sentence,
        "_i_quality_reasons": reasons,
        "_i_ending_scene": False,
    }
    cleanup_result = _run_exaone_scene_story(
        cleanup_scene,
        experiment_name=experiment_name,
        scene_prompt_builder=_build_i_captionless_cleanup_prompt,
        repair_prompt_builder=_build_i_captionless_cleanup_repair_prompt,
        max_new_tokens=I_CLEANUP_MAX_NEW_TOKENS,
        step_prefix=f"EXAONE-cleanup-{stage}",
        step_label=f"{stage} cleanup",
    )
    cleaned = cleanup_result["parsed_result"]
    remaining_reasons = _i_quality_gate_reasons(
        str(cleaned.get("story_sentence") or ""),
        scene,
        story_caption,
        caption_usage_counts,
        previous_sentence=previous_final_sentence,
        check_ending=False,
    )
    if remaining_reasons and "english" not in remaining_reasons:
        raise RuntimeError(
            "exaone_output_invalid: EXAONE cleanup response did not satisfy quality gates. "
            f"scene_index={int(scene['scene_index'])}; stage={stage}; "
            f"initial_reasons={reasons!r}; remaining_reasons={remaining_reasons!r}; "
            f"cleaned_response_head={cleanup_result['raw_response'][:800]!r}; "
            f"llama_runtime={cleanup_result.get('llama_runtime')!r}"
        )
    cleanup_record = {
        "stage": stage,
        "scene_index": int(scene["scene_index"]),
        "reasons": reasons,
        "remaining_reasons": remaining_reasons,
        "prompt": cleanup_result["prompt"],
        "raw_response": cleanup_result["raw_response"],
        "parsed_result": cleaned,
        "json_repair_used": cleanup_result["json_repair_used"],
    }
    cleanup_records.append(cleanup_record)
    if "english" in remaining_reasons:
        cleaned, english_record = _run_i_english_term_translation(
            scene,
            cleaned,
            experiment_name=experiment_name,
            stage=f"{stage}_post_cleanup",
            previous_final_sentence=previous_final_sentence,
            story_caption=story_caption,
            caption_usage_counts=caption_usage_counts,
        )
        if english_record:
            cleanup_records.append(english_record)
    return cleaned, cleanup_records


def _maybe_run_i_ending_cleanup(
    scene: dict[str, Any],
    parsed_result: dict[str, Any],
    *,
    experiment_name: str,
    previous_final_sentence: str,
    story_caption: str,
    caption_usage_counts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sentence = str(parsed_result.get("story_sentence") or "")
    reasons = _i_quality_gate_reasons(
        sentence,
        scene,
        story_caption,
        caption_usage_counts,
        previous_sentence=previous_final_sentence,
        check_ending=True,
    )
    ending_reasons = [reason for reason in reasons if reason == "ending"]
    if not ending_reasons:
        return parsed_result, None

    cleanup_scene = {
        **scene,
        "_i_previous_sentence": previous_final_sentence,
        "_i_current_sentence": sentence,
        "_i_quality_reasons": ending_reasons,
        "_i_ending_scene": True,
    }
    cleanup_result = _run_exaone_scene_story(
        cleanup_scene,
        experiment_name=experiment_name,
        scene_prompt_builder=_build_i_ending_cleanup_prompt,
        repair_prompt_builder=_build_i_ending_cleanup_repair_prompt,
        max_new_tokens=I_CLEANUP_MAX_NEW_TOKENS,
        step_prefix="EXAONE-ending-cleanup",
        step_label="ending cleanup",
    )
    cleaned = cleanup_result["parsed_result"]
    remaining_reasons = _ending_quality_reasons_for_gate(str(cleaned.get("story_sentence") or ""))
    if remaining_reasons:
        raise RuntimeError(
            "exaone_output_invalid: EXAONE ending cleanup response did not satisfy quality gates. "
            f"scene_index={int(scene['scene_index'])}; remaining_reasons={remaining_reasons!r}; "
            f"cleaned_response_head={cleanup_result['raw_response'][:800]!r}; "
            f"llama_runtime={cleanup_result.get('llama_runtime')!r}"
        )
    cleanup_record = {
        "stage": "ending",
        "scene_index": int(scene["scene_index"]),
        "reasons": ending_reasons,
        "remaining_reasons": remaining_reasons,
        "prompt": cleanup_result["prompt"],
        "raw_response": cleanup_result["raw_response"],
        "parsed_result": cleaned,
        "json_repair_used": cleanup_result["json_repair_used"],
    }
    return cleaned, cleanup_record


def _i_refinement_builder_for_scene(scene_index: int) -> Callable[[dict[str, Any]], str]:
    if scene_index == 1:
        return _build_i_opening_refinement_prompt
    if scene_index == 10:
        return _build_i_ending_refinement_prompt
    return _build_i_middle_refinement_prompt


def _run_exaone_i_experiment(scenes: list[dict[str, Any]], story_caption: str) -> dict[str, Any]:
    _ensure_exaone_gguf_available()
    experiment_name = "Experiment_I"
    story_caption = story_caption.strip()
    if not story_caption:
        raise ValueError("Experiment I requires a non-empty story caption.")

    ordered_scenes = [
        {**scene, "_story_caption": story_caption}
        for scene in sorted(scenes, key=lambda item: int(item["scene_index"]))
    ]
    cleanup_results: list[dict[str, Any]] = []
    caption_usage_counts: dict[str, int] = {}

    initial_results: list[dict[str, Any]] = []
    for scene in ordered_scenes:
        initial_result = _run_exaone_scene_story(
            scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_h_scene_prompt,
            repair_prompt_builder=_build_h_scene_repair_prompt,
            step_prefix="EXAONE-initial",
            step_label="initial scene",
        )
        initial_results.append(initial_result)
    initial_by_index = {int(item["scene_index"]): item["parsed_result"] for item in initial_results}

    refined_results: list[dict[str, Any]] = []
    final_scene_results: list[dict[str, Any]] = []
    previous_final_sentence = ""

    for scene in ordered_scenes:
        scene_index = int(scene["scene_index"])
        initial_sentence = initial_by_index[scene_index]["story_sentence"]
        refinement_scene = {
            **scene,
            "_g_previous_sentence": previous_final_sentence,
            "_g_initial_sentence": initial_sentence,
        }

        refined_result = _run_exaone_scene_story(
            refinement_scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_i_context_style_refinement_prompt,
            repair_prompt_builder=_build_i_context_style_repair_prompt,
            max_new_tokens=I_REFINEMENT_MAX_NEW_TOKENS,
            step_prefix="EXAONE-refine",
            step_label="context-style refinement scene",
        )
        refined_results.append(refined_result)
        cleaned_refined, cleanup_record = _maybe_run_i_cleanup(
            scene,
            refined_result["parsed_result"],
            experiment_name=experiment_name,
            stage="refined",
            previous_final_sentence=previous_final_sentence,
            story_caption=story_caption,
            caption_usage_counts=caption_usage_counts,
            check_ending=False,
        )
        if cleanup_record:
            cleanup_results.append(cleanup_record)
        final_scene_results.append(cleaned_refined)
        previous_final_sentence = cleaned_refined["story_sentence"]
        _record_caption_usage(previous_final_sentence, story_caption, caption_usage_counts)

    ending_result: dict[str, Any] | None = None
    if final_scene_results:
        ending_scene = ordered_scenes[-1]
        ending_previous_sentence = final_scene_results[-2]["story_sentence"] if len(final_scene_results) > 1 else ""
        ending_current_sentence = final_scene_results[-1]["story_sentence"]
        ending_input_scene = {
            **ending_scene,
            "_g_previous_sentence": ending_previous_sentence,
            "_i_current_sentence": ending_current_sentence,
            "_i_ending_scene": True,
        }
        ending_result = _run_exaone_scene_story(
            ending_input_scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_i_final_ending_prompt,
            repair_prompt_builder=_build_i_final_ending_repair_prompt,
            max_new_tokens=I_ENDING_MAX_NEW_TOKENS,
            step_prefix="EXAONE-ending",
            step_label="final ending scene",
        )
        cleaned_ending, cleanup_record = _maybe_run_i_cleanup(
            ending_scene,
            ending_result["parsed_result"],
            experiment_name=experiment_name,
            stage="ending",
            previous_final_sentence=ending_previous_sentence,
            story_caption=story_caption,
            caption_usage_counts=_caption_usage_without_sentence(
                caption_usage_counts,
                ending_current_sentence,
                story_caption,
            ),
            check_ending=True,
        )
        if cleanup_record:
            cleanup_results.append(cleanup_record)
        final_scene_results[-1] = cleaned_ending

    scene_sentences = [item["story_sentence"] for item in final_scene_results]
    body = "\n\n".join(scene_sentences)
    title_result = _generate_e_title(
        body,
        experiment_name=experiment_name,
        title_prompt_builder=_build_g_title_prompt,
    )
    json_repair_used = any(item["json_repair_used"] for item in initial_results + refined_results)
    if ending_result is not None:
        json_repair_used = json_repair_used or bool(ending_result["json_repair_used"])
    json_repair_used = json_repair_used or any(item["json_repair_used"] for item in cleanup_results)
    cleanup_summaries = [
        {
            "stage": item["stage"],
            "scene_index": item["scene_index"],
            "reasons": item["reasons"],
            "remaining_reasons": item["remaining_reasons"],
        }
        for item in cleanup_results
    ]
    return {
        "prompt_strategy": "combined_context_style_refinement_500_quality_gate",
        "story_caption": story_caption,
        "exaone_prompt": {
            "story_caption": story_caption,
            "initial_scene_prompts": [item["prompt"] for item in initial_results],
            "refinement_prompts": [item["prompt"] for item in refined_results],
            "ending_refinement_prompt": ending_result["prompt"] if ending_result is not None else "",
            "cleanup_prompts": [item["prompt"] for item in cleanup_results],
            "title_prompt": title_result["prompt"],
        },
        "exaone_raw_response": "\n\n".join(
            f"[initial scene {item['scene_index']}]\n{item['raw_response']}" for item in initial_results
        )
        + "\n\n"
        + "\n\n".join(
            f"[refine scene {item['scene_index']}]\n{item['raw_response']}" for item in refined_results
        )
        + (f"\n\n[ending scene 10]\n{ending_result['raw_response']}" if ending_result is not None else "")
        + "\n\n"
        + "\n\n".join(
            f"[cleanup {item['stage']} scene {item['scene_index']}]\n{item['raw_response']}" for item in cleanup_results
        )
        + f"\n\n[title]\n{title_result['raw_response']}",
        "llama_runtime": title_result.get("llama_runtime") or get_last_llama_runtime(),
        "parsed_result": {
            "story_caption": story_caption,
            "initial_scene_results": [item["parsed_result"] for item in initial_results],
            "refined_scene_results": [item["parsed_result"] for item in refined_results],
            "ending_refinement_result": ending_result["parsed_result"] if ending_result is not None else {},
            "cleanup_results": [
                {
                    "stage": item["stage"],
                    "scene_index": item["scene_index"],
                    "reasons": item["reasons"],
                    "remaining_reasons": item["remaining_reasons"],
                    "parsed_result": item["parsed_result"],
                }
                for item in cleanup_results
            ],
            "final_scene_results": final_scene_results,
            "title_result": title_result["parsed_result"],
        },
        "json_repair_used": json_repair_used,
        "story": {
            "title": title_result["title"],
            "body": body,
            "scene_sentences": scene_sentences,
            "grounding_notes": [],
        },
        "structure": {
            "mode": "combined_context_style_refinement_500_quality_gate",
            "scene_count": len(ordered_scenes),
            "quality_gates": list(I_QUALITY_GATES),
            "cleanup_calls": len(cleanup_results),
            "cleanup_scenes": cleanup_summaries,
            "story_caption_used": True,
            "story_caption_stage": "refinement_only",
            "exaone_initial_scene_calls": len(initial_results),
            "exaone_context_style_refinement_calls": len(refined_results),
            "exaone_ending_refinement_calls": 1 if ending_result is not None else 0,
            "exaone_title_calls": 1,
            "exaone_total_calls": (
                len(initial_results)
                + len(refined_results)
                + (1 if ending_result is not None else 0)
                + len(cleanup_results)
                + 1
            ),
        },
        "plan": {
            "method": (
                "Qwen scene JSON -> EXAONE initial per-scene paragraphs without story_caption -> "
                "EXAONE combined context/style refinement with weak story_caption -> "
                "EXAONE final ending refinement for scene 10 -> quality-gated cleanup -> "
                "code joins body -> EXAONE title"
            ),
            "story_caption": story_caption,
            "story_caption_stage": "refinement_only",
            "scene_order": [scene["image_id"] for scene in ordered_scenes],
            "scene_max_new_tokens": 350,
            "refinement_max_new_tokens": I_REFINEMENT_MAX_NEW_TOKENS,
            "ending_max_new_tokens": I_ENDING_MAX_NEW_TOKENS,
            "cleanup_max_new_tokens": I_CLEANUP_MAX_NEW_TOKENS,
            "title_max_new_tokens": 120,
        },
        "experiment_method": experiment_name,
    }


def _collage_direction_hint(collage_analysis: dict[str, Any] | None) -> str:
    if not collage_analysis:
        return ""
    parts: list[str] = []
    for key in ("overall_story_arc", "ending_read"):
        value = str(collage_analysis.get(key, "")).strip()
        if value:
            parts.append(value)
    for key in ("visual_continuity", "turning_points"):
        values = collage_analysis.get(key)
        if isinstance(values, list):
            parts.extend(str(item).strip() for item in values[:3] if str(item).strip())
    return " / ".join(parts[:8])


def _run_exaone_i_quality_gated_experiment(
    scenes: list[dict[str, Any]],
    story_caption: str,
    experiment_name: str = "Experiment_I",
    collage_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_exaone_gguf_available()
    story_caption = story_caption.strip()
    if not story_caption:
        raise ValueError(f"{experiment_name} requires a non-empty story caption.")

    weak_story_direction = _weak_story_direction(story_caption)
    collage_direction = _collage_direction_hint(collage_analysis)
    if collage_direction:
        weak_story_direction = (
            f"{weak_story_direction}\n"
            "Very weak collage sequence hint for Experiment J. Use only to avoid confusing scene order, "
            "never as required content. Keep individual scene JSON as primary evidence:\n"
            f"{collage_direction}"
        )
    ordered_scenes = [
        {
            **scene,
            "_story_caption": story_caption,
            "_weak_story_direction": weak_story_direction,
            "_scene_position_role": _i_scene_position_role(int(scene["scene_index"])),
        }
        for scene in sorted(scenes, key=lambda item: int(item["scene_index"]))
    ]
    initial_results = [
        _run_exaone_scene_story(
            scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_i_caption_initial_scene_prompt,
            repair_prompt_builder=_build_i_caption_initial_scene_repair_prompt,
            step_prefix="EXAONE-initial",
            step_label="initial scene",
        )
        for scene in ordered_scenes
    ]

    initial_by_index = {int(item["scene_index"]): item for item in initial_results}
    cleanup_results: list[dict[str, Any]] = []
    caption_usage_counts: dict[str, int] = {}
    first_scene = ordered_scenes[0]
    first_final, first_cleanup_records = _maybe_run_i_quality_cleanup(
        first_scene,
        initial_results[0]["parsed_result"],
        experiment_name=experiment_name,
        stage="initial",
        previous_final_sentence="",
        story_caption=story_caption,
        caption_usage_counts=caption_usage_counts,
    )
    cleanup_results.extend(first_cleanup_records)
    final_scene_results = [first_final]
    _record_caption_usage(first_final["story_sentence"], story_caption, caption_usage_counts)
    refined_results: list[dict[str, Any]] = []
    previous_final_sentence = first_final["story_sentence"]

    for scene in ordered_scenes[1:]:
        scene_index = int(scene["scene_index"])
        initial_sentence = initial_by_index[scene_index]["parsed_result"]["story_sentence"]
        refinement_scene = {
            **scene,
            "_g_previous_sentence": previous_final_sentence,
            "_g_initial_sentence": initial_sentence,
        }
        refined_result = _run_exaone_scene_story(
            refinement_scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_i_sequential_refinement_prompt,
            repair_prompt_builder=_build_i_sequential_refinement_repair_prompt,
            max_new_tokens=I_REFINEMENT_MAX_NEW_TOKENS,
            step_prefix="EXAONE-refine",
            step_label="refinement scene",
        )
        refined_results.append(refined_result)
        cleaned_refined, cleanup_records = _maybe_run_i_quality_cleanup(
            scene,
            refined_result["parsed_result"],
            experiment_name=experiment_name,
            stage="refinement",
            previous_final_sentence=previous_final_sentence,
            story_caption=story_caption,
            caption_usage_counts=caption_usage_counts,
        )
        cleanup_results.extend(cleanup_records)
        final_scene_results.append(cleaned_refined)
        previous_final_sentence = cleaned_refined["story_sentence"]
        if scene_index != 10:
            _record_caption_usage(previous_final_sentence, story_caption, caption_usage_counts)

    ending_cleanup_result: dict[str, Any] | None = None
    if len(final_scene_results) >= 2:
        ending_scene = ordered_scenes[-1]
        ending_previous_sentence = final_scene_results[-2]["story_sentence"]
        ending_cleaned, ending_cleanup_result = _maybe_run_i_ending_cleanup(
            ending_scene,
            final_scene_results[-1],
            experiment_name=experiment_name,
            previous_final_sentence=ending_previous_sentence,
            story_caption=story_caption,
            caption_usage_counts=caption_usage_counts,
        )
        if ending_cleanup_result:
            final_scene_results[-1] = ending_cleaned
            ending_translated, ending_translation_record = _run_i_english_term_translation(
                ending_scene,
                ending_cleaned,
                experiment_name=experiment_name,
                stage="ending_post_cleanup",
                previous_final_sentence=ending_previous_sentence,
                story_caption=story_caption,
                caption_usage_counts=caption_usage_counts,
            )
            if ending_translation_record:
                cleanup_results.append(ending_translation_record)
                final_scene_results[-1] = ending_translated
    if final_scene_results:
        _record_caption_usage(final_scene_results[-1]["story_sentence"], story_caption, caption_usage_counts)

    scene_sentences = [item["story_sentence"] for item in final_scene_results]
    body = "\n\n".join(scene_sentences)
    title_result = _generate_e_title(
        body,
        experiment_name=experiment_name,
        title_prompt_builder=_build_g_title_prompt,
    )
    json_repair_used = any(
        item["json_repair_used"]
        for item in initial_results + refined_results
    )
    json_repair_used = json_repair_used or any(item["json_repair_used"] for item in cleanup_results)
    if ending_cleanup_result:
        json_repair_used = json_repair_used or bool(ending_cleanup_result["json_repair_used"])
    json_repair_used = json_repair_used or bool(title_result.get("json_repair_used", False))
    return {
        "prompt_strategy": "generic_position_quality_gated_sequential_refinement",
        "story_caption": story_caption,
        "weak_story_direction": weak_story_direction,
        "collage_analysis": collage_analysis or {},
        "exaone_prompt": {
            "weak_story_direction": weak_story_direction,
            "collage_analysis": collage_analysis or {},
            "initial_scene_prompts": [item["prompt"] for item in initial_results],
            "refinement_prompts": [item["prompt"] for item in refined_results],
            "cleanup_prompts": [item["prompt"] for item in cleanup_results],
            "ending_cleanup_prompt": ending_cleanup_result["prompt"] if ending_cleanup_result else "",
            "title_prompt": title_result["prompt"],
        },
        "exaone_raw_response": "\n\n".join(
            f"[initial scene {item['scene_index']}]\n{item['raw_response']}" for item in initial_results
        )
        + "\n\n"
        + "\n\n".join(
            f"[refine scene {item['scene_index']}]\n{item['raw_response']}" for item in refined_results
        )
        + "\n\n"
        + "\n\n".join(
            f"[cleanup {item['stage']} scene {item['scene_index']}]\n{item['raw_response']}"
            for item in cleanup_results
        )
        + (f"\n\n[ending cleanup]\n{ending_cleanup_result['raw_response']}" if ending_cleanup_result else "")
        + f"\n\n[title]\n{title_result['raw_response']}",
        "llama_runtime": title_result.get("llama_runtime") or get_last_llama_runtime(),
        "parsed_result": {
            "story_caption": story_caption,
            "weak_story_direction": weak_story_direction,
            "collage_analysis": collage_analysis or {},
            "initial_scene_results": [item["parsed_result"] for item in initial_results],
            "refined_scene_results": [item["parsed_result"] for item in refined_results],
            "cleanup_results": [
                {
                    "stage": item["stage"],
                    "scene_index": item["scene_index"],
                    "reasons": item["reasons"],
                    "remaining_reasons": item["remaining_reasons"],
                    "english_terms": item.get("english_terms", []),
                    "translations": item.get("translations", {}),
                    "parsed_result": item["parsed_result"],
                }
                for item in cleanup_results
            ],
            "ending_cleanup_result": ending_cleanup_result["parsed_result"] if ending_cleanup_result else {},
            "final_scene_results": final_scene_results,
            "title_result": title_result["parsed_result"],
        },
        "json_repair_used": json_repair_used,
        "story": {
            "title": title_result["title"],
            "body": body,
            "scene_sentences": scene_sentences,
            "grounding_notes": [],
        },
        "structure": {
            "mode": "generic_position_quality_gated_sequential_refinement",
            "scene_count": len(ordered_scenes),
            "story_caption_used": True,
            "story_caption_stage": "initial_weak_direction_only",
            "story_caption_policy": "weak_direction_not_verbatim_caption",
            "scene_position_roles_used": True,
            "collage_analysis_used": bool(collage_analysis),
            "collage_analysis_stage": "weak_direction_hint" if collage_analysis else "",
            "quality_gates": list(I_QUALITY_GATES),
            "exaone_initial_scene_calls": len(initial_results),
            "exaone_refinement_calls": len(refined_results),
            "cleanup_calls": len(cleanup_results),
            "english_translation_calls": sum(
                1 for item in cleanup_results if "english_translation" in str(item.get("stage", ""))
            ),
            "ending_cleanup_calls": 1 if ending_cleanup_result else 0,
            "exaone_title_calls": 1,
            "exaone_total_calls": (
                len(initial_results)
                + len(refined_results)
                + len(cleanup_results)
                + (1 if ending_cleanup_result else 0)
                + 1
            ),
        },
        "plan": {
            "method": (
                "Qwen scene JSON -> EXAONE initial per-scene paragraphs with generic scene_position_role "
                "and weak_story_direction -> EXAONE sequential refinement without story_caption -> "
                "Python English detection with EXAONE term translation -> quality-gated cleanup -> optional ending cleanup -> "
                "code joins body -> EXAONE title"
            ),
            "story_caption": story_caption,
            "weak_story_direction": weak_story_direction,
            "collage_analysis": collage_analysis or {},
            "collage_direction": collage_direction,
            "collage_analysis_stage": "weak_direction_hint" if collage_analysis else "",
            "story_caption_stage": "initial_weak_direction_only",
            "story_caption_policy": "weak_direction_not_verbatim_caption",
            "scene_position_roles": {
                str(scene["scene_index"]): scene["_scene_position_role"]
                for scene in ordered_scenes
            },
            "scene_order": [scene["image_id"] for scene in ordered_scenes],
            "scene_max_new_tokens": 350,
            "refinement_max_new_tokens": I_REFINEMENT_MAX_NEW_TOKENS,
            "cleanup_max_new_tokens": I_CLEANUP_MAX_NEW_TOKENS,
            "title_max_new_tokens": 120,
        },
        "experiment_method": experiment_name,
    }


def _build_json_repair_prompt(
    raw_response: str,
    scene_count: int,
    scenes: list[dict[str, Any]] | None = None,
) -> str:
    repair_source = _last_jsonish_fragment(raw_response)
    compact_scenes = [_compact_scene(scene) for scene in scenes] if scenes else []
    scene_context = (
        "SCENES_TO_USE_FOR_REWRITING:\n"
        f"{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}\n\n"
        if compact_scenes
        else ""
    )
    return (
        "You are a strict JSON repair tool. Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "All object keys must use double quotes. Remove trailing commas. Escape line breaks inside strings.\n"
        f"The story.scene_sentences array must contain exactly {scene_count} strings.\n"
        "If any story.scene_sentences item is a placeholder, schema example, field name, or explanation, rewrite it into real Korean fairy-tale prose grounded in the matching scene.\n"
        "Do not keep phrases like '1번 그림에 해당하는 문단', '동화 문장', 'one non-empty Korean sentence', 'story_sentence', 'objects', or '그림 근거를 어떻게 반영'.\n"
        "Each story.scene_sentences item must be a natural Korean story paragraph for its scene, not a visual-analysis summary.\n"
        "Required shape:\n"
        "{\n"
        '  "structure": {},\n'
        '  "plan": {},\n'
        '  "story": {\n'
        '    "title": "non-empty Korean title",\n'
        '    "body": "non-empty Korean story body",\n'
        '    "scene_sentences": ["<write real Korean prose for scene 1>", "<write real Korean prose for scene 2>"],\n'
        '    "grounding_notes": []\n'
        "  }\n"
        "}\n\n"
        f"{scene_context}"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{repair_source}\n"
    )


def _base_json_instruction() -> str:
    return (
        "Return exactly one JSON object only. Do not add markdown, explanations, or code fences.\n"
        "story must contain title, body, scene_sentences, and grounding_notes.\n"
        "scene_sentences must contain exactly one real Korean fairy-tale paragraph per input scene.\n"
        "Do not output placeholder/schema/example text such as '1번 그림에 해당하는 문단', '동화 문장', 'one non-empty Korean sentence', 'story_sentence', or 'objects'.\n"
        "Do not put grounding explanations inside scene_sentences; use grounding_notes only for brief evidence notes.\n"
        "반드시 아래 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        "JSON 필드:\n"
        "{\n"
        '  "structure": {...},\n'
        '  "plan": {...},\n'
        '  "story": {\n'
        '    "title": "동화 제목",\n'
        '    "body": "장면 순서대로 이어지는 전체 동화 본문",\n'
        '    "scene_sentences": ["<write real Korean prose for scene 1>", "<write real Korean prose for scene 2>"],\n'
        '    "grounding_notes": ["그림 근거를 어떻게 반영했는지"]\n'
        "  }\n"
        "}\n"
        "scene_sentences 배열 길이는 입력 장면 수와 정확히 같아야 합니다.\n"
    )


def _prompt_c(scenes: list[dict[str, Any]]) -> str:
    compact_scenes = [_compact_scene(scene) for scene in scenes]
    return (
        "실험 C: Qwen이 읽은 아이 손그림 장면 목록을 보고 간단히 구조를 잡은 뒤 "
        "하나의 한국어 동화를 작성하세요.\n"
        "전략: 전체 장면을 한 번에 보고, 단순한 처음-중간-끝 구조로 정리합니다.\n"
        "하드코딩된 줄거리나 외부 지식 없이 scenes 안의 시각 단서만 사용하세요.\n\n"
        f"{_base_json_instruction()}\n"
        f"scenes:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}\n"
    )


def _prompt_d(scenes: list[dict[str, Any]]) -> str:
    compact_scenes = [_compact_scene(scene) for scene in scenes]
    return (
        "실험 D: 같은 Qwen 장면 목록을 사용하되, 한 프롬프트 안에서 4단계로 사고하세요.\n"
        "1) 장면별 시각 단서 구조화\n"
        "2) 전체 이야기 계획 수립\n"
        "3) 장면 순서대로 초안 작성\n"
        "4) 초안을 자체 점검해 시각 근거와 장면 연결을 보정\n"
        "최종 JSON에는 보정된 이야기만 story에 넣고, structure/plan에는 4단계 요약을 담으세요.\n"
        "어떤 이전 실험 결과도 참조하지 마세요.\n\n"
        f"{_base_json_instruction()}\n"
        f"scenes:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}\n"
    )


def _prompt_e(scenes: list[dict[str, Any]]) -> str:
    compact_scenes = [_compact_scene(scene) for scene in scenes]
    return (
        "실험 E: 먼저 내부적으로 다음 단계를 수행하세요. 단, 이 단계별 생각은 출력하지 마세요.\n"
        "1) 각 scene_index별 핵심 시각 단서 1개를 고르세요.\n"
        "2) 10개 장면의 이야기 흐름을 정하세요.\n"
        "3) 각 장면마다 scene_sentences를 정확히 1개씩 작성하세요.\n"
        "4) scene_sentences가 정확히 10개인지 확인하세요.\n"
        "5) 제목, 본문, 장면 문장에 \"해당하는 문단\" 같은 placeholder가 없는지 확인하세요.\n"
        "최종 출력은 JSON 객체 하나만 출력하세요.\n"
        "현재 그림에 없는 내용을 과하게 꾸며 넣지 말고 Qwen scene 단서를 우선하세요.\n"
        "어떤 이전 실험 결과도 참조하지 마세요.\n\n"
        f"{_base_json_instruction()}\n"
        f"scenes:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}\n"
    )


def _prompt_f(scenes: list[dict[str, Any]]) -> str:
    compact_scenes = [_compact_scene(scene) for scene in scenes]
    windows = _scene_windows(scenes)
    return (
        "실험 F: 먼저 전체 장면 목록을 보고 전역 흐름을 이해한 뒤, 각 target_scene_index의 문단을 작성하세요.\n"
        "각 문단은 current_scene의 시각 근거를 중심으로 써야 합니다.\n"
        "previous_scene은 앞 문단에서 자연스럽게 이어지도록 참고하세요.\n"
        "next_scene은 다음 문단으로 넘어갈 여지를 만드는 데만 참고하세요.\n"
        "앞뒤 장면 때문에 current_scene에 보이지 않는 사건이나 사물을 과하게 넣지 마세요.\n"
        "어떤 이전 실험 결과도 참조하지 마세요.\n\n"
        f"{_base_json_instruction()}\n"
        f"all_scenes:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}\n\n"
        f"scene_generation_windows:\n{json.dumps(windows, ensure_ascii=False, indent=2)}\n"
    )


def build_experiment_c(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment C: independent simple structure and story generation."""
    return _run_exaone_experiment(
        "Experiment_C",
        "simple_global_structure_then_story",
        _prompt_c(scenes),
        scenes,
    )


def build_experiment_d(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment D: independent four-step prompting in one EXAONE call."""
    return _run_exaone_experiment(
        "Experiment_D",
        "structure_plan_draft_self_check",
        _prompt_d(scenes),
        scenes,
    )


def build_experiment_e(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment E: generate each scene paragraph independently, then title the joined story."""
    return _run_exaone_per_scene_experiment(scenes)


def build_experiment_f(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment F: E pipeline with persona prompts for Qwen and EXAONE."""
    return _run_exaone_per_scene_experiment(
        scenes,
        experiment_name="Experiment_F",
        prompt_strategy="persona_per_scene_exaone_then_title",
        scene_prompt_builder=_build_f_scene_prompt,
        repair_prompt_builder=_build_f_scene_repair_prompt,
        title_prompt_builder=_build_f_title_prompt,
    )


def build_experiment_g(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment G: CoT + persona prompts with sequential scene refinement."""
    return _run_exaone_g_experiment(scenes)


def build_experiment_h(scenes: list[dict[str, Any]], story_caption: str) -> dict[str, Any]:
    """Experiment H: G pipeline with a story-folder caption guiding EXAONE."""
    return _run_exaone_h_experiment(scenes, story_caption)


def build_experiment_i(
    scenes: list[dict[str, Any]],
    story_caption: str,
    collage_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Experiment I: generic scene-position roles with quality-gated refinement."""
    return _run_exaone_i_quality_gated_experiment(
        scenes,
        story_caption,
        collage_analysis=collage_analysis,
    )


def build_experiment_j(
    scenes: list[dict[str, Any]],
    story_caption: str,
    collage_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Experiment J: Experiment I plus input-folder collage continuity analysis."""
    return _run_exaone_i_quality_gated_experiment(
        scenes,
        story_caption,
        experiment_name="Experiment_J",
        collage_analysis=collage_analysis,
    )


def _html_escape(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _file_url(path: Path) -> str:
    return "file:///" + str(path).replace("\\", "/")


def write_outputs(experiment_name: str, output_dir: Path, scenes: list[dict[str, Any]], result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": experiment_name,
        "vision_model": VISION_MODEL_ID,
        "llm_model": LLM_MODEL_NOTE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_order": [scene["image_id"] for scene in scenes],
        "scenes": scenes,
        **result,
    }
    (output_dir / f"{experiment_name.lower()}_result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    story = result["story"]
    (output_dir / f"{experiment_name.lower()}_story.txt").write_text(
        f"[제목]\n{story['title']}\n\n[동화]\n{story['body']}\n",
        encoding="utf-8",
    )
    scene_cards = []
    for scene, sentence in zip(scenes, story["scene_sentences"]):
        image_path = Path(str(scene.get("image_path") or INPUT_DIR / scene["image_id"]))
        scene_cards.append(
            f"""
            <article class="scene">
              <div class="image-frame"><img src="{_html_escape(_file_url(image_path))}" alt="{_html_escape(scene['image_id'])}"></div>
              <div class="text">
                <p class="no">{scene['scene_index']}번째 그림</p>
                <p class="label">EXAONE 장면 문장</p>
                <p class="sentence">{_html_escape(sentence)}</p>
                <p class="summary-label">Qwen 시각 요약</p>
                <p class="summary">{_html_escape(scene['scene_summary'])}</p>
              </div>
            </article>
            """
        )
    story_paragraphs = "\n".join(f"<p>{_html_escape(part)}</p>" for part in story["body"].split("\n\n"))
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html_escape(experiment_name)} - {_html_escape(story['title'])}</title>
<style>
body {{ margin:0; font-family:"Malgun Gothic",system-ui,sans-serif; background:#fff8e8; color:#2a211a; line-height:1.7; }}
header {{ padding:38px clamp(18px,5vw,70px); background:#fff1cf; border-bottom:1px solid #ddcfbd; }}
h1 {{ margin:0; font-size:clamp(30px,6vw,60px); letter-spacing:0; }}
main {{ max-width:1180px; margin:0 auto; padding:28px clamp(14px,3vw,36px) 60px; }}
.meta {{ color:#5f574f; }}
section {{ margin-top:26px; }}
.book {{ background:#fffdf7; border:1px solid #ddcfbd; border-radius:8px; padding:22px; }}
.book p {{ font-size:18px; margin:0 0 12px; word-break:keep-all; }}
.scene {{ display:grid; grid-template-columns:minmax(230px,42%) 1fr; gap:22px; align-items:center; margin:18px 0; padding:18px; background:#fffdf7; border:1px solid #ddcfbd; border-radius:8px; }}
.image-frame {{ aspect-ratio:4/3; border:1px solid #ddcfbd; border-radius:8px; background:white; overflow:hidden; }}
.image-frame img {{ width:100%; height:100%; object-fit:contain; display:block; }}
.no {{ margin:0 0 8px; color:#964b3f; font-weight:700; }}
.label {{ margin:0 0 6px; color:#2f6652; font-size:13px; font-weight:700; }}
.sentence {{ margin:0; font-size:clamp(18px,2.1vw,24px); word-break:keep-all; }}
.summary-label {{ margin:16px 0 4px; color:#6f6257; font-size:12px; font-weight:700; }}
.summary {{ margin:12px 0 0; color:#74695f; font-size:14px; }}
@media (max-width:760px) {{ .scene {{ grid-template-columns:1fr; }} .image-frame {{ aspect-ratio:1/1; }} }}
</style>
</head>
<body>
<header>
<p class="meta">{_html_escape(experiment_name)} · vision: {_html_escape(VISION_MODEL_ID)} · llm: {_html_escape(LLM_MODEL_NOTE)}</p>
<h1>{_html_escape(story['title'])}</h1>
</header>
<main>
<section class="book"><h2>[동화]</h2>{story_paragraphs}</section>
<section><h2>그림 옆 EXAONE 장면 문장</h2>{"".join(scene_cards)}</section>
</main>
</body>
</html>"""
    (output_dir / f"{experiment_name.lower()}_story.html").write_text(html, encoding="utf-8")


def _experiment_dirs(output_root: Path) -> dict[str, Path]:
    return {
        "c": output_root / "C",
        "d": output_root / "D",
        "e": output_root / "E",
        "f": output_root / "F",
        "g": output_root / "G",
        "h": output_root / "H",
        "i": output_root / "I",
        "j": output_root / "J",
    }


def _experiment_builders() -> dict[str, tuple[str, Any]]:
    return {
        "c": ("Experiment_C", build_experiment_c),
        "d": ("Experiment_D", build_experiment_d),
        "e": ("Experiment_E", build_experiment_e),
        "f": ("Experiment_F", build_experiment_f),
        "g": ("Experiment_G", build_experiment_g),
        "h": ("Experiment_H", build_experiment_h),
        "i": ("Experiment_I", build_experiment_i),
        "j": ("Experiment_J", build_experiment_j),
    }


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
                log_stage(f"Qwen CUDA move failed; reloading on CPU: {exc}", step="Qwen-device", model=VISION_MODEL_ID)
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
            max_pixels=QWEN_MAX_PIXELS,
        )
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


def run_experiment_with_scenes(
    experiment: str,
    scenes: list[dict[str, Any]],
    output_root: str | Path = OUTPUT_ROOT,
    story_caption: str | None = None,
    collage_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    key = experiment.lower()
    dirs = _experiment_dirs(output_root)
    builders = _experiment_builders()
    experiment_name, builder = builders[key]
    set_step_context(experiment=experiment_name, phase="generation")
    log_stage(f"building {experiment_name}", step=key.upper(), model="Qwen scenes + EXAONE GGUF")
    if key == "j":
        result = builder(scenes, story_caption or "", collage_analysis or {})
    elif key == "i":
        result = builder(scenes, story_caption or "", collage_analysis or {})
    elif key == "h":
        result = builder(scenes, story_caption or "")
    else:
        result = builder(scenes)
    write_outputs(experiment_name, dirs[key], scenes, result)
    log_stage(f"saved {key.upper()}: {dirs[key]}", step=key.upper(), model="output")
    return {"output_dir": str(dirs[key]), "result": result}


def run_selected_experiments(
    experiments: list[str] | tuple[str, ...] = ("c", "d", "e", "f", "g", "h", "i", "j"),
    input_dir: str | Path = INPUT_DIR,
    output_root: str | Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = Path(output_root)
    input_dir = Path(input_dir)
    selected = [experiment.lower() for experiment in experiments]
    if "all" in selected:
        selected = ["c", "d", "e", "f", "g", "h", "i", "j"]

    results: dict[str, Any] = {}
    for key in selected:
        story_caption = _read_story_caption(input_dir) if key in {"h", "i", "j"} else None
        collage_analysis = (
            prepare_qwen_collage_for_experiment(
                input_dir,
                output_root,
                story_caption=story_caption or "",
                experiment=key,
            )
            if key in {"i", "j"}
            else None
        )
        set_step_context(experiment=key.upper(), phase="vision")
        log_stage(f"start Experiment {key.upper()} Qwen scene generation", step="Qwen", event="start")
        scenes = prepare_qwen_scenes_for_experiment(
            key,
            input_dir=input_dir,
            output_root=output_root,
            story_caption=story_caption,
            collage_analysis=collage_analysis,
        )
        set_step_context(experiment=key.upper(), phase="vision")
        log_stage(f"Experiment {key.upper()} Qwen scene generation succeeded", step="Qwen", event="success")
        results[key] = run_experiment_with_scenes(
            key,
            scenes,
            output_root=output_root,
            story_caption=story_caption,
            collage_analysis=collage_analysis,
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent Qwen + EXAONE GGUF experiments C/D/E/F/G/H/I/J.")
    parser.add_argument(
        "experiments",
        nargs="*",
        choices=("c", "d", "e", "f", "g", "h", "i", "j", "all"),
        default=["all"],
        help="Experiments to run. Defaults to all.",
    )
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Ordered image input directory.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="Output root directory.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_selected_experiments(
        experiments=args.experiments,
        input_dir=args.input_dir,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
