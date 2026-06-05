"""Run independent C/D/E/F/G/H/I experiments with Qwen vision and EXAONE GGUF writing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
STORY_CAPTION_FILENAME = "caption.txt"
H_REFINEMENT_MAX_NEW_TOKENS = 450
I_REFINEMENT_MAX_NEW_TOKENS = 450
I_CLEANUP_MAX_NEW_TOKENS = 300
I_QUALITY_GATES = ("english", "meta_language", "caption_repetition", "ending")


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
    others: list[Path] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem.isdigit():
                numbered[int(path.stem)] = path
            else:
                others.append(path)
    return [numbered[key] for key in sorted(numbered)] + sorted(others)


def _read_story_caption(input_dir: Path) -> str:
    caption_path = input_dir / STORY_CAPTION_FILENAME
    if not caption_path.exists():
        raise FileNotFoundError(
            f"Experiment H/I requires {STORY_CAPTION_FILENAME} in the selected story folder: {caption_path}"
        )
    caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"Experiment H/I requires a non-empty story caption: {caption_path}")
    return caption


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


def _json_object_candidates(text: str) -> list[str]:
    """Return balanced JSON-object-looking substrings from model text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
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


def _extract_required_json(text: str) -> dict[str, Any]:
    candidates = _json_object_candidates(text)
    if not candidates:
        raise ValueError("EXAONE response did not contain a JSON object.")
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, dict):
            return value
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


def _run_qwen_scene(
    model: Any,
    processor: Any,
    image_path: Path,
    index: int,
    prompt_builder: Callable[[int], str] = _prompt,
    max_new_tokens: int = 220,
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
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _normalize_scene(index, image_path, _extract_json(raw), raw)


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


def _story_from_payload(payload: dict[str, Any], scene_count: int) -> dict[str, Any]:
    story = payload.get("story") if isinstance(payload.get("story"), dict) else payload
    title = str(story.get("title") or "").strip()
    if not title:
        raise ValueError("EXAONE story.title is required.")
    body = str(story.get("body") or "").strip()
    if not body:
        raise ValueError("EXAONE story.body is required.")
    scene_sentences = story.get("scene_sentences")
    if not isinstance(scene_sentences, list):
        scene_sentences = []
    scene_sentences = [str(sentence).strip() for sentence in scene_sentences if str(sentence).strip()]
    if not scene_sentences and body:
        scene_sentences = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if len(scene_sentences) != scene_count:
        raise ValueError(
            f"EXAONE returned {len(scene_sentences)} scene_sentences for {scene_count} scenes."
        )
    grounding_notes = story.get("grounding_notes")
    return {
        "title": title,
        "body": body,
        "scene_sentences": scene_sentences,
        "grounding_notes": grounding_notes if grounding_notes is not None else [],
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
        )
    llama_runtime = get_last_llama_runtime()
    json_repair_used = False
    try:
        payload = _extract_required_json(raw_response)
        story = _story_from_payload(payload, len(scenes))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        repair_prompt = _build_json_repair_prompt(raw_response, len(scenes))
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
            )
        json_repair_used = True
        try:
            payload = _extract_required_json(repair_response)
            story = _story_from_payload(payload, len(scenes))
        except (json.JSONDecodeError, ValueError, TypeError) as repair_exc:
            raise RuntimeError(
                "EXAONE did not return valid required JSON, and JSON repair also failed. "
                f"initial_error={exc}; repair_error={repair_exc}; "
                f"raw_response_head={raw_response[:800]!r}; repair_response_head={repair_response[:800]!r}"
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


def _looks_like_placeholder(value: str) -> bool:
    text = value.strip()
    if not text:
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


def _has_meta_language(value: str) -> bool:
    lowered = value.lower()
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


def _caption_repetition_reasons(
    value: str,
    scene: dict[str, Any],
    story_caption: str,
    caption_usage_counts: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    for keyword in _caption_keywords(story_caption):
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
        reasons.extend(_ending_quality_reasons(value))
    return list(dict.fromkeys(reasons))


def _record_caption_usage(value: str, story_caption: str, caption_usage_counts: dict[str, int]) -> None:
    for keyword in _caption_keywords(story_caption):
        if keyword in value:
            caption_usage_counts[keyword] = caption_usage_counts.get(keyword, 0) + 1


def _sentence_mark_count(value: str) -> int:
    return len(re.findall(r"[.!?。！？]", value))


def _scene_story_from_payload(
    payload: dict[str, Any],
    expected_index: int,
    *,
    enforce_sentence_count: bool = True,
) -> dict[str, Any]:
    scene_index = int(payload.get("scene_index") or expected_index)
    if scene_index != expected_index:
        raise ValueError(f"EXAONE scene_index mismatch: expected {expected_index}, got {scene_index}")
    story_sentence = str(payload.get("story_sentence") or "").strip()
    if _looks_like_placeholder(story_sentence):
        raise ValueError(f"EXAONE scene {expected_index} returned placeholder or empty story_sentence.")
    if enforce_sentence_count:
        sentence_count = _sentence_mark_count(story_sentence)
        if not 3 <= sentence_count <= 5:
            raise ValueError(
                f"EXAONE scene {expected_index} story_sentence must contain 3 to 5 sentences, got {sentence_count}."
            )
    return {"scene_index": scene_index, "story_sentence": story_sentence}


def _has_korean(value: str) -> bool:
    return bool(re.search(r"[가-힣]", value))


def _clean_fallback_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[A-Za-z_/]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("그림에는", "").replace("그림에는", "")
    text = text.replace("그림은", "").replace("그림에는", "")
    text = text.replace("보입니다", "").replace("나타냅니다", "")
    text = text.replace("~처럼 보임", "")
    return text.strip(" .。")


def _fallback_scene_subject(scene: dict[str, Any]) -> str:
    text = json.dumps(_compact_scene(scene), ensure_ascii=False)
    if "소녀" in text:
        return "소녀는"
    if "소년" in text:
        return "소년은"
    if "아이" in text or "어린" in text:
        return "아이는"
    if "고양이" in text:
        return "고양이는"
    if "강아지" in text:
        return "강아지는"
    if "동물" in text:
        return "동물 친구들은"
    return "아이는"


def _fallback_scene_place(scene: dict[str, Any]) -> str:
    setting = _clean_fallback_text(scene.get("setting", ""))
    summary = _clean_fallback_text(scene.get("scene_summary", ""))
    if setting and _has_korean(setting) and "중요" not in setting:
        return setting
    if "숲" in summary:
        return "숲속"
    if "나무" in summary:
        return "나무 아래"
    if "하늘" in summary:
        return "밝은 하늘 아래"
    return "작은 길 위"


def _fallback_scene_focus(scene: dict[str, Any]) -> str:
    text = json.dumps(_compact_scene(scene), ensure_ascii=False)
    for keyword, phrase in (
        ("별", "반짝이는 빛"),
        ("꽃", "작은 꽃"),
        ("나무", "푸른 나무"),
        ("열매", "동그란 열매"),
        ("동물", "동물 친구들"),
        ("하늘", "밝은 하늘"),
        ("구름", "하얀 구름"),
        ("물", "맑은 물"),
    ):
        if keyword in text:
            return phrase
    return "작은 발견"


def _fallback_scene_story_payload(scene: dict[str, Any]) -> dict[str, Any]:
    scene_index = int(scene["scene_index"])
    subject = _fallback_scene_subject(scene)
    place = _fallback_scene_place(scene)
    focus = _fallback_scene_focus(scene)

    if scene_index == 1:
        sentences = [
            f"{subject} {place}에서 조용히 걸음을 멈추었다.",
            f"눈앞의 {focus} 덕분에 작은 이야기가 살며시 시작되었다.",
            "따뜻한 마음이 바람처럼 아이 곁에 머물렀다.",
        ]
    elif scene_index == 10 or bool(scene.get("_i_ending_scene")):
        sentences = [
            f"{subject} {place}에서 편안히 숨을 고르며 미소 지었다.",
            f"눈앞의 {focus} 덕분에 지나온 길이 따뜻하게 느껴졌다.",
            "모두의 마음에는 오래도록 반짝이는 여운이 남았다.",
        ]
    else:
        sentences = [
            f"{subject} {place}에서 잠시 멈춰 섰다.",
            f"눈앞의 {focus} 덕분에 마음이 조금씩 밝아졌다.",
            "아이는 그 빛을 따라 다음 걸음을 천천히 옮겼다.",
        ]

    return {"scene_index": scene_index, "story_sentence": " ".join(sentences)}


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
    *,
    enforce_sentence_count: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _json_object_candidates(text)
    if not candidates:
        payload = _partial_scene_story_payload(text, expected_index)
        if payload is None:
            raise ValueError("EXAONE response did not contain a JSON object.")
        parsed = _scene_story_from_payload(
            payload,
            expected_index,
            enforce_sentence_count=enforce_sentence_count,
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
                enforce_sentence_count=enforce_sentence_count,
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
                enforce_sentence_count=enforce_sentence_count,
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
    fallback_payload_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
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
                enforce_sentence_count=False,
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
                enforce_sentence_count=False,
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
                    "fallback_used": False,
                    "fallback_error": "",
                    "llama_runtime": get_last_llama_runtime(),
                }
            if fallback_payload_builder is not None:
                fallback_payload = fallback_payload_builder(scene)
                fallback_parsed = _scene_story_from_payload(
                    fallback_payload,
                    scene_index,
                    enforce_sentence_count=True,
                )
                raw_response = (
                    f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
                    f"\n\n[fallback_scene_story]\n{json.dumps(fallback_payload, ensure_ascii=False, indent=2)}"
                )
                return {
                    "scene_index": scene_index,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_result": fallback_parsed,
                    "json_repair_used": json_repair_used,
                    "fallback_used": True,
                    "fallback_error": f"initial_error={exc}; repair_error={repair_exc}",
                    "llama_runtime": get_last_llama_runtime(),
                }
            raise RuntimeError(
                f"EXAONE scene {scene_index} did not return valid scene JSON, and JSON repair also failed. "
                f"initial_error={exc}; repair_error={repair_exc}; "
                f"raw_response_head={raw_response[:800]!r}; repair_response_head={repair_response[:800]!r}"
            ) from repair_exc
        raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
    return {
        "scene_index": scene_index,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_result": parsed,
        "json_repair_used": json_repair_used,
        "fallback_used": False,
        "fallback_error": "",
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
    default_title: str = "그림 속 작은 이야기",
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
    title = default_title
    parse_error = ""
    try:
        payload, title = _extract_title_json(raw_response)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        parse_error = str(exc)
        payload = {}
    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_result": payload,
        "title": title,
        "fallback_used": title == default_title,
        "parse_error": parse_error,
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
            "title_fallback_used": title_result["fallback_used"],
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
            "title_fallback_used": title_result["fallback_used"],
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
            "title_fallback_used": title_result["fallback_used"],
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
        scene_prompt_builder=_build_i_cleanup_prompt,
        repair_prompt_builder=_build_i_cleanup_repair_prompt,
        fallback_payload_builder=_fallback_scene_story_payload,
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
    cleanup_record = {
        "stage": stage,
        "scene_index": int(scene["scene_index"]),
        "reasons": reasons,
        "remaining_reasons": remaining_reasons,
        "prompt": cleanup_result["prompt"],
        "raw_response": cleanup_result["raw_response"],
        "parsed_result": cleaned,
        "json_repair_used": cleanup_result["json_repair_used"],
        "fallback_used": cleanup_result.get("fallback_used", False),
        "fallback_error": cleanup_result.get("fallback_error", ""),
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
    initial_scene_results_after_cleanup: dict[int, dict[str, Any]] = {}
    for scene in ordered_scenes:
        initial_result = _run_exaone_scene_story(
            scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_build_h_scene_prompt,
            repair_prompt_builder=_build_h_scene_repair_prompt,
            fallback_payload_builder=_fallback_scene_story_payload,
            step_prefix="EXAONE-initial",
            step_label="initial scene",
        )
        initial_results.append(initial_result)
        cleaned_initial, cleanup_record = _maybe_run_i_cleanup(
            scene,
            initial_result["parsed_result"],
            experiment_name=experiment_name,
            stage="initial",
            previous_final_sentence="",
            story_caption=story_caption,
            caption_usage_counts={},
            check_ending=False,
        )
        if cleanup_record:
            cleanup_results.append(cleanup_record)
        initial_scene_results_after_cleanup[int(scene["scene_index"])] = cleaned_initial

    refined_results: list[dict[str, Any]] = []
    final_scene_results: list[dict[str, Any]] = []
    previous_final_sentence = ""
    opening_refinement_calls = 0
    middle_refinement_calls = 0
    ending_refinement_calls = 0

    for scene in ordered_scenes:
        scene_index = int(scene["scene_index"])
        initial_sentence = initial_scene_results_after_cleanup[scene_index]["story_sentence"]
        refinement_scene = {
            **scene,
            "_g_previous_sentence": previous_final_sentence,
            "_g_initial_sentence": initial_sentence,
        }
        if scene_index == 1:
            opening_refinement_calls += 1
        elif scene_index == 10:
            ending_refinement_calls += 1
        else:
            middle_refinement_calls += 1

        refined_result = _run_exaone_scene_story(
            refinement_scene,
            experiment_name=experiment_name,
            scene_prompt_builder=_i_refinement_builder_for_scene(scene_index),
            repair_prompt_builder=_build_i_refinement_repair_prompt,
            fallback_payload_builder=_fallback_scene_story_payload,
            max_new_tokens=I_REFINEMENT_MAX_NEW_TOKENS,
            step_prefix="EXAONE-refine",
            step_label="refinement scene",
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
            check_ending=scene_index == 10,
        )
        if cleanup_record:
            cleanup_results.append(cleanup_record)
        final_scene_results.append(cleaned_refined)
        previous_final_sentence = cleaned_refined["story_sentence"]
        _record_caption_usage(previous_final_sentence, story_caption, caption_usage_counts)

    scene_sentences = [item["story_sentence"] for item in final_scene_results]
    body = "\n\n".join(scene_sentences)
    title_result = _generate_e_title(
        body,
        experiment_name=experiment_name,
        title_prompt_builder=_build_g_title_prompt,
    )
    json_repair_used = any(item["json_repair_used"] for item in initial_results + refined_results)
    json_repair_used = json_repair_used or any(item["json_repair_used"] for item in cleanup_results)
    cleanup_summaries = [
        {
            "stage": item["stage"],
            "scene_index": item["scene_index"],
            "reasons": item["reasons"],
            "remaining_reasons": item["remaining_reasons"],
            "fallback_used": item.get("fallback_used", False),
        }
        for item in cleanup_results
    ]
    fallback_summaries = [
        {
            "stage": "initial",
            "scene_index": item["scene_index"],
            "fallback_error": item.get("fallback_error", ""),
        }
        for item in initial_results
        if item.get("fallback_used")
    ] + [
        {
            "stage": "refined",
            "scene_index": item["scene_index"],
            "fallback_error": item.get("fallback_error", ""),
        }
        for item in refined_results
        if item.get("fallback_used")
    ] + [
        {
            "stage": f"cleanup_{item['stage']}",
            "scene_index": item["scene_index"],
            "fallback_error": item.get("fallback_error", ""),
        }
        for item in cleanup_results
        if item.get("fallback_used")
    ]
    return {
        "prompt_strategy": "comprehensive_quality_gate_sequential_refinement",
        "story_caption": story_caption,
        "exaone_prompt": {
            "story_caption": story_caption,
            "initial_scene_prompts": [item["prompt"] for item in initial_results],
            "refinement_prompts": [item["prompt"] for item in refined_results],
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
            "cleanup_results": [
                {
                    "stage": item["stage"],
                    "scene_index": item["scene_index"],
                    "reasons": item["reasons"],
                    "remaining_reasons": item["remaining_reasons"],
                    "parsed_result": item["parsed_result"],
                    "fallback_used": item.get("fallback_used", False),
                }
                for item in cleanup_results
            ],
            "final_scene_results": final_scene_results,
            "title_result": title_result["parsed_result"],
            "title_fallback_used": title_result["fallback_used"],
        },
        "json_repair_used": json_repair_used,
        "story": {
            "title": title_result["title"],
            "body": body,
            "scene_sentences": scene_sentences,
            "grounding_notes": [],
        },
        "structure": {
            "mode": "comprehensive_quality_gate_sequential_refinement",
            "scene_count": len(ordered_scenes),
            "quality_gates": list(I_QUALITY_GATES),
            "cleanup_calls": len(cleanup_results),
            "cleanup_scenes": cleanup_summaries,
            "fallback_calls": len(fallback_summaries),
            "fallback_scenes": fallback_summaries,
            "story_caption_used": True,
            "story_caption_stage": "refinement_only",
            "exaone_initial_scene_calls": len(initial_results),
            "exaone_opening_refinement_calls": opening_refinement_calls,
            "exaone_middle_refinement_calls": middle_refinement_calls,
            "exaone_ending_refinement_calls": ending_refinement_calls,
            "exaone_refinement_calls": len(refined_results),
            "exaone_title_calls": 1,
            "exaone_total_calls": len(initial_results) + len(refined_results) + len(cleanup_results) + 1,
        },
        "plan": {
            "method": (
                "Qwen scene JSON -> EXAONE initial per-scene paragraphs without story_caption -> "
                "quality-gated cleanup -> EXAONE opening/middle/ending refinement with weak story_caption -> "
                "quality-gated cleanup -> code joins body -> EXAONE title"
            ),
            "story_caption": story_caption,
            "story_caption_stage": "refinement_only",
            "scene_order": [scene["image_id"] for scene in ordered_scenes],
            "scene_max_new_tokens": 350,
            "refinement_max_new_tokens": I_REFINEMENT_MAX_NEW_TOKENS,
            "cleanup_max_new_tokens": I_CLEANUP_MAX_NEW_TOKENS,
            "title_max_new_tokens": 120,
        },
        "experiment_method": experiment_name,
    }


def _build_json_repair_prompt(raw_response: str, scene_count: int) -> str:
    return (
        "You are a strict JSON repair tool. Convert the model response below into one valid JSON object only.\n"
        "Do not add markdown. Do not explain. Do not include any text before or after JSON.\n"
        "All object keys must use double quotes. Remove trailing commas. Escape line breaks inside strings.\n"
        f"The story.scene_sentences array must contain exactly {scene_count} strings.\n"
        "Required shape:\n"
        "{\n"
        '  "structure": {},\n'
        '  "plan": {},\n'
        '  "story": {\n'
        '    "title": "",\n'
        '    "body": "",\n'
        '    "scene_sentences": [],\n'
        '    "grounding_notes": []\n'
        "  }\n"
        "}\n\n"
        "MODEL_RESPONSE_TO_REPAIR:\n"
        f"{raw_response}\n"
    )


def _base_json_instruction() -> str:
    return (
        "반드시 아래 JSON 객체 하나만 출력하세요. 마크다운, 설명, 코드블록은 쓰지 마세요.\n"
        "JSON 필드:\n"
        "{\n"
        '  "structure": {...},\n'
        '  "plan": {...},\n'
        '  "story": {\n'
        '    "title": "동화 제목",\n'
        '    "body": "장면 순서대로 이어지는 전체 동화 본문",\n'
        '    "scene_sentences": ["1번 그림에 해당하는 문단", "2번 그림에 해당하는 문단"],\n'
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


def build_experiment_i(scenes: list[dict[str, Any]], story_caption: str) -> dict[str, Any]:
    """Experiment I: H pipeline with comprehensive quality gates and ending cleanup."""
    return _run_exaone_i_experiment(scenes, story_caption)


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
                <p class="sentence">{_html_escape(sentence)}</p>
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
.sentence {{ margin:0; font-size:clamp(18px,2.1vw,24px); word-break:keep-all; }}
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
<section><h2>그림 옆 장면 문장</h2>{"".join(scene_cards)}</section>
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
    with timed_step("Qwen-load", "Qwen2.5-VL model load", model=VISION_MODEL_ID):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
            local_files_only=local_only,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
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


def prepare_qwen_scenes_for_experiment(
    experiment: str,
    input_dir: str | Path = INPUT_DIR,
    output_root: str | Path = OUTPUT_ROOT,
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
    elif key in {"g", "h", "i"}:
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


def run_experiment_with_scenes(
    experiment: str,
    scenes: list[dict[str, Any]],
    output_root: str | Path = OUTPUT_ROOT,
    story_caption: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    key = experiment.lower()
    dirs = _experiment_dirs(output_root)
    builders = _experiment_builders()
    experiment_name, builder = builders[key]
    set_step_context(experiment=experiment_name, phase="generation")
    log_stage(f"building {experiment_name}", step=key.upper(), model="Qwen scenes + EXAONE GGUF")
    if key in {"h", "i"}:
        result = builder(scenes, story_caption or "")
    else:
        result = builder(scenes)
    write_outputs(experiment_name, dirs[key], scenes, result)
    log_stage(f"saved {key.upper()}: {dirs[key]}", step=key.upper(), model="output")
    return {"output_dir": str(dirs[key]), "result": result}


def run_selected_experiments(
    experiments: list[str] | tuple[str, ...] = ("c", "d", "e", "f", "g", "h", "i"),
    input_dir: str | Path = INPUT_DIR,
    output_root: str | Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = Path(output_root)
    input_dir = Path(input_dir)
    selected = [experiment.lower() for experiment in experiments]
    if "all" in selected:
        selected = ["c", "d", "e", "f", "g", "h", "i"]

    results: dict[str, Any] = {}
    for key in selected:
        story_caption = _read_story_caption(input_dir) if key in {"h", "i"} else None
        set_step_context(experiment=key.upper(), phase="vision")
        log_stage(f"start Experiment {key.upper()} Qwen scene generation", step="Qwen", event="start")
        scenes = prepare_qwen_scenes_for_experiment(
            key,
            input_dir=input_dir,
            output_root=output_root,
        )
        set_step_context(experiment=key.upper(), phase="vision")
        log_stage(f"Experiment {key.upper()} Qwen scene generation succeeded", step="Qwen", event="success")
        results[key] = run_experiment_with_scenes(
            key,
            scenes,
            output_root=output_root,
            story_caption=story_caption,
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent Qwen + EXAONE GGUF experiments C/D/E/F/G/H/I.")
    parser.add_argument(
        "experiments",
        nargs="*",
        choices=("c", "d", "e", "f", "g", "h", "i", "all"),
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
