"""Story generation and translation for Experiment A."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

from utils import (
    DEFAULT_EXAONE_GGUF_PATH,
    LLAMA_CLI_FILENAME,
    LLAMA_CLI_PATH,
    PROJECT_ROOT,
    configured_llama_gpu_layers,
    ensure_exaone_gguf_model,
    get_device,
    get_exaone_components,
    get_gpt2_components,
    get_nllb_components,
    has_nvidia_gpu,
    log_stage,
    timed_step,
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "front",
    "in",
    "is",
    "of",
    "on",
    "s",
    "the",
    "to",
    "with",
}

LAST_LLAMA_RUNTIME: dict[str, Any] = {"mode": "unknown"}
LLAMA_CPP_WINDOWS_RELEASE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b9500/"
    "llama-b9500-bin-win-vulkan-x64.zip"
)

CONCEPT_KO = {
    "baby": "아기",
    "bird": "새",
    "boy": "남자아이",
    "car": "자동차",
    "cactus": "선인장",
    "cat": "고양이",
    "child": "아이",
    "children": "아이들",
    "cloud": "구름",
    "dog": "강아지",
    "drawing": "그림",
    "family": "가족",
    "flower": "꽃",
    "flying": "나는 모습",
    "girl": "여자아이",
    "grass": "풀밭",
    "happy": "행복한 마음",
    "home": "집",
    "house": "집",
    "little": "작은 아이",
    "mother": "엄마",
    "moon": "달",
    "outside": "바깥",
    "person": "사람",
    "playing": "놀이",
    "rainbow": "무지개",
    "sky": "하늘",
    "star": "별",
    "stars": "별",
    "standing": "서 있는 모습",
    "stork": "황새",
    "sun": "해",
    "sunlight": "햇살",
    "tree": "나무",
    "tiger": "호랑이",
    "white ball": "하얀 공",
}

PHRASE_KO = {
    "a girl": "여자아이",
    "girl": "여자아이",
    "little girl": "여자아이",
    "a tiger": "호랑이",
    "tiger": "호랑이",
    "night sky outside": "밤하늘 아래 바깥 길",
    "outside at night": "밤하늘 아래 바깥 길",
    "night sky": "밤하늘",
    "white ball": "하얀 공",
    "baseball": "하얀 공",
    "sharing": "나눔",
    "children's drawing": "아이의 그림",
    "happy": "행복한 마음",
    "warm": "따뜻한",
    "joyful": "즐거운",
    "warm and cheerful": "따뜻하고 즐거운",
    "warm and easy": "따뜻하고 편안한",
    "calm and curious": "차분하고 호기심 어린",
    "calm and joyful": "차분하고 즐거운",
    "calm and magical": "차분하고 신비로운",
    "wonder": "신비로운",
    "park": "공원",
    "in front of house": "집 앞",
    "in front of a house": "집 앞",
    "house in front": "집 앞",
    "door": "문",
    "friendship under the stars": "별빛 아래 나누는 우정",
    "family bonding under the stars": "별빛 아래 나누는 다정한 마음",
    "family fun under the stars": "별빛 아래 나누는 즐거운 마음",
    "family joy under the stars": "별빛 아래 가족이 나누는 기쁨",
    "nature exploration": "자연을 살피는 모험",
}


def _clean_concepts(vision: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    """Convert noisy OpenCLIP words into story-ready Korean concepts."""
    object_scores = vision.get("object_scores", {})
    objects = vision.get("objects", [])
    concepts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for word in objects:
        key = str(word).lower().strip()
        if key in STOPWORDS or len(key) <= 1 or key in seen:
            continue
        label_ko = CONCEPT_KO.get(key)
        if not label_ko:
            continue
        seen.add(key)
        concepts.append(
            {
                "source": key,
                "label_ko": label_ko,
                "score": object_scores.get(word),
            }
        )
        if len(concepts) >= limit:
            break
    if not concepts:
        concepts.append({"source": "drawing", "label_ko": "그림", "score": None})
    return concepts


def _join_people(labels: list[str]) -> str:
    """Join Korean character labels with natural particles."""
    if not labels:
        return "아이"
    if len(labels) == 1:
        return labels[0]
    head = labels[0]
    for label in labels[1:]:
        head = f"{head}{_particle(head, '과', '와')} {label}"
    return head


def _join_items(labels: list[str]) -> str:
    """Join Korean item labels for story prose."""
    if not labels:
        return "그림"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", 그리고 {labels[-1]}"


def _has_jongseong(text: str) -> bool:
    """Return whether the last Hangul syllable has a final consonant."""
    for char in reversed(text.strip()):
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28 != 0
        if char.isalnum():
            return False
    return False


def _particle(text: str, consonant: str, vowel: str) -> str:
    """Pick a Korean particle based on the previous word's final consonant."""
    return consonant if _has_jongseong(text) else vowel


def _as_subject(text: str) -> str:
    """Attach 이/가 to a Korean phrase."""
    return f"{text}{_particle(text, '이', '가')}"


def _as_topic(text: str) -> str:
    """Attach 은/는 to a Korean phrase."""
    return f"{text}{_particle(text, '은', '는')}"


def _as_object(text: str) -> str:
    """Attach 을/를 to a Korean phrase."""
    return f"{text}{_particle(text, '을', '를')}"


def _scene_label(index: int) -> str:
    """Return a Korean sequence label for a scene paragraph."""
    labels = {
        1: "첫 번째 그림에서는",
        2: "두 번째 그림에서는",
        3: "세 번째 그림에서는",
        4: "네 번째 그림에서는",
        5: "다섯 번째 그림에서는",
        6: "여섯 번째 그림에서는",
        7: "일곱 번째 그림에서는",
        8: "여덟 번째 그림에서는",
        9: "아홉 번째 그림에서는",
        10: "열 번째 그림에서는",
    }
    return labels.get(index, f"{index}번째 그림에서는")


def _coerce_str_list(value: Any, default: list[str]) -> list[str]:
    """Return a clean list of non-empty Korean strings."""
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [str(item) for item in value if str(item).strip()]
    else:
        candidates = default
    result = [item.strip() for item in candidates if item.strip()]
    return result or default


def _to_korean_hint(value: Any) -> str:
    """Translate common English model labels into Korean story labels."""
    text = str(value).strip()
    key = text.lower()
    return PHRASE_KO.get(key) or CONCEPT_KO.get(key) or text


def _coerce_ko_list(value: Any, default: list[str]) -> list[str]:
    """Return a clean list with common English labels normalized to Korean."""
    result: list[str] = []
    for item in _coerce_str_list(value, default):
        normalized = _to_korean_hint(item)
        if _looks_mostly_english(normalized):
            continue
        if normalized in STOPWORDS:
            continue
        result.append(normalized)
    return result or default


def _looks_mostly_english(text: str) -> bool:
    """Detect text that is mostly English instead of Korean story prose."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    ascii_letters = [char for char in letters if ord(char) < 128]
    return len(ascii_letters) / len(letters) > 0.6


def _has_hangul(text: str) -> bool:
    return any("\uac00" <= char <= "\ud7a3" for char in text)


def _is_placeholder_text(text: str) -> bool:
    stripped = str(text).strip()
    return not stripped or stripped in {"...", "…", "\"...\"", "['...']", "[\"...\"]"}


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
    return candidates


def _json_object_candidates(text: str) -> list[str]:
    """Return balanced JSON object candidates, preserving response order."""
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


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the most likely JSON object from an EXAONE response."""
    candidates = _json_object_candidates(text)
    if not candidates:
        raise ValueError("EXAONE response did not contain a JSON object.")
    last_error: Exception | None = None
    parsed_objects: list[dict[str, Any]] = []
    for candidate in reversed(candidates):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(payload, dict):
            parsed_objects.append(payload)
            if "structured_json" in payload and "plan_json" in payload:
                return payload
        last_error = ValueError("EXAONE JSON response was not an object.")
    if parsed_objects:
        return parsed_objects[0]
    if last_error:
        raise last_error
    raise ValueError("EXAONE response did not contain a JSON object.")


def _build_structured_plan_prompt(vision: dict[str, Any]) -> str:
    """Build the shared EXAONE prompt for Korean structure and story planning."""
    return (
        "다음 vision_json은 손그림을 BLIP/OpenCLIP으로 분석한 영어 단서입니다.\n"
        "영어 단서를 한국어로 해석해, 한국어 동화 생성을 위한 구조와 계획을 만드세요.\n"
        "반드시 JSON 객체 하나만 출력하세요. 설명, 마크다운, 코드블록, 프롬프트 반복은 쓰지 마세요.\n"
        "모든 문자열 값은 한국어로 쓰세요. children, girl, dog 같은 영어 단어를 그대로 쓰지 마세요.\n"
        "문장을 길게 늘이지 말고 각 필드는 짧고 완결된 한국어로 쓰세요.\n"
        "보이는 단서에 근거하되, 아이 손그림에서 자연스럽게 추론 가능한 달/별/밤/바구니 같은 요소는 "
        "raw_caption이나 objects에 없더라도 단서가 있으면 story_items에 반영해도 됩니다.\n\n"
        "필수 JSON 형식:\n"
        "{\n"
        '  "structured_json": {\n'
        '    "characters": ["..."],\n'
        '    "place": "...",\n'
        '    "visible_items": ["..."],\n'
        '    "story_items": ["..."],\n'
        '    "mood": "...",\n'
        '    "theme": "...",\n'
        '    "main_event": "..."\n'
        "  },\n"
        '  "plan_json": {\n'
        '    "title": "...",\n'
        '    "beginning": "...",\n'
        '    "middle": "...",\n'
        '    "ending": "...",\n'
        '    "style": {"audience": "어린이", "tone": "따뜻하고 쉬운 문장", "length": "3~5문장"}\n'
        "  }\n"
        "}\n\n"
        f"vision_json:\n{json.dumps(vision, ensure_ascii=False, indent=2)}\n"
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    raw_value = str(payload.get(key, "")).strip()
    value = _to_korean_hint(raw_value)
    if _is_placeholder_text(value):
        raise ValueError(f"EXAONE JSON missing text field: {key}")
    if _looks_mostly_english(value):
        raise ValueError(f"EXAONE JSON field is not Korean enough: {key}")
    if not _has_hangul(value):
        raise ValueError(f"EXAONE JSON field does not contain Korean text: {key}")
    return _to_korean_hint(value)


def _required_structured_text(payload: dict[str, Any], key: str) -> str:
    raw_value = str(payload.get(key, "")).strip()
    value = _to_korean_hint(raw_value)
    if _is_placeholder_text(value):
        raise ValueError(f"EXAONE JSON missing text field: {key}")
    return value


def _required_ko_list(payload: dict[str, Any], key: str, limit: int) -> list[str]:
    values = [value for value in _coerce_ko_list(payload.get(key), []) if not _is_placeholder_text(value)]
    if not values:
        raise ValueError(f"EXAONE JSON missing list field: {key}")
    if not any(_has_hangul(value) for value in values):
        raise ValueError(f"EXAONE JSON list does not contain Korean text: {key}")
    return values[:limit]


def _required_structured_list(payload: dict[str, Any], key: str, limit: int) -> list[str]:
    values = [
        _to_korean_hint(value)
        for value in _coerce_str_list(payload.get(key), [])
        if not _is_placeholder_text(value)
    ]
    if not values:
        raise ValueError(f"EXAONE JSON missing list field: {key}")
    return values[:limit]


def _language_quality_warnings(fields: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(f"{prefix}.{child_key}" if prefix else str(child_key), child_value)
            return
        if isinstance(value, list):
            for index, child_value in enumerate(value):
                visit(f"{prefix}[{index}]", child_value)
            return
        text = str(value or "").strip()
        if text and _looks_mostly_english(text):
            warnings.append(f"{prefix}:mostly_english")

    visit("", fields)
    return warnings


def _normalize_exaone_story_schema(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize EXAONE JSON without treating English residue as a hard failure."""
    structured_raw = payload.get("structured_json", payload)
    plan_raw = payload.get("plan_json", {})
    if not isinstance(structured_raw, dict):
        raise ValueError("EXAONE JSON missing structured_json object.")
    if not isinstance(plan_raw, dict):
        raise ValueError("EXAONE JSON missing plan_json object.")

    structured = {
        "characters": _required_structured_list(structured_raw, "characters", 3),
        "place": _required_structured_text(structured_raw, "place"),
        "visible_items": _required_structured_list(structured_raw, "visible_items", 8),
        "story_items": _required_structured_list(structured_raw, "story_items", 5),
        "mood": _required_structured_text(structured_raw, "mood"),
        "theme": _required_structured_text(structured_raw, "theme"),
        "main_event": _to_korean_hint(str(structured_raw.get("main_event", "")).strip()),
        "source": "exaone",
    }

    plan = {
        "title": _required_structured_text(plan_raw, "title"),
        "beginning": _required_structured_text(plan_raw, "beginning"),
        "middle": _required_structured_text(plan_raw, "middle"),
        "ending": _required_structured_text(plan_raw, "ending"),
        "style": plan_raw.get("style") if isinstance(plan_raw.get("style"), dict) else {},
        "source": "exaone",
    }
    warnings = _language_quality_warnings(
        {
            "structured_json": structured,
            "plan_json": {key: value for key, value in plan.items() if key != "style"},
        }
    )
    if warnings:
        structured["validation_warnings"] = warnings
        plan["validation_warnings"] = warnings
    return structured, plan


def _build_structured_plan_repair_prompt(raw_response: str, vision: dict[str, Any]) -> str:
    return (
        "아래 모델 응답을 유효한 JSON 객체 하나로만 고치세요.\n"
        "설명, 마크다운, 코드블록, 프롬프트 반복은 쓰지 마세요.\n"
        "모든 문자열 값은 반드시 한국어로 쓰세요. 영어 단어와 영어 문장을 그대로 두지 마세요.\n"
        "vision_json의 시각 단서에 근거해서 빠진 필드를 채우되, 동화 본문을 길게 쓰지 마세요.\n"
        "필수 JSON 형식:\n"
        "{\n"
        '  "structured_json": {\n'
        '    "characters": ["..."],\n'
        '    "place": "...",\n'
        '    "visible_items": ["..."],\n'
        '    "story_items": ["..."],\n'
        '    "mood": "...",\n'
        '    "theme": "...",\n'
        '    "main_event": "..."\n'
        "  },\n"
        '  "plan_json": {\n'
        '    "title": "...",\n'
        '    "beginning": "...",\n'
        '    "middle": "...",\n'
        '    "ending": "...",\n'
        '    "style": {"audience": "어린이", "tone": "따뜻하고 쉬운 문장", "length": "3~5문장"}\n'
        "  }\n"
        "}\n\n"
        f"vision_json:\n{json.dumps(vision, ensure_ascii=False, indent=2)}\n\n"
        "model_response:\n"
        f"{raw_response}\n"
    )


def generate_structured_plan_exaone(
    vision: dict[str, Any],
    max_new_tokens: int = 700,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use EXAONE once to translate vision JSON into Korean structured/plan JSON."""
    prompt = _build_structured_plan_prompt(vision)

    with timed_step(8, "EXAONE structured story planning", model="LGAI-EXAONE/EXAONE-4.0-1.2B HF"):
        import torch

        tokenizer, model = get_exaone_components()
        device = get_device()
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.03,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    try:
        payload = _extract_json_object(raw_response)
        structured, plan = _normalize_exaone_story_schema(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        repair_prompt = _build_structured_plan_repair_prompt(raw_response, vision)
        with timed_step(9, "EXAONE structured JSON repair", model="LGAI-EXAONE/EXAONE-4.0-1.2B HF"):
            import torch

            tokenizer, model = get_exaone_components()
            device = get_device()
            messages = [{"role": "user", "content": repair_prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.03,
                )
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            repair_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        try:
            payload = _extract_json_object(repair_response)
            structured, plan = _normalize_exaone_story_schema(payload)
        except (json.JSONDecodeError, ValueError, TypeError) as repair_exc:
            raise RuntimeError(
                "EXAONE did not return valid structured/plan JSON, and JSON repair also failed. "
                f"initial_error={exc}; repair_error={repair_exc}; "
                f"raw_response_head={raw_response[:800]!r}; repair_response_head={repair_response[:800]!r}"
            ) from repair_exc
        raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
    return structured, plan, raw_response


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def get_last_llama_runtime() -> dict[str, Any]:
    """Return metadata for the last llama.cpp invocation."""
    return dict(LAST_LLAMA_RUNTIME)


def _strip_outer_code_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _llama_prompt_with_response_marker(prompt: str, marker: str) -> str:
    return f"{prompt.rstrip()}\n{marker}\n"


def _clean_llama_output(stdout: str, response_marker: str | None = None) -> tuple[str, bool]:
    raw_response = stdout.split("[ Prompt:", 1)[0]
    raw_response = raw_response.replace("Exiting...", "").replace("\r\n", "\n").strip()
    raw_response = re.sub(
        r"(?s)^.*?available commands:.*?\n\n",
        "",
        raw_response,
    )
    marker_stripped = False
    if response_marker and response_marker in raw_response:
        raw_response = raw_response.rsplit(response_marker, 1)[-1]
        marker_stripped = True
    raw_response = raw_response.strip()
    raw_response = re.sub(r"^__LLAMA_RESPONSE_START_[^\s\n]*\s*", "", raw_response).strip()
    raw_response = re.sub(r"^\s*>\s*", "", raw_response).strip()
    return _strip_outer_code_fence(raw_response), marker_stripped


def _llama_gpu_layers() -> int:
    return configured_llama_gpu_layers()


def _copy_llama_cli_candidate(candidate: str, llama_path: str) -> str | None:
    """Copy a discovered llama-cli binary into the project-local default path."""
    if not candidate or not os.path.exists(candidate):
        return None
    if not _is_llama_cli_usable(candidate):
        print(f"[llama] Ignoring unusable llama-cli candidate: {candidate}")
        return None
    try:
        os.makedirs(os.path.dirname(llama_path), exist_ok=True)
        if os.path.abspath(candidate) != os.path.abspath(llama_path):
            candidate_dir = os.path.dirname(candidate)
            llama_dir = os.path.dirname(llama_path)
            for name in os.listdir(candidate_dir):
                source = os.path.join(candidate_dir, name)
                if os.path.isfile(source):
                    shutil.copy2(source, os.path.join(llama_dir, name))
        print(f"[llama] Prepared llama-cli at {llama_path}")
        return llama_path
    except OSError as exc:
        print(f"[llama] Found llama-cli but could not copy it to project path: {exc}")
        return candidate


def _is_llama_cli_usable(candidate: str) -> bool:
    """Return whether the discovered llama-cli can actually be executed."""
    try:
        completed = subprocess.run(
            [candidate, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        return completed.returncode in {0, 1}
    except (OSError, subprocess.SubprocessError):
        return False


def _find_llama_cli_installation() -> str | None:
    """Find llama-cli from PATH or common winget installation locations."""
    path_hit = shutil.which(LLAMA_CLI_FILENAME)
    if path_hit:
        return path_hit

    candidates: list[str] = []
    if os.name == "nt":
        roots = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages"),
            os.path.join(os.environ.get("ProgramFiles", ""), "WinGet", "Packages"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps"),
        ]
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for current_root, _dirs, files in os.walk(root):
                if LLAMA_CLI_FILENAME in files:
                    candidates.append(os.path.join(current_root, LLAMA_CLI_FILENAME))
                    break
            if candidates:
                break

    return candidates[0] if candidates else None


def _download_llama_cpp_release(llama_path: str) -> str | None:
    """Download a portable llama.cpp release and place llama-cli in the default path."""
    if os.name != "nt":
        return None

    release_url = os.environ.get("LLAMA_CPP_RELEASE_URL", LLAMA_CPP_WINDOWS_RELEASE_URL)
    tools_root = os.path.dirname(os.path.dirname(os.path.dirname(llama_path)))
    download_dir = os.path.join(tools_root, "downloads")
    extract_dir = os.path.join(tools_root, "release")
    archive_path = os.path.join(download_dir, os.path.basename(release_url))

    try:
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(extract_dir, exist_ok=True)
        print(f"[llama] Downloading llama.cpp release: {release_url}")
        urllib.request.urlretrieve(release_url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)

        for current_root, _dirs, files in os.walk(extract_dir):
            if LLAMA_CLI_FILENAME not in files:
                continue
            candidate = os.path.join(current_root, LLAMA_CLI_FILENAME)
            prepared_cli = _copy_llama_cli_candidate(candidate, llama_path)
            if prepared_cli:
                return prepared_cli
    except (OSError, zipfile.BadZipFile, urllib.error.URLError) as exc:
        print(f"[llama] llama.cpp release download failed: {exc}")
    return None


def _maybe_prepare_llama_cli(llama_cli: str) -> str:
    """Find or prepare llama.cpp locally when missing."""
    llama_path = os.path.abspath(os.path.expanduser(llama_cli))
    if os.path.exists(llama_path):
        if _is_llama_cli_usable(llama_path):
            return llama_path
        print(f"[llama] Ignoring unusable project-local llama-cli: {llama_path}")

    installed_cli = _find_llama_cli_installation()
    prepared_cli = _copy_llama_cli_candidate(installed_cli or "", llama_path)
    if prepared_cli:
        return prepared_cli

    if not _env_flag("AUTO_INSTALL_LLAMA_CPP", True):
        return llama_path

    if os.name == "nt":
        winget_exe = shutil.which("winget")
        if winget_exe:
            try:
                print("[llama] llama-cli not found; installing llama.cpp with winget...")
                subprocess.run(
                    [
                        winget_exe,
                        "install",
                        "--id",
                        "ggml.llamacpp",
                        "--exact",
                        "--silent",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                    ],
                    check=True,
                )
                installed_cli = _find_llama_cli_installation()
                prepared_cli = _copy_llama_cli_candidate(installed_cli or "", llama_path)
                if prepared_cli:
                    return prepared_cli
            except (OSError, subprocess.CalledProcessError) as exc:
                print(f"[llama] winget llama.cpp install failed; trying source build if possible: {exc}")

        prepared_cli = _download_llama_cpp_release(llama_path)
        if prepared_cli:
            return prepared_cli

    source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(llama_path))), "src")
    build_dir = os.path.join(source_dir, "build")
    git_exe = shutil.which("git")
    cmake_exe = shutil.which("cmake")
    if not git_exe or not cmake_exe:
        print("[llama] git or cmake was not found; skipping llama.cpp source build.")
        return llama_path

    try:
        os.makedirs(os.path.dirname(source_dir), exist_ok=True)
        if not os.path.exists(source_dir):
            subprocess.run(
                [git_exe, "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", source_dir],
                check=True,
            )
        cmake_args = [cmake_exe, "-S", source_dir, "-B", build_dir]
        if has_nvidia_gpu():
            cmake_args.append("-DGGML_CUDA=ON")
        subprocess.run(cmake_args, check=True)
        subprocess.run([cmake_exe, "--build", build_dir, "--config", "Release", "-j"], check=True)

        candidates = [
            os.path.join(build_dir, "bin", "Release", "llama-cli.exe"),
            os.path.join(build_dir, "bin", "llama-cli.exe"),
            os.path.join(build_dir, "bin", "llama-cli"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                os.makedirs(os.path.dirname(llama_path), exist_ok=True)
                shutil.copy2(candidate, llama_path)
                print(f"[llama] Prepared llama-cli at {llama_path}")
                return llama_path
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[llama] llama.cpp auto-build failed; use an existing llama-cli if available: {exc}")
    return llama_path


def ensure_exaone_gguf_runtime(model_path: str = "", llama_cli_path: str = "") -> dict[str, Any]:
    """Prepare and verify EXAONE GGUF + llama.cpp before story generation starts."""
    log_stage("ensuring EXAONE GGUF runtime", step="preflight", model="EXAONE GGUF + llama.cpp")
    resolved_model_path = ensure_exaone_gguf_model(
        model_path or os.environ.get("EXAONE_GGUF_MODEL_PATH") or DEFAULT_EXAONE_GGUF_PATH
    )
    llama_cli = _maybe_prepare_llama_cli(llama_cli_path or os.environ.get("LLAMA_CLI_PATH", LLAMA_CLI_PATH))
    setup_error = _llama_path_error(llama_cli, resolved_model_path)
    if setup_error:
        raise FileNotFoundError(setup_error)
    if not _is_llama_cli_usable(llama_cli):
        raise FileNotFoundError(f"llama.cpp CLI exists but is not executable: {llama_cli}")

    runtime = {
        "model_path": resolved_model_path,
        "llama_cli": llama_cli,
        "gpu_layers": _llama_gpu_layers(),
        "cuda_cpu_retry": _env_flag("LLAMA_CUDA_RETRY_CPU", True),
    }
    log_stage(f"EXAONE GGUF ready: {resolved_model_path}", step="preflight", model="EXAONE-4.0-1.2B-IQ4_XS.gguf")
    log_stage(f"llama-cli ready: {llama_cli}", step="preflight", model="llama.cpp")
    return runtime


def _llama_path_error(llama_cli: str, model_path: str) -> str | None:
    """Return a clear setup error before invoking llama.cpp."""
    missing: list[str] = []
    if not os.path.exists(llama_cli):
        if _env_flag("AUTO_INSTALL_LLAMA_CPP", True):
            auto_install_hint = (
                " Automatic install/build was attempted but did not produce llama-cli; "
                "install winget/git/cmake or set LLAMA_CLI_PATH."
            )
        else:
            auto_install_hint = " AUTO_INSTALL_LLAMA_CPP=0 is set; enable it or set LLAMA_CLI_PATH."
        missing.append(
            "llama.cpp CLI not found. Set LLAMA_CLI_PATH or build/place it at: "
            f"{llama_cli}.{auto_install_hint}"
        )
    if not os.path.exists(model_path):
        missing.append(
            "EXAONE GGUF model file not found. Set EXAONE_GGUF_MODEL_PATH "
            f"or place the model at: {model_path}."
        )
    if missing:
        return " ".join(missing)
    return None


def _llama_cwd() -> str:
    """Return the stable cwd used for llama.cpp invocations."""
    return str(PROJECT_ROOT)


def _path_for_llama_cli(path: str) -> str:
    """Prefer project-relative paths so llama.cpp avoids non-ASCII absolute paths."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _local_exaone_gguf_copy_path(source_path: str) -> str:
    """Copy an external GGUF into the project-local model folder for relative-path retry."""
    source = Path(source_path).expanduser().resolve()
    target = Path(DEFAULT_EXAONE_GGUF_PATH).expanduser().resolve().parent / source.name
    if source == target:
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return str(target)


def _text_tail(value: Any, limit: int = 2000) -> str:
    text = "" if value is None else str(value)
    return text[-limit:] if len(text) > limit else text


def _llama_failure_details(exc: BaseException) -> dict[str, Any]:
    details: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    returncode = getattr(exc, "returncode", None)
    if returncode is not None:
        details["returncode"] = returncode
    stdout = getattr(exc, "stdout", None)
    stderr = getattr(exc, "stderr", None)
    if stdout:
        details["stdout_tail"] = _text_tail(stdout)
    if stderr:
        details["stderr_tail"] = _text_tail(stderr)
    return details


def _llama_error_text(exc: BaseException) -> str:
    details = _llama_failure_details(exc)
    return "\n".join(str(value) for value in details.values() if value)


def _is_llama_model_load_failure(exc: BaseException) -> bool:
    text = _llama_error_text(exc).lower()
    return any(
        marker in text
        for marker in (
            "failed to load the model",
            "failed to load model",
            "unable to load model",
            "error loading model",
        )
    )


def _format_llama_failure(exc: BaseException) -> str:
    details = _llama_failure_details(exc)
    parts = [details["error"]]
    for key in ("returncode", "stderr_tail", "stdout_tail"):
        value = details.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value!r}")
    return "; ".join(parts)


def _llama_success_runtime(
    *,
    mode: str,
    gpu_layers: int,
    cpu_retry_used: bool,
    llama_cli: str,
    model_path: str,
    model_path_arg: str,
    cwd: str,
    completed: subprocess.CompletedProcess[str],
    cleaned_response: str,
    response_marker_stripped: bool,
    **extra: Any,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "mode": mode,
        "gpu_layers": gpu_layers,
        "cpu_retry_used": cpu_retry_used,
        "llama_cli": llama_cli,
        "model_path": model_path,
        "model_path_arg": model_path_arg,
        "cwd": cwd,
        "returncode": completed.returncode,
        "stdout_tail": _text_tail(completed.stdout),
        "stderr_tail": _text_tail(completed.stderr),
        "cleaned_response_tail": _text_tail(cleaned_response),
        "response_marker_stripped": response_marker_stripped,
    }
    runtime.update(extra)
    return runtime


def _build_llama_command(
    llama_cli: str,
    model_path: str,
    prompt_path: str,
    max_new_tokens: int,
    context_size: int,
    temperature: str,
    top_p: str | None,
    gpu_layers: int,
    force_cpu: bool = False,
    json_schema: str | None = None,
) -> list[str]:
    command = [
        llama_cli,
        "-m",
        model_path,
        "-f",
        prompt_path,
        "-n",
        str(max_new_tokens),
        "-c",
        str(context_size),
        "--temp",
        temperature,
    ]
    if top_p is not None:
        command.extend(["--top-p", top_p])
    if json_schema:
        command.extend(["--json-schema", json_schema])
    if force_cpu or gpu_layers <= 0:
        command.extend(["-ngl", "0", "--device", "none"])
    else:
        command.extend(["-ngl", str(gpu_layers)])
        llama_device = os.environ.get("LLAMA_DEVICE")
        if llama_device:
            command.extend(["--device", llama_device])
    command.extend(
        [
            "--single-turn",
            "--log-disable",
            "--no-warmup",
            "--simple-io",
            "--no-display-prompt",
        ]
    )
    return command


def _run_llama_prompt(
    prompt: str,
    max_new_tokens: int,
    model_path: str = "",
    timeout: int = 180,
    context_size: int = 4096,
    temperature: str = "0.55",
    top_p: str | None = "0.9",
    json_schema: str | dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run llama.cpp with GPU offload when available and CPU retry when needed."""
    resolved_model_path = ensure_exaone_gguf_model(
        model_path or os.environ.get("EXAONE_GGUF_MODEL_PATH") or DEFAULT_EXAONE_GGUF_PATH
    )
    llama_cli = _maybe_prepare_llama_cli(os.environ.get("LLAMA_CLI_PATH", LLAMA_CLI_PATH))
    setup_error = _llama_path_error(llama_cli, resolved_model_path)
    run_cwd = _llama_cwd()
    if setup_error:
        runtime = {
            "mode": "unavailable",
            "cpu_retry_used": False,
            "llama_cli": llama_cli,
            "model_path": resolved_model_path,
            "cwd": run_cwd,
            "error": setup_error,
        }
        LAST_LLAMA_RUNTIME.clear()
        LAST_LLAMA_RUNTIME.update(runtime)
        raise FileNotFoundError(setup_error)

    gpu_layers = _llama_gpu_layers()
    cpu_retry_enabled = _env_flag("LLAMA_CUDA_RETRY_CPU", True)
    force_cpu = gpu_layers <= 0

    response_marker = f"__LLAMA_RESPONSE_START_{uuid.uuid4().hex}__"
    prompt_for_llama = _llama_prompt_with_response_marker(prompt, response_marker)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as prompt_file:
        prompt_file.write(prompt_for_llama)
        prompt_path = prompt_file.name
    try:
        llama_model_path = _path_for_llama_cli(resolved_model_path)
        json_schema_arg = json.dumps(json_schema, ensure_ascii=False) if isinstance(json_schema, dict) else json_schema
        command = _build_llama_command(
            llama_cli,
            llama_model_path,
            prompt_path,
            max_new_tokens,
            context_size,
            temperature,
            top_p,
            gpu_layers,
            force_cpu=force_cpu,
            json_schema=json_schema_arg,
        )
        mode = "cpu_forced" if os.environ.get("LLAMA_GPU_LAYERS") == "0" else "cpu_only" if force_cpu else "gpu"
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=run_cwd,
            )
            cleaned_response, marker_stripped = _clean_llama_output(completed.stdout, response_marker)
            runtime = _llama_success_runtime(
                mode=mode,
                gpu_layers=0 if force_cpu else gpu_layers,
                cpu_retry_used=False,
                llama_cli=llama_cli,
                model_path=resolved_model_path,
                model_path_arg=llama_model_path,
                cwd=run_cwd,
                completed=completed,
                cleaned_response=cleaned_response,
                response_marker_stripped=marker_stripped,
            )
            LAST_LLAMA_RUNTIME.clear()
            LAST_LLAMA_RUNTIME.update(runtime)
            return cleaned_response, runtime
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if force_cpu or not cpu_retry_enabled:
                failure = _llama_failure_details(exc)
                retry_error = exc
                retry_model_path = resolved_model_path
                retry_model_arg = llama_model_path
                retry_used_local_copy = False
                if _is_llama_model_load_failure(exc):
                    try:
                        retry_model_path = _local_exaone_gguf_copy_path(resolved_model_path)
                        retry_model_arg = _path_for_llama_cli(retry_model_path)
                        if retry_model_path != resolved_model_path:
                            retry_used_local_copy = True
                            retry_command = _build_llama_command(
                                llama_cli,
                                retry_model_arg,
                                prompt_path,
                                max_new_tokens,
                                context_size,
                                temperature,
                                top_p,
                                gpu_layers,
                                force_cpu=force_cpu,
                                json_schema=json_schema_arg,
                            )
                            completed = subprocess.run(
                                retry_command,
                                check=True,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                timeout=timeout,
                                cwd=run_cwd,
                            )
                            cleaned_response, marker_stripped = _clean_llama_output(
                                completed.stdout,
                                response_marker,
                            )
                            runtime = _llama_success_runtime(
                                mode=f"{mode}_local_model_retry",
                                gpu_layers=0 if force_cpu else gpu_layers,
                                cpu_retry_used=False,
                                llama_cli=llama_cli,
                                model_path=retry_model_path,
                                model_path_arg=retry_model_arg,
                                cwd=run_cwd,
                                completed=completed,
                                cleaned_response=cleaned_response,
                                response_marker_stripped=marker_stripped,
                                local_model_retry_used=True,
                                original_model_path=resolved_model_path,
                                initial_error=failure,
                            )
                            LAST_LLAMA_RUNTIME.clear()
                            LAST_LLAMA_RUNTIME.update(runtime)
                            return cleaned_response, runtime
                    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as retry_exc:
                        retry_error = retry_exc
                LAST_LLAMA_RUNTIME.clear()
                LAST_LLAMA_RUNTIME.update(
                    {
                        "mode": mode,
                        "gpu_layers": 0 if force_cpu else gpu_layers,
                        "cpu_retry_used": False,
                        "local_model_retry_used": retry_used_local_copy,
                        "llama_cli": llama_cli,
                        "model_path": retry_model_path,
                        "model_path_arg": retry_model_arg,
                        "original_model_path": resolved_model_path,
                        "cwd": run_cwd,
                        "error": _format_llama_failure(retry_error),
                        "initial_error": failure,
                    }
                )
                raise RuntimeError(_format_llama_failure(retry_error)) from retry_error
            print(f"[llama] GPU run failed; retrying on CPU: {exc}")
            cpu_command = _build_llama_command(
                llama_cli,
                llama_model_path,
                prompt_path,
                max_new_tokens,
                context_size,
                temperature,
                top_p,
                gpu_layers=0,
                force_cpu=True,
                json_schema=json_schema_arg,
            )
            try:
                completed = subprocess.run(
                    cpu_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    cwd=run_cwd,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as cpu_exc:
                cpu_failure = _llama_failure_details(cpu_exc)
                retry_error = cpu_exc
                retry_model_path = resolved_model_path
                retry_model_arg = llama_model_path
                retry_used_local_copy = False
                if _is_llama_model_load_failure(cpu_exc):
                    try:
                        retry_model_path = _local_exaone_gguf_copy_path(resolved_model_path)
                        retry_model_arg = _path_for_llama_cli(retry_model_path)
                        if retry_model_path != resolved_model_path:
                            retry_used_local_copy = True
                            retry_command = _build_llama_command(
                                llama_cli,
                                retry_model_arg,
                                prompt_path,
                                max_new_tokens,
                                context_size,
                                temperature,
                                top_p,
                                gpu_layers=0,
                                force_cpu=True,
                                json_schema=json_schema_arg,
                            )
                            completed = subprocess.run(
                                retry_command,
                                check=True,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                timeout=timeout,
                                cwd=run_cwd,
                            )
                            cleaned_response, marker_stripped = _clean_llama_output(
                                completed.stdout,
                                response_marker,
                            )
                            runtime = _llama_success_runtime(
                                mode="cpu_retry_local_model_retry",
                                gpu_layers=0,
                                cpu_retry_used=True,
                                llama_cli=llama_cli,
                                model_path=retry_model_path,
                                model_path_arg=retry_model_arg,
                                cwd=run_cwd,
                                completed=completed,
                                cleaned_response=cleaned_response,
                                response_marker_stripped=marker_stripped,
                                local_model_retry_used=True,
                                original_model_path=resolved_model_path,
                                gpu_error=_llama_failure_details(exc),
                                cpu_error=cpu_failure,
                            )
                            LAST_LLAMA_RUNTIME.clear()
                            LAST_LLAMA_RUNTIME.update(runtime)
                            return cleaned_response, runtime
                    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as retry_exc:
                        retry_error = retry_exc
                LAST_LLAMA_RUNTIME.clear()
                LAST_LLAMA_RUNTIME.update(
                    {
                        "mode": "cpu_retry",
                        "gpu_layers": 0,
                        "cpu_retry_used": True,
                        "local_model_retry_used": retry_used_local_copy,
                        "llama_cli": llama_cli,
                        "model_path": retry_model_path,
                        "model_path_arg": retry_model_arg,
                        "original_model_path": resolved_model_path,
                        "cwd": run_cwd,
                        "gpu_error": _llama_failure_details(exc),
                        "cpu_error": cpu_failure,
                        "error": _format_llama_failure(retry_error),
                    }
                )
                raise RuntimeError(_format_llama_failure(retry_error)) from retry_error
            cleaned_response, marker_stripped = _clean_llama_output(completed.stdout, response_marker)
            runtime = _llama_success_runtime(
                mode="cpu_retry",
                gpu_layers=0,
                cpu_retry_used=True,
                llama_cli=llama_cli,
                model_path=resolved_model_path,
                model_path_arg=llama_model_path,
                cwd=run_cwd,
                completed=completed,
                cleaned_response=cleaned_response,
                response_marker_stripped=marker_stripped,
                local_model_retry_used=False,
                gpu_error=_llama_failure_details(exc),
            )
            LAST_LLAMA_RUNTIME.clear()
            LAST_LLAMA_RUNTIME.update(runtime)
            return cleaned_response, runtime
    finally:
        os.unlink(prompt_path)


def generate_structured_plan_exaone_gguf(
    vision: dict[str, Any],
    max_new_tokens: int = 700,
    model_path: str = "",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use EXAONE GGUF through llama.cpp for Korean structured/plan JSON."""
    marker = "__EXAONE_RESPONSE_START__"
    prompt = _build_structured_plan_prompt(vision)
    prompt = (
        "### 지시\n"
        f"{prompt}\n"
        f"{marker}\n"
    )

    with timed_step(8, "EXAONE GGUF structured story planning", model="EXAONE-4.0-1.2B-IQ4_XS.gguf"):
        raw_response, _runtime = _run_llama_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            model_path=model_path,
            timeout=240,
            context_size=4096,
            temperature="0",
            top_p=None,
        )
        if marker in raw_response:
            raw_response = raw_response.rsplit(marker, 1)[-1]
        raw_response = raw_response.strip()

    try:
        payload = _extract_json_object(raw_response)
        structured, plan = _normalize_exaone_story_schema(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        repair_prompt = _build_structured_plan_repair_prompt(raw_response, vision)
        repair_response = _run_exaone_gguf_prompt(
            repair_prompt,
            max_new_tokens=max(max_new_tokens, 700),
            model_path=model_path,
            timeout=240,
            context_size=4096,
        )
        try:
            payload = _extract_json_object(repair_response)
            structured, plan = _normalize_exaone_story_schema(payload)
        except (json.JSONDecodeError, ValueError, TypeError) as repair_exc:
            raise RuntimeError(
                "EXAONE GGUF did not return valid structured/plan JSON, and JSON repair also failed. "
                f"initial_error={exc}; repair_error={repair_exc}; "
                f"raw_response_head={raw_response[:800]!r}; repair_response_head={repair_response[:800]!r}"
            ) from repair_exc
        raw_response = f"{raw_response}\n\n[json_repair_response]\n{repair_response}"
    return structured, plan, raw_response


def _run_exaone_gguf_prompt(
    prompt: str,
    max_new_tokens: int,
    model_path: str = "",
    timeout: int = 180,
    context_size: int = 4096,
    json_schema: str | dict[str, Any] | None = None,
) -> str:
    """Run an EXAONE GGUF prompt through the local llama.cpp CLI."""
    raw_response, _runtime = _run_llama_prompt(
        prompt,
        max_new_tokens=max_new_tokens,
        model_path=model_path,
        timeout=timeout,
        context_size=context_size,
        temperature="0.55",
        top_p="0.9",
        json_schema=json_schema,
    )
    return raw_response.strip()


def _short_text(value: Any, limit: int = 120) -> str:
    """Keep model prompts compact by trimming verbose scene hints."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _count_story_sentences(story: str) -> int:
    """Count simple sentence endings in generated Korean story text."""
    return len([part for part in re.split(r"(?<=[.!?。])\s+", story.strip()) if part.strip()])


def _clean_sequence_story_text(story: str, marker: str | None = None) -> str:
    cleaned = story.strip()
    if marker and marker in cleaned:
        cleaned = cleaned.rsplit(marker, 1)[-1]
    if "(truncated)" in cleaned:
        cleaned = cleaned.split("(truncated)", 1)[-1].strip()
    cleaned = re.sub(r"^\s*>\s*", "", cleaned).strip()
    cleaned = _strip_outer_code_fence(cleaned)
    cleaned = re.sub(r"^markdown\s+", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"^story_body:\s*", "", cleaned, flags=re.I).strip()
    return cleaned


def _build_sequence_story_rewrite_prompt(
    raw_story: str,
    compact_scenes: list[dict[str, Any]],
    required_story_sentences: int,
) -> str:
    return (
        "아래 모델 응답은 최종 동화 본문으로 쓰기에 부족합니다.\n"
        "ordered_scenes의 순서와 시각 근거만 사용해서 한국어 동화 본문만 다시 작성하세요.\n"
        "제목, JSON, 마크다운, 설명, 프롬프트 반복 없이 본문만 출력하세요.\n"
        f"최소 {required_story_sentences}문장 이상으로 쓰고, 각 장면의 사건이 자연스럽게 이어져야 합니다.\n\n"
        f"ordered_scenes:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}\n\n"
        "previous_model_response:\n"
        f"{raw_story}\n"
    )


def generate_sequence_story_exaone_gguf(
    scene_records: list[dict[str, Any]],
    max_new_tokens: int = 420,
) -> tuple[str, str]:
    """Ask EXAONE GGUF to write one non-repetitive story from ordered scenes."""
    marker = "__STORY_BODY_START__"
    compact_scenes = []
    for record in scene_records:
        structured = record["structured_json"]
        vision = record["vision"]
        compact_scenes.append(
            {
                "scene_index": record["scene_index"],
                "image_id": record["image_id"],
                "caption": _short_text(vision.get("raw_caption", ""), 90),
                "who": _short_text(vision.get("who", ""), 60),
                "actions": _short_text(vision.get("actions", ""), 80),
                "scene": _short_text(vision.get("scene", ""), 60),
                "mood": _short_text(vision.get("mood", ""), 40),
                "characters": structured.get("characters", []),
                "place": structured.get("place", ""),
                "story_items": (structured.get("story_items", []) or [])[:4],
                "theme": _short_text(structured.get("theme", ""), 60),
            }
        )
    prompt = (
        "아래는 순서가 있는 아이 손그림 장면들입니다.\n"
        "각 장면을 01, 02, 03 순서대로 이어서 하나의 한국어 동화로 써 주세요.\n"
        "조건:\n"
        "- 템플릿처럼 '첫 번째 장면', '그다음 장면'을 반복하지 마세요.\n"
        "- 장면마다 달라지는 사건이 자연스럽게 이어져야 합니다.\n"
        "- BLIP이 잘못 본 단서는 과하게 확정하지 말고, 보이는 핵심 등장인물과 분위기를 중심으로 쓰세요.\n"
        "- 각 장면마다 반드시 어린이가 읽기 쉬운 한국어로 5~7문장씩 충분히 써 주세요.\n"
        "- 예를 들어 장면이 10개라면 전체는 50~70문장이어야 합니다.\n"
        "- 전체 문장 수는 장면 수에 맞춰 늘리고, 장면별 사건을 한 문장으로 짧게 요약하지 마세요.\n"
        "- 제목, 설명, JSON, 마크다운 없이 동화 본문만 출력하세요.\n\n"
        f"ordered_scenes:\n{json.dumps(compact_scenes, ensure_ascii=False, indent=2)}\n\n"
        f"{marker}\n"
    )
    with timed_step(12, "EXAONE GGUF sequence story writing", model="EXAONE-4.0-1.2B-IQ4_XS.gguf"):
        raw_story = _run_exaone_gguf_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            timeout=240,
            context_size=8192,
        )
    raw_story = _clean_sequence_story_text(raw_story, marker)
    story = raw_story.strip()
    story_start = re.search(r"(어느\s+[^\n]+)", story)
    if story_start:
        story = story[story_start.start() :]
    story = re.sub(r"^>\s*", "", story).strip()
    story = re.sub(r"^동화 본문:\s*", "", story).strip()
    story = re.sub(r"^```(?:text)?\s*|\s*```$", "", story).strip()
    required_story_sentences = len(scene_records)
    if (
        not story
        or "exceeds the available context size" in story
        or story.startswith("Error:")
        or story.startswith("아래는 순서가 있는")
        or "ordered_scenes:" in story
        or _count_story_sentences(story) < required_story_sentences
    ):
        rewrite_prompt = _build_sequence_story_rewrite_prompt(
            raw_story,
            compact_scenes,
            required_story_sentences,
        )
        with timed_step(13, "EXAONE GGUF sequence story rewrite", model="EXAONE-4.0-1.2B-IQ4_XS.gguf"):
            rewrite_story = _run_exaone_gguf_prompt(
                rewrite_prompt,
                max_new_tokens=max_new_tokens,
                timeout=240,
                context_size=8192,
            )
        rewritten = _clean_sequence_story_text(rewrite_story)
        if (
            not rewritten
            or "ordered_scenes:" in rewritten
            or rewritten.startswith("Error:")
            or _count_story_sentences(rewritten) < required_story_sentences
        ):
            raise RuntimeError(
                "exaone_output_invalid: EXAONE GGUF did not return a valid sequence story after rewrite. "
                f"cleaned_response_head={raw_story[:800]!r}; rewrite_response_head={rewritten[:800]!r}; "
                f"llama_runtime={get_last_llama_runtime()!r}"
            )
        return rewritten, f"{raw_story}\n\n[rewrite_response]\n{rewrite_story}"
    return story, raw_story


def generate_story_en(vision: dict, max_new_tokens: int = 200) -> str:
    """Generate an English children's story from the vision JSON using GPT-2."""
    # vision JSON의 관찰 결과를 GPT-2가 이어 쓸 수 있는 이야기 도입부로 구성합니다.
    seed = (
        f"A children's story.\n\n"
        f"Once upon a time, there was {vision['raw_caption']}. "
        f"The main character was {vision['who']}, {vision['actions']} {vision['scene']}. "
        f"The mood was {vision['mood']}.\n\n"
        f"The story begins:\n"
    )

    with timed_step(8, "GPT-2 English story generation", model="gpt2-medium"):
        import torch

        # GPT-2 모델과 토크나이저는 캐시로 재사용해 반복 실행 비용을 줄입니다.
        tokenizer, model = get_gpt2_components()
        device = get_device()
        inputs = tokenizer(seed, return_tensors="pt").to(device)
        with torch.inference_mode():
            # 샘플링 파라미터를 낮은 반복성과 적당한 다양성에 맞춰 동화 문장을 생성합니다.
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        story = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return story


def translate_en_ko(text_en: str) -> str:
    """Translate English text into Korean using NLLB only."""
    with timed_step(9, "NLLB English-to-Korean translation", model="facebook/nllb-200-distilled-600M"):
        import torch

        # NLLB는 명시적인 source/target 언어 코드가 있어야 원하는 방향으로 번역됩니다.
        tokenizer, model = get_nllb_components()
        device = get_device()
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(text_en, return_tensors="pt", truncation=True).to(device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("kor_Hang")
        with torch.inference_mode():
            # forced_bos_token_id로 한국어 출력을 강제합니다.
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=512,
            )
        translation = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    return translation.strip()


