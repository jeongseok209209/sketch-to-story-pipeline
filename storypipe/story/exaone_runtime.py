"""[담당 2 · 스토리] EXAONE 기반 한국어 생성 + GGUF 런타임.

구조화 플랜 생성(HF/GGUF), 시퀀스 스토리(실험 B), 그리고 EXAONE GGUF 실행을 담당한다.

재현성 핵심: EXAONE GGUF는 과거 llama-cli 바이너리 subprocess(하드코딩 Windows-Vulkan zip
다운로드/소스빌드)에서 **llama-cpp-python(in-process)**으로 전환했다. 같은 모델·동일 llama.cpp
코어를 쓰되 바이너리 확보 의존을 없애 어느 컴퓨터에서나 pip 설치만으로 동작한다.
공개 API(`_run_exaone_gguf_prompt`, `generate_*_exaone_gguf`, `get_last_llama_runtime`,
`ensure_exaone_gguf_runtime`)는 그대로 유지하고 내부 실행부만 교체했다.
"""

from __future__ import annotations

import json
import re
from typing import Any

from storypipe.common.jsonparse import json_object_candidates as _json_object_candidates
from storypipe.common.logging import timed_step
from storypipe.common.models import ensure_exaone_gguf_model
from storypipe.common.runtime import configured_llama_gpu_layers, get_device
from storypipe.story.baseline import CONCEPT_KO, PHRASE_KO, STOPWORDS
from storypipe.story.loaders import get_exaone_components

LAST_LLAMA_RUNTIME: dict[str, Any] = {"mode": "unknown"}


# ─────────────────────────────────────────────────────────────────────────────
# EXAONE GGUF 실행부 (llama-cpp-python in-process) — 과거 llama-cli subprocess 대체
# ─────────────────────────────────────────────────────────────────────────────
def _coerce_temperature(value: str | float) -> float:
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.55


def _coerce_top_p(value: str | float | None) -> float:
    if value is None:
        return 1.0
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.9


def _maybe_grammar(json_schema: str | dict[str, Any] | None) -> Any | None:
    """json_schema(dict|str)를 llama-cpp LlamaGrammar로 변환. 실패하면 None(프롬프트가 JSON 유도)."""
    if not json_schema:
        return None
    try:
        from llama_cpp import LlamaGrammar

        schema_text = json_schema if isinstance(json_schema, str) else json.dumps(json_schema)
        return LlamaGrammar.from_json_schema(schema_text)
    except Exception as exc:
        print(f"[llama] JSON-schema grammar unavailable; continuing without it: {exc}")
        return None


def _run_llama_prompt(
    prompt: str,
    max_new_tokens: int,
    model_path: str = "",
    timeout: int = 180,
    context_size: int = 4096,
    temperature: str | float = "0.55",
    top_p: str | float | None = "0.9",
    json_schema: str | dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """EXAONE GGUF 프롬프트를 llama-cpp-python으로 실행한다.

    (``timeout``/``context_size``는 하위호환용 인자. 컨텍스트는 모델 로드 시 n_ctx로 고정한다.)
    """
    from storypipe.story.loaders import get_exaone_gguf_components

    llm = get_exaone_gguf_components(model_path)
    grammar = _maybe_grammar(json_schema)
    kwargs: dict[str, Any] = {
        "max_tokens": int(max_new_tokens),
        "temperature": _coerce_temperature(temperature),
        "top_p": _coerce_top_p(top_p),
    }
    if grammar is not None:
        kwargs["grammar"] = grammar

    gpu_layers = configured_llama_gpu_layers()
    try:
        result = llm.create_completion(prompt, **kwargs)
        text = result["choices"][0]["text"]
    except Exception as exc:
        LAST_LLAMA_RUNTIME.clear()
        LAST_LLAMA_RUNTIME.update(
            {
                "mode": "gpu" if gpu_layers > 0 else "cpu",
                "gpu_layers": gpu_layers,
                "backend": "llama-cpp-python",
                "error": str(exc),
            }
        )
        raise

    cleaned = _strip_outer_code_fence(text.strip())
    runtime = {
        "mode": "gpu" if gpu_layers > 0 else "cpu",
        "gpu_layers": gpu_layers,
        "backend": "llama-cpp-python",
        "max_new_tokens": int(max_new_tokens),
        "grammar_used": grammar is not None,
        "cleaned_response_tail": cleaned[-2000:],
    }
    LAST_LLAMA_RUNTIME.clear()
    LAST_LLAMA_RUNTIME.update(runtime)
    return cleaned, runtime


def ensure_exaone_gguf_runtime(model_path: str = "", llama_cli_path: str = "") -> dict[str, Any]:
    """EXAONE GGUF 실행 준비: 모델 파일 확보 + llama-cpp-python import 확인.

    (``llama_cli_path``는 하위호환용으로 받지만 무시한다 — 더 이상 외부 바이너리를 쓰지 않음.)
    """
    resolved_model = ensure_exaone_gguf_model(model_path)
    try:
        import llama_cpp  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "llama-cpp-python is required to run EXAONE GGUF. Install dependencies with "
            "`pip install -r requirements.txt` (or `pip install llama-cpp-python`), then re-run "
            f"`storypipe doctor`. Import error: {exc}"
        ) from exc
    gpu_layers = configured_llama_gpu_layers()
    runtime = {
        "backend": "llama-cpp-python",
        "model_path": resolved_model,
        "mode": "gpu" if gpu_layers > 0 else "cpu",
        "gpu_layers": gpu_layers,
    }
    LAST_LLAMA_RUNTIME.clear()
    LAST_LLAMA_RUNTIME.update(runtime)
    return runtime


# ─────────────────────────────────────────────────────────────────────────────
# 아래 본문은 기존 generators.py에서 이동한 코드(구조화 플랜/EXAONE HF/시퀀스 스토리).
# ─────────────────────────────────────────────────────────────────────────────
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

