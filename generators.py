"""Story generation and translation for Experiment A."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import Any

from utils import (
    DEFAULT_EXAONE_GGUF_PATH,
    LLAMA_CLI_PATH,
    get_device,
    get_exaone_components,
    get_gpt2_components,
    get_nllb_components,
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

CONCEPT_KO = {
    "baby": "아기",
    "bird": "새",
    "boy": "남자아이",
    "car": "자동차",
    "cat": "고양이",
    "child": "아이",
    "children": "아이들",
    "cloud": "구름",
    "dog": "강아지",
    "drawing": "그림",
    "family": "가족",
    "flower": "꽃",
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
    "sun": "해",
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
    "wonder": "신비로운",
    "park": "공원",
    "in front of house": "집 앞",
    "in front of a house": "집 앞",
    "door": "문",
    "friendship under the stars": "별빛 아래 나누는 우정",
    "family bonding under the stars": "별빛 아래 나누는 다정한 마음",
    "family fun under the stars": "별빛 아래 나누는 즐거운 마음",
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


def build_structured_json(vision: dict[str, Any]) -> dict[str, Any]:
    """Build structured story information from vision JSON without loading a language model."""
    concepts = _clean_concepts(vision)
    concept_words = {concept["source"] for concept in concepts}
    concept_labels = [concept["label_ko"] for concept in concepts]

    characters = [
        label
        for label in concept_labels
        if label in {"아이", "아이들", "작은 아이", "여자아이", "남자아이", "가족", "엄마", "아기", "호랑이"}
    ]
    if not characters:
        who = str(vision.get("who", "")).lower()
        if "tiger" in who and "girl" in who:
            characters = ["여자아이", "호랑이"]
        elif "tiger" in who:
            characters = ["아이", "호랑이"]
        elif "girl" in who:
            characters = ["여자아이"]
        elif "family" in who:
            characters = ["가족"]
        elif "child" in who or "children" in who:
            characters = ["아이들"]
        else:
            characters = ["아이"]
    characters = list(dict.fromkeys(characters))
    if "여자아이" in characters and "아이들" in characters:
        characters.remove("아이들")
    if "여자아이" in characters and "작은 아이" in characters:
        characters.remove("작은 아이")
    if "호랑이" in characters and "여자아이" in characters:
        characters = ["여자아이", "호랑이"]

    story_items = [
        label
        for label in concept_labels
        if label not in set(characters) | {"행복한 마음", "서 있는 모습", "그림", "바깥", "놀이"}
    ]
    if not story_items:
        story_items = [
            label
            for label in concept_labels
            if label not in set(characters) | {"행복한 마음", "서 있는 모습", "바깥", "놀이"}
        ]
    if not story_items:
        story_items = ["그림"]

    if "house" in concept_words or "home" in concept_words:
        place = "집 앞"
    elif "outside" in concept_words:
        place = "바깥 길"
    elif "tree" in concept_words or "grass" in concept_words:
        place = "바깥 풀밭"
    elif "sun" in concept_words or "sky" in concept_words:
        place = "햇살이 비치는 곳"
    else:
        scene = str(vision.get("scene", "")).strip()
        if scene.lower() == "outside":
            place = "바깥 길"
        else:
            place = scene if scene and scene.lower() not in {"unknown", "none"} else "그림 속 마을"

    mood_text = str(vision.get("mood", "")).lower()
    if "happy" in mood_text or "happy" in concept_words:
        mood = "따뜻하고 즐거운"
    elif "sad" in mood_text:
        mood = "조용하지만 다정한"
    else:
        mood = "포근한"

    return {
        "characters": characters[:3],
        "place": place,
        "visible_items": concept_labels,
        "story_items": story_items[:4],
        "mood": mood,
        "theme": "함께 보내는 소중한 하루",
        "raw_hints": {
            "caption": vision.get("raw_caption", ""),
            "who": vision.get("who", ""),
            "actions": vision.get("actions", ""),
            "scene": vision.get("scene", ""),
            "mood": vision.get("mood", ""),
        },
    }


def build_plan_json(structured: dict[str, Any]) -> dict[str, Any]:
    """Build a compact children's story plan from structured information."""
    characters = _join_people(structured.get("characters", ["아이"]))
    place = structured.get("place", "그림 속 마을")
    mood = structured.get("mood", "포근한")
    items = structured.get("story_items") or structured.get("visible_items", ["그림"])
    item_text = _join_items(items[:4])
    return {
        "title": f"{place}의 작은 이야기",
        "beginning": (
            f"{_as_subject(characters)} {place}에서 서로를 만난다."
            if "호랑이" in structured.get("characters", [])
            else f"{_as_subject(characters)} {place}에서 {_as_object(item_text)} 바라본다."
        ),
        "middle": f"모두가 {mood} 마음으로 함께 시간을 보내며 작은 즐거움을 나눈다.",
        "ending": "서로를 아끼는 마음을 느끼고 행복하게 하루를 마무리한다.",
        "style": {
            "audience": "어린이",
            "tone": "따뜻하고 쉬운 문장",
            "length": "3~5문장",
        },
    }


def _coerce_str_list(value: Any, fallback: list[str]) -> list[str]:
    """Return a clean list of non-empty Korean strings."""
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = [str(item) for item in value if str(item).strip()]
    else:
        candidates = fallback
    result = [item.strip() for item in candidates if item.strip()]
    return result or fallback


def _to_korean_hint(value: Any) -> str:
    """Translate common English model labels into Korean story labels."""
    text = str(value).strip()
    key = text.lower()
    return PHRASE_KO.get(key) or CONCEPT_KO.get(key) or text


def _coerce_ko_list(value: Any, fallback: list[str]) -> list[str]:
    """Return a clean list with common English labels normalized to Korean."""
    result: list[str] = []
    for item in _coerce_str_list(value, fallback):
        normalized = _to_korean_hint(item)
        if _looks_mostly_english(normalized):
            continue
        if normalized in STOPWORDS:
            continue
        result.append(normalized)
    return result or fallback


def _looks_mostly_english(text: str) -> bool:
    """Detect plan text that should be replaced by the Korean fallback plan."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    ascii_letters = [char for char in letters if ord(char) < 128]
    return len(ascii_letters) / len(letters) > 0.6


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an EXAONE response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("EXAONE response did not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _build_structured_plan_prompt(vision: dict[str, Any]) -> str:
    """Build the shared EXAONE prompt for Korean structure and story planning."""
    return (
        "다음 vision_json은 손그림을 BLIP/OpenCLIP으로 분석한 영어 단서입니다.\n"
        "한국어 동화 생성을 위해 의미를 정리해 주세요.\n"
        "반드시 JSON 객체만 출력하세요. 설명, 마크다운, 코드블록은 쓰지 마세요.\n"
        "보이는 단서에 근거하되, 아이 손그림에서 자연스럽게 추론 가능한 달/별/밤/바구니 같은 요소는 "
        "raw_caption이나 objects에 없더라도 단서가 있으면 story_items에 반영해도 됩니다.\n\n"
        "출력 형식:\n"
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


def _merge_exaone_story_schema(
    payload: dict[str, Any],
    fallback_structured: dict[str, Any],
    fallback_plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize EXAONE JSON into the pipeline's structured/plan schema."""
    structured_raw = payload.get("structured_json", payload)
    plan_raw = payload.get("plan_json", {})
    if not isinstance(structured_raw, dict):
        structured_raw = {}
    if not isinstance(plan_raw, dict):
        plan_raw = {}

    structured = dict(fallback_structured)
    structured["characters"] = _coerce_ko_list(
        structured_raw.get("characters"),
        fallback_structured.get("characters", ["아이"]),
    )[:3]
    structured["place"] = _to_korean_hint(
        structured_raw.get("place") or fallback_structured.get("place") or "그림 속 마을"
    )
    structured["visible_items"] = _coerce_ko_list(
        structured_raw.get("visible_items"),
        fallback_structured.get("visible_items", ["그림"]),
    )[:8]
    structured["story_items"] = _coerce_ko_list(
        structured_raw.get("story_items"),
        fallback_structured.get("story_items", ["그림"]),
    )[:5]
    structured["mood"] = _to_korean_hint(
        structured_raw.get("mood") or fallback_structured.get("mood") or "포근한"
    )
    structured["theme"] = _to_korean_hint(
        structured_raw.get("theme") or fallback_structured.get("theme") or "함께 보내는 소중한 하루"
    )
    structured["main_event"] = _to_korean_hint(structured_raw.get("main_event", ""))
    if _looks_mostly_english(structured["main_event"]):
        structured["main_event"] = ""
    structured["source"] = "exaone"

    normalized_fallback_plan = build_plan_json(structured)
    plan = dict(normalized_fallback_plan)
    plan["title"] = str(plan_raw.get("title") or fallback_plan.get("title") or "작은 이야기").strip()
    plan["beginning"] = str(
        plan_raw.get("beginning") or normalized_fallback_plan.get("beginning") or ""
    ).strip()
    plan["middle"] = str(plan_raw.get("middle") or normalized_fallback_plan.get("middle") or "").strip()
    plan["ending"] = str(plan_raw.get("ending") or normalized_fallback_plan.get("ending") or "").strip()
    if any(_looks_mostly_english(str(plan.get(key, ""))) for key in ("title", "beginning", "middle", "ending")):
        plan = dict(normalized_fallback_plan)
    plan["style"] = plan_raw.get("style") if isinstance(plan_raw.get("style"), dict) else normalized_fallback_plan.get("style", {})
    plan["source"] = "exaone"
    return structured, plan


def generate_structured_plan_exaone(
    vision: dict[str, Any],
    max_new_tokens: int = 180,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use EXAONE once to translate vision JSON into Korean structured/plan JSON."""
    fallback_structured = build_structured_json(vision)
    fallback_plan = build_plan_json(fallback_structured)
    prompt = _build_structured_plan_prompt(vision)

    with timed_step(8, "EXAONE structured story planning"):
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
        structured, plan = _merge_exaone_story_schema(payload, fallback_structured, fallback_plan)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        structured = dict(fallback_structured)
        structured["source"] = "fallback_after_exaone_parse_error"
        plan = dict(fallback_plan)
        plan["source"] = "fallback_after_exaone_parse_error"
        raw_response = f"{raw_response}\n\n[parse_error] {exc}"
    return structured, plan, raw_response


def generate_structured_plan_exaone_gguf(
    vision: dict[str, Any],
    max_new_tokens: int = 180,
    model_path: str = "",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use EXAONE GGUF through llama.cpp for Korean structured/plan JSON."""
    fallback_structured = build_structured_json(vision)
    fallback_plan = build_plan_json(fallback_structured)
    marker = "__EXAONE_RESPONSE_START__"
    prompt = _build_structured_plan_prompt(vision)
    prompt = (
        "### 지시\n"
        f"{prompt}\n"
        f"{marker}\n"
    )

    with timed_step(8, "EXAONE GGUF structured story planning"):
        resolved_model_path = model_path or os.environ.get("EXAONE_GGUF_MODEL_PATH") or DEFAULT_EXAONE_GGUF_PATH
        llama_cli = os.environ.get("LLAMA_CLI_PATH", LLAMA_CLI_PATH)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as prompt_file:
            prompt_file.write(prompt)
            prompt_path = prompt_file.name
        try:
            completed = subprocess.run(
                [
                    llama_cli,
                    "-m",
                    resolved_model_path,
                    "-f",
                    prompt_path,
                    "-n",
                    str(max_new_tokens),
                    "-c",
                    "2048",
                    "--temp",
                    "0",
                    "--single-turn",
                    "--device",
                    "none",
                    "--log-disable",
                    "--no-warmup",
                    "--simple-io",
                    "--no-display-prompt",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        finally:
            os.unlink(prompt_path)
        raw_response = completed.stdout
        if marker in raw_response:
            raw_response = raw_response.rsplit(marker, 1)[-1]
        raw_response = raw_response.split("[ Prompt:", 1)[0]
        raw_response = raw_response.replace("Exiting...", "").strip()
        raw_response = re.sub(
            r"(?s)^.*?available commands:.*?\n\n",
            "",
            raw_response,
        )
        raw_response = raw_response.strip()

    try:
        payload = _extract_json_object(raw_response)
        structured, plan = _merge_exaone_story_schema(payload, fallback_structured, fallback_plan)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        structured = dict(fallback_structured)
        structured["source"] = "fallback_after_exaone_gguf_parse_error"
        plan = dict(fallback_plan)
        plan["source"] = "fallback_after_exaone_gguf_parse_error"
        raw_response = f"{raw_response}\n\n[parse_error] {exc}"
    return structured, plan, raw_response


def _run_exaone_gguf_prompt(
    prompt: str,
    max_new_tokens: int,
    model_path: str = "",
    timeout: int = 180,
    context_size: int = 4096,
) -> str:
    """Run an EXAONE GGUF prompt through the local llama.cpp CLI."""
    resolved_model_path = model_path or os.environ.get("EXAONE_GGUF_MODEL_PATH") or DEFAULT_EXAONE_GGUF_PATH
    llama_cli = os.environ.get("LLAMA_CLI_PATH", LLAMA_CLI_PATH)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as prompt_file:
        prompt_file.write(prompt)
        prompt_path = prompt_file.name
    try:
        completed = subprocess.run(
            [
                llama_cli,
                "-m",
                resolved_model_path,
                "-f",
                prompt_path,
                "-n",
                str(max_new_tokens),
                "-c",
                str(context_size),
                "--temp",
                "0.55",
                "--top-p",
                "0.9",
                "--single-turn",
                "--device",
                "none",
                "--log-disable",
                "--no-warmup",
                "--simple-io",
                "--no-display-prompt",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    finally:
        os.unlink(prompt_path)
    raw_response = completed.stdout.split("[ Prompt:", 1)[0]
    raw_response = raw_response.replace("Exiting...", "").strip()
    raw_response = re.sub(
        r"(?s)^.*?available commands:.*?\n\n",
        "",
        raw_response,
    )
    return raw_response.strip()


def _short_text(value: Any, limit: int = 120) -> str:
    """Keep model prompts compact by trimming verbose scene hints."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def generate_story_draft(structured: dict[str, Any], plan: dict[str, Any]) -> str:
    """Generate a deterministic Korean fairy-tale draft from a plan."""
    characters = _join_people(structured.get("characters", ["아이"]))
    place = structured.get("place", "그림 속 마을")
    mood = structured.get("mood", "포근한")
    items = structured.get("story_items") or structured.get("visible_items", ["그림"])
    item_text = _join_items(items[:4])
    main_event = str(structured.get("main_event", "")).strip()
    plan_middle = str(plan.get("middle", "")).strip()
    if main_event:
        item_sentence = f"그림 속에서는 {main_event}."
    elif "호랑이" in structured.get("characters", []):
        if any(item in {"밤하늘", "달", "별"} for item in items):
            sky_items = [item for item in items if item in {"밤하늘", "달", "별"}]
            item_sentence = (
                f"{_join_items(sky_items[:3])} 아래에서 처음에는 조금 놀랐지만, "
                "둘은 곧 서로를 바라보며 마음을 열었어요."
            )
        else:
            item_sentence = "처음에는 조금 놀랐지만, 둘은 곧 서로를 바라보며 마음을 열었어요."
    elif item_text == "집":
        item_sentence = "따뜻한 집은 모두를 반겨 주는 것처럼 환하게 서 있었어요."
    else:
        item_sentence = f"곁에는 {item_text}도 함께 있어 그림 속 세상이 더 환해 보였어요."
    if plan_middle and not plan_middle.endswith(("다.", "요.", "요", ".")):
        plan_middle = f"{plan_middle}."
    middle_sentence = (
        plan_middle.replace("나눈다.", "나누었어요.").replace("보낸다.", "보냈어요.")
        if plan_middle
        else f"{_as_topic(characters)} 오늘 본 것들을 함께 이야기하며 활짝 웃었답니다."
    )
    return (
        f"어느 {mood} 날, {_as_topic(characters)} {place}에 모였어요. "
        f"{item_sentence} "
        f"{middle_sentence} "
        "그날의 작은 그림은 모두에게 오래 기억될 따뜻한 이야기가 되었어요."
    )


def generate_sequence_story_draft(scene_records: list[dict[str, Any]]) -> str:
    """Generate one connected Korean story from ordered scene records."""
    if not scene_records:
        return ""

    paragraphs: list[str] = []
    for index, record in enumerate(scene_records, start=1):
        structured = record["structured_json"]
        characters = _join_people(structured.get("characters", ["아이"]))
        place = _to_korean_hint(structured.get("place", "그림 속 마을"))
        mood = _to_korean_hint(structured.get("mood", "포근한"))
        raw_items = structured.get("story_items") or structured.get("visible_items") or ["그림"]
        items = [_to_korean_hint(item) for item in raw_items[:4]]
        item_text = _join_items(items[:3])
        main_event = _to_korean_hint(str(structured.get("main_event", "")).strip())
        if main_event:
            event_sentence = f"그림 속에서는 {main_event}."
        elif "호랑이" in structured.get("characters", []):
            event_sentence = f"{_as_topic(characters)} {_as_object(item_text)} 보며 조금씩 마음을 열었어요."
        else:
            event_sentence = f"{_as_topic(characters)} {_as_object(item_text)} 발견하고 다음 일을 궁금해했어요."

        label = _scene_label(index)
        if index == 1:
            bridge = "이 작은 발견은 앞으로 이어질 모험의 첫 약속이 되었어요."
        elif index == len(scene_records):
            bridge = "마지막에는 모든 그림이 하나로 이어지며 다정한 마음을 남겼어요."
        else:
            bridge = "그래서 다음 그림으로 넘어갈 때마다 이야기는 조금 더 깊어졌어요."

        paragraph_sentences = [
            f"{label} {_as_topic(characters)} {place}에 있었어요.",
            f"{mood} 분위기 속에서 {item_text}도 함께 보여서 그림이 더 또렷하게 느껴졌어요.",
            event_sentence,
            "처음에는 무엇이 일어날지 몰라 조심스러웠지만, 모두는 서로를 바라보며 용기를 냈어요.",
            bridge,
        ]
        paragraphs.append(" ".join(paragraph_sentences))

    return "\n\n".join(paragraphs)


def _caption_line(record: dict[str, Any]) -> str:
    """Build one compact Korean-readable caption line from a scene record."""
    vision = record["vision"]
    structured = record["structured_json"]
    scene_index = record.get("scene_index", 0)
    raw_caption = str(vision.get("raw_caption", "")).strip()
    characters = _join_people(structured.get("characters", ["아이"]))
    place = _to_korean_hint(structured.get("place") or vision.get("scene") or "그림 속 장소")
    mood = _to_korean_hint(structured.get("mood") or vision.get("mood") or "알 수 없는")
    items = structured.get("story_items") or structured.get("visible_items") or vision.get("objects", [])
    item_text = _join_items([_to_korean_hint(item) for item in items[:3]])
    return (
        f"{scene_index}번: {characters} / 장소: {place} / 보이는 것: {item_text} / "
        f"분위기: {mood} / 원본 캡션: {raw_caption}"
    )


def build_image_captions(scene_records: list[dict[str, Any]]) -> str:
    """Return ordered image-caption text for the user's first prompt."""
    return "\n".join(_caption_line(record) for record in scene_records)


def build_story_structure_prompt(image_captions: str) -> str:
    """Fill the first user-provided prompt with image captions."""
    return f"""당신은 아이가 그린 여러 장의 그림을 보고 이야기 구조를 정리하는 보조 작가입니다.

아래 그림 설명 목록을 보고, 동화를 만들기 위한 이야기 흐름을 먼저 정리하세요.

[그림 설명 목록]
{image_captions}

[분석 조건]

1. 그림의 순서를 유지하세요.
2. 각 그림을 하나의 장면으로 정리하세요.
3. 각 장면의 분위기와 감정을 함께 적으세요.
4. 장면 사이가 자연스럽게 이어지도록 원인과 결과를 만들어 주세요.
5. 전체 이야기가 처음-중간-끝 구조를 가지도록 정리하세요.
6. 무조건 밝은 방향으로만 해석하지 말고, 그림의 분위기에 따라 쓸쓸함, 무서움, 신비로움, 외로움도 반영하세요.

[출력 형식]
전체 분위기:
주인공:
이야기의 핵심 감정 변화:

장면별 정리:
1번 장면:

* 그림 내용:
* 분위기:
* 감정:
* 다음 장면으로 이어지는 이유:

2번 장면:

* 그림 내용:
* 분위기:
* 감정:
* 다음 장면으로 이어지는 이유:

마지막 장면:

* 그림 내용:
* 분위기:
* 감정:
* 결말 방향:"""


def build_story_writing_prompt(story_structure: str) -> str:
    """Fill the second user-provided prompt with the generated story structure."""
    return f"""당신은 아이들이 그린 여러 장의 그림을 바탕으로 하나의 동화를 쓰는 작가입니다.

아래 이야기 구조를 바탕으로 자연스럽고 동화 같은 이야기를 작성하세요.

[이야기 구조]
{story_structure}

[작성 조건]

1. 장면 순서를 유지하세요.
2. 모든 장면이 하나의 이야기처럼 자연스럽게 이어지게 하세요.
3. 장면 전환은 인물의 행동, 감정, 사건의 결과로 이어지게 하세요.
4. 동화적인 표현을 사용하세요. 예: 살금살금, 반짝반짝, 조그만 마음, 구름 이불, 달님이 속삭였어요
5. 그림의 분위기가 밝으면 밝게, 어두우면 조심스럽게, 쓸쓸하면 잔잔하게, 이상하면 신비롭게 표현하세요.
6. 무조건 행복한 이야기로 만들지 말고, 그림에 맞는 감정과 결말을 사용하세요.
7. 아이들이 이해할 수 있는 쉬운 문장으로 작성하세요.
8. 하지만 단순한 설명문처럼 쓰지 말고, 동화처럼 장면이 떠오르게 작성하세요.
9. 주인공의 감정 변화가 분명히 드러나게 하세요.
10. 그림 개수가 적으면 짧게, 많으면 길게 작성하세요.
11. 그림 1장당 1~3문장 정도로 작성하세요.
12. 지금 출력은 너무 길지 않게 제목 1줄과 이야기 3줄 정도로만 작성하세요.

[출력 형식]
제목:

이야기:"""


def generate_prompt_story_structure(scene_records: list[dict[str, Any]]) -> str:
    """Generate a story structure in the shape requested by the first prompt."""
    if not scene_records:
        return ""

    first = scene_records[0]["structured_json"]
    last = scene_records[-1]["structured_json"]
    first_characters = _join_people(first.get("characters", ["아이"]))
    last_place = _to_korean_hint(last.get("place", "마지막 그림"))
    scene_lines = []
    for index, record in enumerate(scene_records, start=1):
        structured = record["structured_json"]
        characters = _join_people(structured.get("characters", ["아이"]))
        place = _to_korean_hint(structured.get("place", "그림 속 장소"))
        mood = _to_korean_hint(structured.get("mood", "알 수 없는"))
        items = structured.get("story_items") or structured.get("visible_items") or ["그림"]
        item_text = _join_items([_to_korean_hint(item) for item in items[:3]])
        if "호랑이" in structured.get("characters", []):
            feeling = "낯설지만 궁금한 마음"
        elif index == 1:
            feeling = "조심스러운 기대"
        elif index == len(scene_records):
            feeling = "잔잔한 안도감"
        else:
            feeling = "조금씩 커지는 호기심"
        if index == len(scene_records):
            bridge = f"{last_place}에서 지나온 장면들을 떠올리며 마음을 정리한다."
        else:
            bridge = "방금 본 것이 다음 그림의 길잡이가 되어 발걸음이 이어진다."
        scene_lines.append(
            f"{index}번 장면:\n\n"
            f"* 그림 내용: {characters}이/가 {place}에서 {item_text}을/를 본다.\n"
            f"* 분위기: {mood}\n"
            f"* 감정: {feeling}\n"
            f"* 다음 장면으로 이어지는 이유: {bridge}"
        )
    return (
        "전체 분위기: 따뜻함 속에 낯섦과 신비로움이 조금 섞인 모험\n"
        f"주인공: {first_characters}\n"
        "이야기의 핵심 감정 변화: 조심스러운 기대에서 낯선 만남을 지나 잔잔한 안도감으로 이동\n\n"
        "장면별 정리:\n"
        + "\n\n".join(scene_lines)
    )


def generate_short_story_from_structure(story_structure: str, scene_records: list[dict[str, Any]]) -> str:
    """Generate a short three-line fairy tale from the first prompt's structure."""
    if not scene_records:
        return "제목:\n\n이야기:"

    first = scene_records[0]["structured_json"]
    middle = scene_records[len(scene_records) // 2]["structured_json"]
    last = scene_records[-1]["structured_json"]
    characters = _join_people(first.get("characters", ["아이"]))
    first_place = _to_korean_hint(first.get("place", "그림 속 마을"))
    middle_items = middle.get("story_items") or middle.get("visible_items") or ["그림"]
    middle_item_text = _join_items([_to_korean_hint(item) for item in middle_items[:2]])
    last_place = _to_korean_hint(last.get("place", "하늘 아래"))
    return (
        "제목: 그림들이 속삭인 작은 길\n\n"
        "이야기:\n"
        f"{_as_topic(characters)} {first_place}에서 반짝반짝 빛나는 첫 그림을 따라 살금살금 걸어갔어요.\n"
        f"길 위에서 {_as_object(middle_item_text)} 만나자 조그만 마음은 조금 낯설고 신비롭게 두근거렸어요.\n"
        f"마지막에 {last_place}에 닿은 아이들은 모든 그림이 이어 준 마음을 조용히 품고 돌아섰어요."
    )


def generate_prompt_twostep_short_story(
    scene_records: list[dict[str, Any]],
) -> dict[str, str]:
    """Run the user's two-prompt flow with a compact deterministic writer."""
    image_captions = build_image_captions(scene_records)
    structure_prompt = build_story_structure_prompt(image_captions)
    story_structure = generate_prompt_story_structure(scene_records)
    writing_prompt = build_story_writing_prompt(story_structure)
    story_final = generate_short_story_from_structure(story_structure, scene_records)
    return {
        "image_captions": image_captions,
        "structure_prompt": structure_prompt,
        "story_structure": story_structure,
        "writing_prompt": writing_prompt,
        "story_final": story_final,
    }


def _count_story_sentences(story: str) -> int:
    """Count simple sentence endings in generated Korean story text."""
    return len([part for part in re.split(r"(?<=[.!?。])\s+", story.strip()) if part.strip()])


def generate_sequence_story_exaone_gguf(
    scene_records: list[dict[str, Any]],
    max_new_tokens: int = 420,
) -> tuple[str, str]:
    """Ask EXAONE GGUF to write one non-repetitive story from ordered scenes."""
    fallback_story = generate_sequence_story_draft(scene_records)
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
    with timed_step(12, "EXAONE GGUF sequence story writing"):
        raw_story = _run_exaone_gguf_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            timeout=240,
            context_size=8192,
        )
    if marker in raw_story:
        raw_story = raw_story.rsplit(marker, 1)[-1]
    story = raw_story.strip()
    if "(truncated)" in story:
        story = story.split("(truncated)", 1)[-1].strip()
    story_start = re.search(r"(어느\s+[^\n]+)", story)
    if story_start:
        story = story[story_start.start() :]
    story = re.sub(r"^>\s*", "", story).strip()
    story = re.sub(r"^동화 본문:\s*", "", story).strip()
    story = re.sub(r"^```(?:text)?\s*|\s*```$", "", story).strip()
    if (
        not story
        or _looks_mostly_english(story)
        or "exceeds the available context size" in story
        or story.startswith("Error:")
        or story.startswith("아래는 순서가 있는")
        or "ordered_scenes:" in story
        or _count_story_sentences(story) < len(scene_records) * 5
    ):
        return fallback_story, f"{raw_story}\n\n[fallback] invalid_context_or_too_short_output"
    return story, raw_story


def polish_story_ko_exaone(
    story_draft: str,
    structured: dict[str, Any],
    plan: dict[str, Any],
    max_new_tokens: int = 60,
) -> str:
    """Polish a Korean draft into a child-friendly story using one EXAONE call."""
    prompt = (
        "아래 초안을 어린이가 읽기 좋은 한국어 동화 문체로 다듬어 주세요.\n"
        "조건:\n"
        "- 초안의 사건과 등장 요소를 크게 바꾸지 않기\n"
        "- 3~5문장으로 자연스럽게 쓰기\n"
        "- 따뜻한 결말 유지하기\n"
        "- 동화 본문만 출력하기\n\n"
        f"구조화 정보: {structured}\n"
        f"이야기 기획: {plan}\n"
        f"초안: {story_draft}\n"
    )

    with timed_step(10, "EXAONE Korean story polishing"):
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
                do_sample=True,
                top_p=0.9,
                temperature=0.65,
                repetition_penalty=1.08,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        story = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return story or story_draft


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

    with timed_step(8, "GPT-2 English story generation"):
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
    with timed_step(9, "NLLB English-to-Korean translation"):
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


def generate_story_ko_exaone(vision: dict, max_new_tokens: int = 60) -> str:
    """Generate a Korean children's story directly from vision JSON using EXAONE."""
    objects = ", ".join(vision.get("objects", [])) or "없음"
    prompt = (
        "아래 시각 단서를 바탕으로 어린이가 읽기 좋은 한국어 동화를 써 주세요.\n"
        "조건:\n"
        "- 2~4문장으로 자연스럽게 쓰기\n"
        "- 그림에 보이는 대상과 분위기를 이야기 안에 반영하기\n"
        "- 무섭거나 폭력적인 내용 없이 따뜻한 결말로 마무리하기\n"
        "- 설명문이나 목록이 아니라 동화 본문만 출력하기\n\n"
        f"원본 캡션: {vision.get('raw_caption', '')}\n"
        f"등장/사물 후보: {objects}\n"
        f"주인공: {vision.get('who', '')}\n"
        f"행동: {vision.get('actions', '')}\n"
        f"장소: {vision.get('scene', '')}\n"
        f"분위기: {vision.get('mood', '')}\n"
    )

    with timed_step(8, "EXAONE Korean story generation"):
        import torch

        tokenizer, model = get_exaone_components()
        device = get_device()
        messages = [
            {"role": "user", "content": prompt},
        ]
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
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.1,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        story = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return story
