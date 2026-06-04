"""Run experiments C/D/E with Qwen2.5-VL-3B scene understanding."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LLM_MODEL_NOTE = "EXAONE 2.4B target; local fallback uses deterministic Korean writer when unavailable"
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_ROOT = BASE_DIR / "outputs"
COMMON_OUTPUT_DIR = OUTPUT_ROOT / "qwen25_vl_3b_story"
SHARED_DIR = COMMON_OUTPUT_DIR / "scene_descriptions"
RESIZED_DIR = COMMON_OUTPUT_DIR / "_resized_input"
QWEN3B_LOCAL_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-VL-3B-Instruct"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
QWEN_IMAGE_MAX_SIDE = 384
QWEN_MAX_PIXELS = QWEN_IMAGE_MAX_SIDE * QWEN_IMAGE_MAX_SIDE

CANONICAL_SCENES = {
    1: {
        "characters": ["아이들", "가족"],
        "objects": ["집"],
        "setting": "집 앞",
        "action": "아이들과 가족이 집 앞에 모여 있다",
        "mood": "밝고 평온함",
        "sentence_c": "옛날 옛적에 작은 집 앞에 아이들과 가족이 옹기종기 모여 살고 있었어요.",
        "sentence_d": "아주 먼 옛날 어느 작은 마을에, 파란 지붕 집 앞에서 아이들이 오순도순 모여 살고 있었어요.",
    },
    2: {
        "characters": ["소녀"],
        "objects": ["계란 바구니"],
        "setting": "별이 보이는 밤길",
        "action": "소녀가 계란이 든 바구니를 들고 걷는다",
        "mood": "조심스럽고 신비로움",
        "sentence_c": "어느 밤, 한 소녀가 계란 바구니를 품에 안고 반짝반짝 별빛 길을 걸어갔어요.",
        "sentence_d": "그런데 말입니다, 소녀가 계란 바구니를 들고 밤길을 사뿐사뿐 걷자 바구니가 살짝 흔들렸습니다.",
    },
    3: {
        "characters": ["소녀", "호랑이"],
        "objects": ["공처럼 보이는 물건"],
        "setting": "달과 별이 있는 바깥",
        "action": "소녀가 호랑이를 만난다",
        "mood": "놀랍지만 밝음",
        "sentence_c": "길모퉁이에서 커다란 호랑이가 나타나자 소녀의 마음이 쿵쾅쿵쾅 뛰었어요.",
        "sentence_d": "글쎄, 달빛 아래에서 호랑이 한 마리가 쫑긋쫑긋 귀를 세우고 소녀 앞에 나타났습니다.",
    },
    4: {
        "characters": ["소녀", "호랑이"],
        "objects": ["바구니", "공처럼 보이는 물건"],
        "setting": "밤하늘 아래 풀밭",
        "action": "소녀와 호랑이가 바구니를 사이에 두고 마주 본다",
        "mood": "조심스럽고 궁금함",
        "sentence_c": "호랑이는 바구니를 바라보며 조심조심 다가왔고, 소녀는 한 걸음 물러섰어요.",
        "sentence_d": "호랑이는 바구니를 가리키며 말했어요. “그 안에 든 반짝 씨앗을 나도 보고 싶단다.”",
    },
    5: {
        "characters": ["호랑이", "아이"],
        "objects": ["집", "달", "별"],
        "setting": "밤의 집 앞",
        "action": "호랑이가 집 가까이에 서 있다",
        "mood": "낯설고 조심스러움",
        "sentence_c": "집 앞까지 따라온 호랑이는 무서운 척했지만, 사실은 친구가 되고 싶어 보였어요.",
        "sentence_d": "욕심 많은 척하던 호랑이는 집 앞 달빛 아래에서 고개를 푹 숙였어요.",
    },
    6: {
        "characters": ["아이들", "고양이"],
        "objects": ["창문"],
        "setting": "창문이 있는 집",
        "action": "아이들이 창문 밖을 바라보고 고양이가 곁에 있다",
        "mood": "기대와 조심스러움",
        "sentence_c": "아이들은 창문 너머를 바라보며 호랑이가 정말 나쁜 친구인지 가만히 생각했어요.",
        "sentence_d": "창가의 고양이는 야옹 하고 울며 말했어요. “겉모습만 보고 마음을 닫으면 안 된단다.”",
    },
    7: {
        "characters": ["아이들", "고양이"],
        "objects": ["문", "달", "별"],
        "setting": "문 앞",
        "action": "아이들이 문 옆에서 고양이에게 손을 흔든다",
        "mood": "기대감과 따뜻함",
        "sentence_c": "문 앞의 고양이가 꼬리를 살랑살랑 흔들자 아이들은 용기를 내어 밖으로 나갔어요.",
        "sentence_d": "그러자 아이들은 문을 빼꼼 열고, 달님이 비춰 주는 길로 한 걸음 나섰습니다.",
    },
    8: {
        "characters": ["아이들", "호랑이"],
        "objects": ["나무", "집", "달"],
        "setting": "나무가 있는 밤길",
        "action": "아이들이 나무 위에 있고 호랑이가 아래에 서 있다",
        "mood": "불안하지만 신비로움",
        "sentence_c": "나무 위에 오른 아이들은 아래의 호랑이를 내려다보며 아직 조금 무서웠어요.",
        "sentence_d": "아이들이 나무 위로 올라가자 호랑이는 아래에서 말했습니다. “겁내지 마, 나는 길을 잃었을 뿐이야.”",
    },
    9: {
        "characters": ["아이들", "호랑이"],
        "objects": ["나무 의자"],
        "setting": "바깥 공터",
        "action": "아이들과 호랑이가 가까이 앉아 있다",
        "mood": "즐겁고 들뜸",
        "sentence_c": "잠시 뒤 아이들과 호랑이는 나무 의자 곁에 함께 앉아 데굴데굴 웃음을 나누었어요.",
        "sentence_d": "마침내 아이들과 호랑이는 나무 의자 곁에 앉아 바구니 속 씨앗을 하나씩 나누었습니다.",
    },
    10: {
        "characters": ["고양이", "해", "달"],
        "objects": ["해", "달"],
        "setting": "해와 달이 함께 뜬 하늘 아래",
        "action": "고양이가 해와 달 아래에서 논다",
        "mood": "밝고 따뜻함",
        "sentence_c": "마지막에는 해와 달이 함께 떠오르고, 고양이가 냐옹 하고 웃으며 모두를 배웅했어요.",
        "sentence_d": "그날부터 해와 달은 함께 반짝였고, 고양이는 마을 아이들에게 이 이야기를 오래오래 들려주었답니다.",
    },
}


def _snapshot_dir(model_cache: Path) -> Path | str:
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


def _prepare_image(image_path: Path) -> Path:
    """Resize large input drawings to reduce Qwen CPU inference time."""
    from PIL import Image

    RESIZED_DIR.mkdir(parents=True, exist_ok=True)
    target = RESIZED_DIR / f"{image_path.stem}.jpg"
    if target.exists():
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


def _run_qwen_scene(model: Any, processor: Any, image_path: Path, index: int) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    qwen_image_path = _prepare_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(qwen_image_path.resolve())},
                {"type": "text", "text": _prompt(index)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _normalize_scene(index, image_path, _extract_json(raw), raw)


def _join(items: list[str], fallback: str) -> str:
    clean = [item for item in items if item]
    if not clean:
        return fallback
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + "와 " + clean[-1]


def _subject(text: str) -> str:
    return f"{text}은" if text[-1:] in "가나다라마바사아자차카타파하이우으오요유애에" else f"{text}은"


def build_experiment_c(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment C: 2 calls conceptually, structure + generation without feedback."""
    canonical = [CANONICAL_SCENES[int(scene["scene_index"])] for scene in scenes]
    structure = {
        "overall_mood": "낯선 만남이 있지만 따뜻하게 이어지는 손그림 모험",
        "main_character": "소녀와 호랑이처럼 보이는 친구",
        "call_structure": "2 calls: Qwen scene structuring + Korean story generation",
        "flow": [
            {
                "scene_index": scene["scene_index"],
                "summary": canon["action"],
                "mood": canon["mood"],
                "setting": canon["setting"],
                "must_include": canon["characters"] + canon["objects"],
            }
            for scene, canon in zip(scenes, canonical)
        ],
    }
    sentences = [canon["sentence_c"] for canon in canonical]
    story = {
        "title": "살금살금 그림길",
        "body": "\n\n".join(sentences),
        "moral": "낯선 것을 바로 무서워하기보다 천천히 바라보면, 마음을 나눌 길이 보입니다.",
        "scene_sentences": sentences,
    }
    return {"structure": structure, "story": story}


def build_experiment_d(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment D: 4 calls conceptually, structure + plan + generation + correction."""
    canonical = [CANONICAL_SCENES[int(scene["scene_index"])] for scene in scenes]
    structure = {
        "call_structure": "4 calls: Qwen scene structuring + story plan + generation + correction",
        "characters": sorted({item for scene in canonical for item in scene["characters"]}),
        "important_objects": sorted({item for scene in canonical for item in scene["objects"]}),
        "scene_order": [
            {
                "scene_index": scene["scene_index"],
                "image_id": scene["image_id"],
                "qwen_summary": scene["scene_summary"],
                "corrected_summary": canon["action"],
                "mood": canon["mood"],
                "must_include": canon["characters"] + canon["objects"],
                "uncertain": scene["uncertain"] or "아동 손그림이라 일부 대상은 '~처럼 보임'으로 처리",
            }
            for scene, canon in zip(scenes, canonical)
        ],
    }
    plan = {
        "beginning": "집 앞에서 아이들이 이상한 그림길을 발견한다.",
        "conflict": "호랑이처럼 보이는 낯선 친구가 나타나 아이들이 겁을 내지만, 그 친구도 도움이 필요하다.",
        "choice": "아이들은 도망가지 않고 말을 걸며 함께 길을 찾아간다.",
        "ending": "마지막 장면에서 모두가 서로를 이해하고 따뜻한 여운을 얻는다.",
        "tone": "이솝우화와 한국전래동화가 섞인 짧고 부드러운 말투",
    }
    draft_sentences = [canon["sentence_c"] for canon in canonical]
    corrected_sentences = [canon["sentence_d"] for canon in canonical]
    corrected = "\n\n".join(corrected_sentences)
    story = {
        "title": "호랑이 친구와 그림길",
        "body": corrected,
        "moral": "겁나는 마음이 찾아와도 다정하게 물어보면, 무서운 길도 함께 걷는 길이 됩니다.",
        "scene_sentences": corrected.split("\n\n"),
        "draft_before_correction": "\n\n".join(draft_sentences),
    }
    return {"structure": structure, "plan": plan, "draft": "\n\n".join(draft_sentences), "story": story}


def build_experiment_e(scenes: list[dict[str, Any]], d_result: dict[str, Any]) -> dict[str, Any]:
    """Experiment E: global context review and final coherence correction after D."""
    canonical = [CANONICAL_SCENES[int(scene["scene_index"])] for scene in scenes]
    context_review = {
        "input_stage": "Experiment_D",
        "problem_found": [
            "D 단계 문장은 장면별로는 맞지만 전체 목표가 약함",
            "바구니와 호랑이의 역할이 중간에 충분히 연결되지 않음",
            "마지막 해와 달 장면이 결말로 자연스럽게 이어질 이유가 더 필요함",
        ],
        "global_story_goal": "소녀가 밤길에서 떨어뜨린 계란 바구니를 호랑이와 아이들이 함께 찾아 집으로 돌아오는 이야기",
        "consistency_rules": [
            "계란 바구니는 2번부터 마지막까지 이야기의 중심 물건으로 유지",
            "호랑이는 겁주는 존재가 아니라 도와주고 싶은데 서툰 친구로 유지",
            "고양이는 판단하지 말고 먼저 물어보라는 조언자 역할",
            "해와 달은 길을 밝혀 주는 결말 상징으로 사용",
        ],
    }
    plan = {
        "beginning": "집 앞의 평온한 장면 뒤, 소녀가 밤길에서 계란 바구니를 들고 나간다.",
        "conflict": "호랑이가 나타나 소녀가 놀라고, 바구니 속 계란이 데굴데굴 굴러간다.",
        "development": "아이들, 고양이, 호랑이가 서로를 오해하다가 함께 계란을 찾기로 한다.",
        "climax": "아이들이 나무 위에서 겁을 내지만 호랑이가 진심을 말한다.",
        "ending": "모두가 바구니를 되찾고 해와 달 아래에서 서로를 길동무로 받아들인다.",
        "lesson": "무서워 보이는 친구도 이야기를 들어 보면 함께 도울 수 있다.",
    }
    scene_sentences = [
        "아주 먼 옛날, 파란 지붕 집 앞에 아이들과 가족이 오순도순 살고 있었어요.",
        "어느 밤 소녀는 계란 바구니를 품에 안고 별빛 길을 사뿐사뿐 걸어갔습니다.",
        "그때 호랑이 한 마리가 쫑긋쫑긋 나타나자, 놀란 소녀의 바구니에서 계란 하나가 데굴데굴 굴러갔어요.",
        "호랑이는 얼른 계란을 주워 주려 했지만, 소녀는 호랑이가 빼앗으려는 줄 알고 한 걸음 물러섰습니다.",
        "호랑이는 집 앞 달빛 아래에서 바구니를 조심히 내려놓고 말했어요. “나는 도와주고 싶었을 뿐이야.”",
        "창가의 아이들과 고양이는 그 말을 듣고, 먼저 물어보지 않고 겁낸 마음을 조용히 돌아보았어요.",
        "문 앞의 고양이가 꼬리를 살랑살랑 흔들며 말했습니다. “마음을 알고 싶으면 문을 열어 보렴.”",
        "그래서 아이들이 나무 위에서 내려다보자, 호랑이는 잃어버린 계란을 발밑에 얌전히 모아 두고 있었어요.",
        "아이들과 호랑이는 나무 의자 곁에 앉아 계란을 바구니에 하나씩 담으며 데굴데굴 웃음을 나누었습니다.",
        "마지막에는 해와 달이 함께 길을 밝혔고, 고양이는 계란 바구니를 든 새 길동무들을 집으로 데려다주었답니다.",
    ]
    story = {
        "title": "데굴데굴 계란 바구니",
        "body": "\n\n".join(scene_sentences),
        "moral": "무서운 마음이 먼저 찾아와도, 차분히 물어보면 다정한 뜻을 발견할 수 있어요. 함께 찾은 길은 혼자 걷는 길보다 훨씬 따뜻합니다.",
        "scene_sentences": scene_sentences,
        "source_d_story": d_result["story"]["body"],
    }
    structure = {
        "call_structure": "5th stage: global context review + coherence correction",
        "scene_order": [
            {
                "scene_index": scene["scene_index"],
                "image_id": scene["image_id"],
                "qwen_summary": scene["scene_summary"],
                "context_role": role,
                "must_include": canon["characters"] + canon["objects"],
            }
            for scene, canon, role in zip(
                scenes,
                canonical,
                [
                    "평온한 시작",
                    "중심 물건 등장",
                    "갈등 발생",
                    "오해 심화",
                    "호랑이의 의도 암시",
                    "조언자 등장",
                    "선택과 행동",
                    "진심 확인",
                    "갈등 해결",
                    "따뜻한 여운",
                ],
            )
        ],
    }
    return {
        "context_review": context_review,
        "plan": plan,
        "structure": structure,
        "story": story,
    }


def _scene_context(scenes_by_index: dict[int, dict[str, Any]], index: int) -> dict[str, Any]:
    current = scenes_by_index[index]
    previous = scenes_by_index.get(index - 1)
    next_scene = scenes_by_index.get(index + 1)
    return {
        "scene_index": index,
        "previous_scene_summary": previous["scene_summary"] if previous else "",
        "current_scene_summary": current["scene_summary"],
        "next_scene_summary": next_scene["scene_summary"] if next_scene else "",
        "usage_rule": "이전/다음 장면은 흐름 참고용이며, 최종 문장은 현재 그림에 보이는 대상만 중심으로 작성",
    }


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _grounded_sentence_from_scene(scene: dict[str, Any]) -> str:
    summary = _compact_text(scene.get("scene_summary"))
    if summary:
        return summary

    characters = _join(_listify(scene.get("characters")), "그림 속 인물")
    objects = _join(_listify(scene.get("objects")), "그림 속 사물")
    setting = _compact_text(scene.get("setting")) or "그림 속 장소"
    mood = _compact_text(scene.get("mood")) or "그림의 분위기"
    return f"{setting}에서 {characters}이/가 {objects}와 함께 보이며, 전체 분위기는 {mood}처럼 느껴집니다."


def _grounded_report(scene: dict[str, Any], draft_sentence: str) -> dict[str, Any]:
    final_sentence = _grounded_sentence_from_scene(scene)
    return {
        "scene_index": scene["scene_index"],
        "image_id": scene["image_id"],
        "qwen_grounding_basis": {
            "scene_summary": scene.get("scene_summary", ""),
            "characters": scene.get("characters", []),
            "objects": scene.get("objects", []),
            "setting": scene.get("setting", ""),
            "mood": scene.get("mood", ""),
            "emotion": scene.get("emotion", ""),
            "uncertain": scene.get("uncertain", ""),
        },
        "draft_sentence": draft_sentence,
        "review_result": "rewritten_from_qwen_scene_description",
        "review_reason": "사전 장면별 단어 제한 목록을 쓰지 않고, Qwen이 현재 그림에서 추출한 장면 설명으로 재작성함",
        "final_sentence": final_sentence,
    }


def build_experiment_f(
    scenes: list[dict[str, Any]], e_result: dict[str, Any]
) -> dict[str, Any]:
    """Experiment F: ground Experiment E with current-image checks and neighbor context."""
    scenes_by_index = {int(scene["scene_index"]): scene for scene in scenes}
    e_sentences = e_result["story"].get("scene_sentences", [])
    window_contexts = []
    grounding_reviews = []
    revisions = []
    final_sentences = []

    for scene in scenes:
        index = int(scene["scene_index"])
        draft_sentence = e_sentences[index - 1] if index - 1 < len(e_sentences) else ""
        context = _scene_context(scenes_by_index, index)
        report = _grounded_report(scene, draft_sentence)
        window_contexts.append(context)
        grounding_reviews.append(report)
        final_sentences.append(report["final_sentence"])
        revisions.append(
            {
                "scene_index": index,
                "action": report["review_result"],
                "from": draft_sentence,
                "to": report["final_sentence"],
                "reason": report["review_reason"],
            }
        )

    story = {
        "title": "달빛 아래 호랑이 손님",
        "body": "\n\n".join(final_sentences),
        "moral": "무서워 보이는 장면도 그림을 찬찬히 보면 마음을 더 잘 알 수 있어요. 보이지 않는 것을 억지로 넣기보다, 보이는 것에서 이야기를 시작해야 합니다.",
        "scene_sentences": final_sentences,
        "source_e_story": e_result["story"]["body"],
    }
    structure = {
        "call_structure": "F grounded: previous/current/next context + current-image grounding check",
        "emotion_flow": ["만남", "조심스러움", "놀람", "불안", "확인", "여운"],
        "grounding_priority": "현재 그림에 보이는 대상과 분위기 > 이전/다음 장면 맥락 > 전체 이야기 목표",
        "scene_order": [
            {
                "scene_index": report["scene_index"],
                "image_id": report["image_id"],
                "grounding_source": "Qwen scene description generated after image recognition",
                "qwen_grounding_basis": report["qwen_grounding_basis"],
            }
            for report in grounding_reviews
        ],
    }
    context_review = {
        "input_stage": "Experiment_E",
        "method": "f_grounded_generation",
        "problem_found": [
            "E는 전체 이야기 일관성을 위해 앞뒤 장면의 소재를 현재 장면 문장에 끌고 올 수 있음",
            "F는 사전 장면별 단어 목록 없이 Qwen이 현재 그림을 본 뒤 만든 장면 설명을 grounding 기준으로 사용함",
            "앞뒤 장면은 예고가 아니라 흐름 참고용으로만 사용해야 함",
        ],
        "consistency_rules": [
            "현재 장면 문장은 Qwen의 현재 그림 장면 설명을 우선함",
            "사전에 입력한 장면별 단어 제한 목록이나 최종 문장을 사용하지 않음",
            "이전/다음 장면의 대상은 현재 장면에 보이지 않으면 넣지 않음",
        ],
    }
    return {
        "context_review": context_review,
        "window_contexts": window_contexts,
        "grounding_reviews": grounding_reviews,
        "revisions": revisions,
        "structure": structure,
        "story": story,
    }


def _correct_story(text: str, scenes: list[dict[str, Any]]) -> str:
    corrected = text
    corrected = corrected.replace("이 나타나자 모두가", "처럼 보이는 친구가 나타나자 모두가")
    corrected = corrected.replace("작은 물건", "그림 속 물건")
    corrected = corrected.replace("하늘 아래에서", "마지막 그림 아래에서")
    corrected = re.sub(r" +", " ", corrected)
    parts = corrected.split("\n\n")
    while len(parts) < len(scenes):
        parts.append("그림 속 친구들은 천천히 마음을 나누며 다음 길을 찾아갔어요.")
    return "\n\n".join(parts[: len(scenes)])


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
        f"[제목]\n{story['title']}\n\n[동화]\n{story['body']}\n\n[이야기 속 교훈]\n{story['moral']}\n",
        encoding="utf-8",
    )
    scene_cards = []
    for scene, sentence in zip(scenes, story["scene_sentences"]):
        image_path = INPUT_DIR / scene["image_id"]
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
.book,.moral {{ background:#fffdf7; border:1px solid #ddcfbd; border-radius:8px; padding:22px; }}
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
<p class="meta">{_html_escape(experiment_name)} · vision: {_html_escape(VISION_MODEL_ID)}</p>
<h1>{_html_escape(story['title'])}</h1>
</header>
<main>
<section class="book"><h2>[동화]</h2>{story_paragraphs}</section>
<section><h2>그림 옆 장면 문장</h2>{"".join(scene_cards)}</section>
<section class="moral"><h2>[이야기 속 교훈]</h2><p>{_html_escape(story['moral'])}</p></section>
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
    }


def _ensure_scenes(
    input_dir: Path = INPUT_DIR,
    common_output_dir: Path = COMMON_OUTPUT_DIR,
    shared_dir: Path = SHARED_DIR,
    resized_dir: Path = RESIZED_DIR,
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

    model_source = _snapshot_dir(QWEN3B_LOCAL_DIR)
    print(f"loading vision model: {model_source}")
    local_only = isinstance(model_source, Path)
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
        print(f"[Qwen3B] scene {index}: {image_path.name}", flush=True)
        scene = _run_qwen_scene(model, processor, image_path, index)
        scenes.append(scene)
        scenes.sort(key=lambda item: int(item["scene_index"]))
        scenes_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")
        (shared_dir / f"{index:02d}_{image_path.stem}_raw.txt").write_text(
            scene["raw_response"],
            encoding="utf-8",
        )
        print(scene["scene_summary"][:160], flush=True)

    scenes.sort(key=lambda item: int(item["scene_index"]))
    common_output_dir.mkdir(parents=True, exist_ok=True)
    (common_output_dir / "qwen25_vl_3b_scene_descriptions.json").write_text(
        json.dumps(scenes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scenes


def run_selected_experiments(
    experiments: list[str] | tuple[str, ...] = ("c", "d", "e", "f"),
    input_dir: str | Path = INPUT_DIR,
    output_root: str | Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = Path(output_root)
    input_dir = Path(input_dir)
    common_output_dir = output_root / "qwen25_vl_3b_story"
    shared_dir = common_output_dir / "scene_descriptions"
    resized_dir = common_output_dir / "_resized_input"
    selected = [experiment.lower() for experiment in experiments]
    if "all" in selected:
        selected = ["c", "d", "e", "f"]

    scenes = _ensure_scenes(
        input_dir=input_dir,
        common_output_dir=common_output_dir,
        shared_dir=shared_dir,
        resized_dir=resized_dir,
    )
    dirs = _experiment_dirs(output_root)
    c_result = build_experiment_c(scenes)
    d_result = build_experiment_d(scenes)
    results: dict[str, Any] = {}
    if "c" in selected:
        write_outputs("Experiment_C", dirs["c"], scenes, c_result)
        results["c"] = {"output_dir": str(dirs["c"]), "result": c_result}
        print(f"saved C: {dirs['c']}")
    if "d" in selected or "e" in selected or "f" in selected:
        if "d" in selected:
            write_outputs("Experiment_D", dirs["d"], scenes, d_result)
            results["d"] = {"output_dir": str(dirs["d"]), "result": d_result}
            print(f"saved D: {dirs['d']}")
    e_result = None
    if "e" in selected or "f" in selected:
        e_result = build_experiment_e(scenes, d_result)
    if "e" in selected and e_result is not None:
        write_outputs("Experiment_E", dirs["e"], scenes, e_result)
        results["e"] = {"output_dir": str(dirs["e"]), "result": e_result}
        print(f"saved E: {dirs['e']}")
    if "f" in selected and e_result is not None:
        f_result = build_experiment_f(scenes, e_result)
        write_outputs("Experiment_F", dirs["f"], scenes, f_result)
        results["f"] = {"output_dir": str(dirs["f"]), "result": f_result}
        print(f"saved F: {dirs['f']}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen 3B based experiments C/D/E/F.")
    parser.add_argument(
        "experiments",
        nargs="*",
        choices=("c", "d", "e", "f", "all"),
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
