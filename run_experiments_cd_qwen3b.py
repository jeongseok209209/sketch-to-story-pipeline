"""Run independent C/D/E/F experiments with Qwen vision and EXAONE GGUF writing."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generators import _run_exaone_gguf_prompt
from utils import DEFAULT_EXAONE_GGUF_PATH, LLAMA_CLI_PATH


VISION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LLM_MODEL_NOTE = "EXAONE GGUF via llama.cpp"
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


def _extract_required_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("EXAONE response did not contain a JSON object.")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("EXAONE JSON response was not an object.")
    return value


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
    model_path = Path(os.environ.get("EXAONE_GGUF_MODEL_PATH") or DEFAULT_EXAONE_GGUF_PATH).expanduser()
    llama_cli = Path(os.environ.get("LLAMA_CLI_PATH") or LLAMA_CLI_PATH).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            "EXAONE GGUF model file not found. Set EXAONE_GGUF_MODEL_PATH "
            f"or place the model at: {model_path}"
        )
    if not llama_cli.exists():
        raise FileNotFoundError(
            "llama.cpp CLI not found. Set LLAMA_CLI_PATH or build it at: "
            f"{llama_cli}"
        )


def _story_from_payload(payload: dict[str, Any], scene_count: int) -> dict[str, Any]:
    story = payload.get("story") if isinstance(payload.get("story"), dict) else payload
    title = str(story.get("title") or "그림 속 작은 이야기").strip()
    body = str(story.get("body") or "").strip()
    scene_sentences = story.get("scene_sentences")
    if not isinstance(scene_sentences, list):
        scene_sentences = []
    scene_sentences = [str(sentence).strip() for sentence in scene_sentences if str(sentence).strip()]
    if not scene_sentences and body:
        scene_sentences = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if not body and scene_sentences:
        body = "\n\n".join(scene_sentences)
    if len(scene_sentences) != scene_count:
        raise ValueError(
            f"EXAONE returned {len(scene_sentences)} scene_sentences for {scene_count} scenes."
        )
    moral = str(story.get("moral") or "").strip()
    grounding_notes = story.get("grounding_notes")
    return {
        "title": title,
        "body": body,
        "scene_sentences": scene_sentences,
        "moral": moral,
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
    raw_response = _run_exaone_gguf_prompt(
        prompt,
        max_new_tokens=max_new_tokens,
        timeout=300,
        context_size=context_size,
    )
    payload = _extract_required_json(raw_response)
    story = _story_from_payload(payload, len(scenes))
    return {
        "prompt_strategy": prompt_strategy,
        "exaone_prompt": prompt,
        "exaone_raw_response": raw_response,
        "parsed_result": payload,
        "fallback_used": False,
        "story": story,
        "structure": payload.get("structure", {}),
        "plan": payload.get("plan", {}),
        "experiment_method": experiment_name,
    }


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
        '    "moral": "이야기 속 교훈",\n'
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
        "실험 E: 전체 장면을 처음부터 하나의 이야기 맥락으로 강하게 연결하세요.\n"
        "전략: 장면별 묘사보다 전체 감정 흐름, 원인-결과 연결, 반복되는 등장 요소의 일관성을 먼저 설계한 뒤 씁니다.\n"
        "다만 현재 그림에 없는 내용을 과하게 꾸며 넣지 말고 Qwen scene 단서를 우선하세요.\n"
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
    """Experiment E: independent global coherence-first prompting."""
    return _run_exaone_experiment(
        "Experiment_E",
        "global_context_emotion_flow_first",
        _prompt_e(scenes),
        scenes,
    )


def build_experiment_f(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Experiment F: independent scene-window prompting with previous/current/next context."""
    result = _run_exaone_experiment(
        "Experiment_F",
        "global_overview_then_previous_current_next_scene_windows",
        _prompt_f(scenes),
        scenes,
        max_new_tokens=2600,
    )
    result["scene_generation_windows"] = _scene_windows(scenes)
    return result


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
<p class="meta">{_html_escape(experiment_name)} · vision: {_html_escape(VISION_MODEL_ID)} · llm: {_html_escape(LLM_MODEL_NOTE)}</p>
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
    builders = {
        "c": ("Experiment_C", build_experiment_c),
        "d": ("Experiment_D", build_experiment_d),
        "e": ("Experiment_E", build_experiment_e),
        "f": ("Experiment_F", build_experiment_f),
    }
    results: dict[str, Any] = {}
    for key in selected:
        experiment_name, builder = builders[key]
        result = builder(scenes)
        write_outputs(experiment_name, dirs[key], scenes, result)
        results[key] = {"output_dir": str(dirs[key]), "result": result}
        print(f"saved {key.upper()}: {dirs[key]}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent Qwen + EXAONE GGUF experiments C/D/E/F.")
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
