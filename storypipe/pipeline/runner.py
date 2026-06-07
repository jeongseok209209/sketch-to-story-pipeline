"""[담당 3 · 파이프라인] 실험 C~J 오케스트레이션 (비전+스토리 통합 + 출력 저장).

vision.qwen_scenes(장면 추출) + story.experiments(스토리 생성)을 묶어 실행하고 결과(JSON/TXT/HTML)를
저장한다. vision과 story를 모두 import하는 유일한 통합 지점이다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from storypipe.common.config import (
    INPUT_DIR,
    OUTPUT_ROOT,
    VISION_MODEL_ID,
)
from storypipe.common.config import experiment_dirs as _experiment_dirs
from storypipe.common.io import file_url as _file_url
from storypipe.common.io import html_escape as _html_escape
from storypipe.common.logging import log_stage, set_step_context
from storypipe.story.experiments import (
    build_experiment_c,
    build_experiment_d,
    build_experiment_e,
    build_experiment_f,
    build_experiment_g,
    build_experiment_h,
    build_experiment_i,
    build_experiment_j,
)
from storypipe.vision.qwen_scenes import (
    _read_story_caption,
    prepare_qwen_collage_for_experiment,
    prepare_qwen_scenes_for_experiment,
)

LLM_MODEL_NOTE = "EXAONE GGUF via llama.cpp"


# ─────────────────────────────────────────────────────────────────────────────
# 아래 본문은 기존 run_experiments_cd_qwen3b.py의 출력/오케스트레이션 구역에서 이동한 코드.
# ─────────────────────────────────────────────────────────────────────────────
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

