"""Blind randomized human evaluation dashboard for generated stories."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "outputs"
EVALUATION_DIR = OUTPUT_ROOT / "evaluations"
MAPPING_FILE = EVALUATION_DIR / "blind_mapping.json"
RECORDS_FILE = EVALUATION_DIR / "evaluation_records.jsonl"
SUMMARY_FILE = EVALUATION_DIR / "evaluation_summary.json"
INPUT_DIR = BASE_DIR / "inputs"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
EXPERIMENTS = ("A", "B", "C", "D", "E", "F")

QUANTITATIVE_METRICS = {
    "visual_groundedness": "그림 근거 충실도",
    "object_action_accuracy": "핵심 대상/행동 정확도",
    "low_hallucination": "환각 적음",
    "emotion_tone_alignment": "감정/분위기 반영도",
    "scene_linguistic_quality": "장면 문장 언어 품질",
}

QUALITATIVE_OPTIONS = {
    "story_coherence": {
        "label": "전개 구조",
        "options": ("처음-중간-끝 명확", "부분적으로 명확", "구조 약함"),
    },
    "logical_flow": {
        "label": "논리 일관성",
        "options": ("자연스럽게 이어짐", "일부 어색함", "흐름이 끊김"),
    },
    "scene_connection": {
        "label": "장면 연결 방식",
        "options": ("원인-결과 중심", "느슨한 연결", "나열식", "불연속적"),
    },
    "groundedness_pattern": {
        "label": "사실 기반성/그림 기반성",
        "options": ("대부분 그림 기반", "일부 과한 상상", "그림과 무관한 내용 많음"),
    },
    "fairy_tale_suitability": {
        "label": "동화 적합성",
        "options": ("적합", "부분적으로 적합", "부적합"),
    },
    "main_failure_type": {
        "label": "주요 실패 유형",
        "options": (
            "문제 없음",
            "반복적 전개",
            "갑작스러운 결말",
            "과한 상상",
            "장면 간 모순",
            "감정 흐름 약함",
            "언어 품질 문제",
        ),
    },
}


@dataclass(frozen=True)
class ResultFile:
    experiment: str
    path: Path


@dataclass(frozen=True)
class SceneView:
    scene_index: int
    image_id: str
    image_path: Path | None
    scene_summary: str
    generated_sentence: str


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    experiment: str
    result_file: Path
    scenes: list[SceneView]
    full_story: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_result_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _resolve_image(image_id: str, scene_payload: dict[str, Any] | None = None) -> Path | None:
    if scene_payload:
        image_path = scene_payload.get("image_path")
        if image_path:
            candidate = Path(str(image_path))
            if candidate.exists():
                return candidate

    if image_id:
        candidate = INPUT_DIR / image_id
        if candidate.exists():
            return candidate
        stem = Path(image_id).stem
        for extension in IMAGE_EXTENSIONS:
            candidate = INPUT_DIR / f"{stem}{extension}"
            if candidate.exists():
                return candidate

    return None


def _split_story_units(story: str) -> list[str]:
    clean = story.strip()
    if not clean:
        return []
    clean = clean.replace("\r\n", "\n")
    clean = clean.replace("[동화]", "").replace("[제목]", "").strip()
    if "이야기:" in clean:
        clean = clean.split("이야기:", 1)[-1].strip()
    parts = [part.strip() for part in clean.split("\n\n") if part.strip()]
    if len(parts) > 1:
        return parts
    return [part.strip() for part in clean.split("\n") if part.strip()]


def _compose_a_scene_summary(vision: dict[str, Any]) -> str:
    pieces = []
    raw_caption = str(vision.get("raw_caption", "")).strip()
    who = str(vision.get("who", "")).strip()
    actions = str(vision.get("actions", "")).strip()
    scene = str(vision.get("scene", "")).strip()
    mood = str(vision.get("mood", "")).strip()
    if raw_caption:
        pieces.append(raw_caption)
    if who:
        pieces.append(f"주인공: {who}")
    if actions:
        pieces.append(f"행동: {actions}")
    if scene:
        pieces.append(f"장소: {scene}")
    if mood:
        pieces.append(f"분위기: {mood}")
    return " / ".join(pieces) or "장면 설명 없음"


def _compose_sequence_scene_summary(scene: dict[str, Any]) -> str:
    structured = scene.get("structured_json") or {}
    vision = scene.get("vision") or {}
    parts = []
    caption = str(vision.get("raw_caption", "")).strip()
    characters = structured.get("characters") or []
    items = structured.get("story_items") or structured.get("visible_items") or []
    place = str(structured.get("place") or vision.get("scene") or "").strip()
    mood = str(structured.get("mood") or vision.get("mood") or "").strip()
    if caption:
        parts.append(caption)
    if characters:
        parts.append("등장: " + ", ".join(str(item) for item in characters))
    if items:
        parts.append("단서: " + ", ".join(str(item) for item in items[:4]))
    if place:
        parts.append(f"장소: {place}")
    if mood:
        parts.append(f"분위기: {mood}")
    return " / ".join(parts) or "장면 설명 없음"


def _story_from_cd_result(record: dict[str, Any]) -> tuple[str, list[str]]:
    story = record.get("story") or {}
    title = str(story.get("title", "")).strip()
    body = str(story.get("body", "")).strip()
    full_story = f"제목: {title}\n\n{body}".strip() if title else body
    sentences = story.get("scene_sentences") or []
    if isinstance(sentences, list):
        scene_sentences = [str(sentence).strip() for sentence in sentences]
    else:
        scene_sentences = []
    if not scene_sentences:
        scene_sentences = _split_story_units(body)
    return full_story, scene_sentences


def _case_from_record(case_id: str, result_file: Path, experiment: str) -> EvaluationCase:
    record = _read_json(result_file)

    if experiment == "A":
        image_id = str(record.get("image_id", "")).strip()
        vision = record.get("vision") or {}
        full_story = str(record.get("story_final", "")).strip()
        scene_payload = (record.get("steps") or {}).get("01_image_input") or {}
        scenes = [
            SceneView(
                scene_index=1,
                image_id=image_id,
                image_path=_resolve_image(image_id, scene_payload),
                scene_summary=_compose_a_scene_summary(vision),
                generated_sentence=full_story,
            )
        ]
        return EvaluationCase(case_id, experiment, result_file, scenes, full_story)

    if experiment == "B":
        full_story = str(record.get("story_final", "")).strip()
        story_units = _split_story_units(full_story)
        scenes_payload = record.get("scenes") or []
        scenes = []
        for index, scene in enumerate(scenes_payload, start=1):
            image_id = str(scene.get("image_id", "")).strip()
            scenes.append(
                SceneView(
                    scene_index=int(scene.get("scene_index") or index),
                    image_id=image_id,
                    image_path=_resolve_image(image_id),
                    scene_summary=_compose_sequence_scene_summary(scene),
                    generated_sentence=story_units[index - 1] if index - 1 < len(story_units) else "",
                )
            )
        return EvaluationCase(case_id, experiment, result_file, scenes, full_story)

    full_story, scene_sentences = _story_from_cd_result(record)
    scenes_payload = record.get("scenes") or []
    scenes = []
    for index, scene in enumerate(scenes_payload, start=1):
        image_id = str(scene.get("image_id", "")).strip()
        summary = str(scene.get("scene_summary") or "").strip() or "장면 설명 없음"
        scenes.append(
            SceneView(
                scene_index=int(scene.get("scene_index") or index),
                image_id=image_id,
                image_path=_resolve_image(image_id),
                scene_summary=summary,
                generated_sentence=scene_sentences[index - 1] if index - 1 < len(scene_sentences) else "",
            )
        )
    return EvaluationCase(case_id, experiment, result_file, scenes, full_story)


def discover_result_files() -> list[ResultFile]:
    results: list[ResultFile] = []
    for experiment in EXPERIMENTS:
        directory = OUTPUT_ROOT / experiment
        if not directory.exists():
            continue
        if experiment == "A":
            paths = sorted(directory.glob("*_experiment_a.json"))
        elif experiment == "B":
            paths = sorted(directory.glob("sequence_story.json"))
        else:
            paths = sorted(directory.glob(f"experiment_{experiment.lower()}_result.json"))
            if not paths:
                paths = sorted(directory.glob("*_result.json"))
        results.extend(ResultFile(experiment, path) for path in paths)
    return results


def create_blind_mapping() -> dict[str, dict[str, str]]:
    result_files = discover_result_files()
    shuffled = list(result_files)
    random.SystemRandom().shuffle(shuffled)
    mapping = {
        f"case_{index:03d}": {
            "experiment": result.experiment,
            "result_file": _relative(result.path),
        }
        for index, result in enumerate(shuffled, start=1)
    }
    _write_json(MAPPING_FILE, mapping)
    return mapping


def load_or_create_mapping() -> dict[str, dict[str, str]]:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    if MAPPING_FILE.exists():
        mapping = _read_json(MAPPING_FILE)
        if isinstance(mapping, dict) and mapping:
            return mapping
    return create_blind_mapping()


def load_cases(mapping: dict[str, dict[str, str]]) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    errors: list[str] = []
    for case_id in sorted(mapping):
        item = mapping[case_id]
        experiment = str(item.get("experiment", "")).upper()
        result_file = _resolve_result_path(str(item.get("result_file", "")))
        if experiment not in EXPERIMENTS or not result_file.exists():
            errors.append(f"{case_id}: 결과 파일을 읽을 수 없습니다.")
            continue
        try:
            cases.append(_case_from_record(case_id, result_file, experiment))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            errors.append(f"{case_id}: {exc}")
    for error in errors:
        st.warning(error)
    return cases


def load_records() -> list[dict[str, Any]]:
    if not RECORDS_FILE.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in RECORDS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def latest_records_by_case(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("case_id", ""))
        if case_id:
            latest[case_id] = record
    return latest


def save_record(record: dict[str, Any]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with RECORDS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _score_average(scene_quantitative: list[dict[str, Any]]) -> float:
    values = []
    for scene in scene_quantitative:
        for metric in QUANTITATIVE_METRICS:
            value = scene.get(metric)
            if isinstance(value, int | float):
                values.append(float(value))
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def build_summary(
    mapping: dict[str, dict[str, str]],
    latest_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    qualitative_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    for case_id, record in latest_records.items():
        item = mapping.get(case_id)
        if not item:
            continue
        experiment = str(item.get("experiment", "")).upper()
        if experiment not in EXPERIMENTS:
            continue

        for scene in record.get("scene_quantitative", []):
            for metric in QUANTITATIVE_METRICS:
                value = scene.get(metric)
                if isinstance(value, int | float):
                    metric_values[experiment][metric].append(float(value))
        overall = record.get("scene_average_score")
        if isinstance(overall, int | float):
            metric_values[experiment]["scene_average_score"].append(float(overall))

        story_qualitative = record.get("story_qualitative") or {}
        for key in QUALITATIVE_OPTIONS:
            value = str(story_qualitative.get(key, "")).strip()
            if value:
                qualitative_counts[experiment][key][value] += 1

    quantitative = {}
    for experiment, metrics in metric_values.items():
        quantitative[experiment] = {
            metric: round(sum(values) / len(values), 3)
            for metric, values in metrics.items()
            if values
        }

    qualitative = {
        experiment: {
            key: dict(counter)
            for key, counter in fields.items()
        }
        for experiment, fields in qualitative_counts.items()
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_cases": sorted(latest_records),
        "quantitative_by_experiment": quantitative,
        "qualitative_by_experiment": qualitative,
    }
    _write_json(SUMMARY_FILE, summary)
    return summary


def _existing_scene_scores(
    existing_record: dict[str, Any] | None,
    scene_index: int,
) -> dict[str, int]:
    if not existing_record:
        return {}
    for scene in existing_record.get("scene_quantitative", []):
        if int(scene.get("scene_index", -1)) == scene_index:
            return {
                metric: int(scene.get(metric, 3))
                for metric in QUANTITATIVE_METRICS
            }
    return {}


def _existing_qualitative(
    existing_record: dict[str, Any] | None,
    key: str,
) -> str | None:
    if not existing_record:
        return None
    value = (existing_record.get("story_qualitative") or {}).get(key)
    return str(value) if value else None


def _render_case_navigation(cases: list[EvaluationCase]) -> EvaluationCase:
    if "case_position" not in st.session_state:
        st.session_state.case_position = 0
    st.session_state.case_position = min(st.session_state.case_position, len(cases) - 1)
    current_position = st.session_state.case_position
    current_case = cases[current_position]

    left, center, right = st.columns([1, 2, 1])
    with left:
        if st.button("이전", disabled=current_position == 0):
            st.session_state.case_position = max(0, current_position - 1)
            st.rerun()
    with center:
        st.markdown(
            f"### 현재 평가 대상: `{current_case.case_id}`  \n"
            f"진행률: `{current_position + 1} / {len(cases)}`"
        )
    with right:
        if st.button("다음", disabled=current_position == len(cases) - 1):
            st.session_state.case_position = min(len(cases) - 1, current_position + 1)
            st.rerun()
    return current_case


def _render_scene(scene: SceneView, existing_record: dict[str, Any] | None, case_id: str) -> None:
    st.markdown(f"#### {scene.scene_index}번째 장면")
    image_col, eval_col = st.columns([1.2, 1])
    with image_col:
        if scene.image_path and scene.image_path.exists():
            st.image(str(scene.image_path), use_container_width=True)
        else:
            st.info("이미지 파일을 찾을 수 없습니다.")
        st.markdown("**장면 설명**")
        st.write(scene.scene_summary or "장면 설명 없음")
        st.markdown("**생성 문장**")
        st.write(scene.generated_sentence or "이 장면에 직접 대응되는 문장이 없습니다. 전체 이야기를 함께 참고하세요.")

    with eval_col:
        existing_scores = _existing_scene_scores(existing_record, scene.scene_index)
        for metric, label in QUANTITATIVE_METRICS.items():
            st.slider(
                label,
                min_value=1,
                max_value=5,
                value=existing_scores.get(metric, 3),
                step=1,
                key=f"{case_id}_{scene.scene_index}_{metric}",
            )


def _collect_scene_quantitative(case: EvaluationCase) -> list[dict[str, int]]:
    scenes = []
    for scene in case.scenes:
        scores = {
            "scene_index": scene.scene_index,
        }
        for metric in QUANTITATIVE_METRICS:
            scores[metric] = int(st.session_state[f"{case.case_id}_{scene.scene_index}_{metric}"])
        scenes.append(scores)
    return scenes


def _render_qualitative_form(case: EvaluationCase, existing_record: dict[str, Any] | None) -> None:
    st.markdown("### 전체 이야기 정성 평가")
    for key, config in QUALITATIVE_OPTIONS.items():
        options = list(config["options"])
        existing_value = _existing_qualitative(existing_record, key)
        index = options.index(existing_value) if existing_value in options else 0
        st.radio(
            config["label"],
            options,
            index=index,
            key=f"{case.case_id}_{key}",
            horizontal=True,
        )


def _collect_story_qualitative(case: EvaluationCase) -> dict[str, str]:
    return {
        key: str(st.session_state[f"{case.case_id}_{key}"])
        for key in QUALITATIVE_OPTIONS
    }


def _render_summary(summary: dict[str, Any]) -> None:
    st.divider()
    st.markdown("## 분석 요약")
    st.caption("이 영역은 모든 case 평가가 저장된 뒤에만 실제 실험명 기준으로 표시됩니다.")

    quantitative_rows = []
    for experiment in EXPERIMENTS:
        metrics = summary.get("quantitative_by_experiment", {}).get(experiment)
        if not metrics:
            continue
        quantitative_rows.append(
            {
                "실험": experiment,
                "그림 근거 충실도": metrics.get("visual_groundedness", ""),
                "핵심 대상/행동 정확도": metrics.get("object_action_accuracy", ""),
                "환각 적음": metrics.get("low_hallucination", ""),
                "감정/분위기 반영도": metrics.get("emotion_tone_alignment", ""),
                "장면 문장 언어 품질": metrics.get("scene_linguistic_quality", ""),
                "전체 장면 평균": metrics.get("scene_average_score", ""),
            }
        )
    if quantitative_rows:
        st.markdown("### 실험별 정량 평균")
        st.table(quantitative_rows)

    qualitative = summary.get("qualitative_by_experiment", {})
    if qualitative:
        st.markdown("### 실험별 정성 분포")
        for experiment in EXPERIMENTS:
            fields = qualitative.get(experiment)
            if not fields:
                continue
            st.markdown(f"#### {experiment}")
            rows = []
            for key, config in QUALITATIVE_OPTIONS.items():
                counts = fields.get(key, {})
                if not counts:
                    continue
                distribution = ", ".join(f"{label}: {count}" for label, count in counts.items())
                rows.append({"항목": config["label"], "분포": distribution})
            if rows:
                st.table(rows)


def main() -> None:
    st.set_page_config(
        page_title="Blind Story Evaluation",
        layout="wide",
    )
    st.title("블라인드 스토리 평가")
    st.caption("평가 화면에는 익명 case ID만 표시됩니다. 실험명과 모델명은 숨깁니다.")

    mapping = load_or_create_mapping()
    cases = load_cases(mapping)
    if not cases:
        st.error("평가할 결과 파일이 없습니다. 먼저 A/B/C/D/E/F 결과를 생성하세요.")
        st.code("python run.py all", language="bash")
        return

    records = load_records()
    latest_records = latest_records_by_case(records)
    current_case = _render_case_navigation(cases)
    existing_record = latest_records.get(current_case.case_id)

    completed = len([case for case in cases if case.case_id in latest_records])
    st.progress(completed / len(cases), text=f"저장된 평가: {completed} / {len(cases)}")

    with st.form(f"evaluation_form_{current_case.case_id}"):
        st.markdown("## 장면별 정량 평가")
        st.caption("각 장면은 1점부터 5점까지 평가합니다.")
        for scene in current_case.scenes:
            with st.expander(f"{scene.scene_index}번째 장면 평가", expanded=True):
                _render_scene(scene, existing_record, current_case.case_id)

        st.divider()
        st.markdown("## 전체 이야기")
        st.write(current_case.full_story or "전체 이야기 본문이 없습니다.")
        _render_qualitative_form(current_case, existing_record)

        submitted = st.form_submit_button("현재 case 평가 저장", use_container_width=True)

    if submitted:
        scene_quantitative = _collect_scene_quantitative(current_case)
        story_qualitative = _collect_story_qualitative(current_case)
        record = {
            "case_id": current_case.case_id,
            "scene_quantitative": scene_quantitative,
            "story_qualitative": story_qualitative,
            "scene_average_score": _score_average(scene_quantitative),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_record(record)
        latest_records[current_case.case_id] = record
        build_summary(mapping, latest_records)
        st.success(f"{current_case.case_id} 평가를 저장했습니다.")

    if all(case.case_id in latest_records for case in cases):
        summary = build_summary(mapping, latest_records)
        _render_summary(summary)
    else:
        st.info("모든 case 평가를 저장하면 실험명 기준 분석 요약이 표시됩니다.")


if __name__ == "__main__":
    main()
