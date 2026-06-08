"""[담당 3 / 평가] 생성된 동화에 대한 블라인드 무작위 평가 Streamlit 대시보드.

storypipe demo(또는 cli)가 streamlit 하위프로세스로 띄우며, 저장소 루트의 outputs/를 읽는다.
"""


import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
import streamlit as st


# dashboard.py는 저장소 루트에 있음. (blind_mapping.json 경로가 루트 기준 상대경로)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "outputs"
EVALUATION_DIR = OUTPUT_ROOT / "evaluations"
MAPPING_FILE = EVALUATION_DIR / "blind_mapping.json"
RECORDS_FILE = EVALUATION_DIR / "evaluation_records.jsonl"
SUMMARY_FILE = EVALUATION_DIR / "evaluation_summary.json"
EXCEL_RESULTS_FILE = EVALUATION_DIR / "evaluation_results.xlsx"
INPUT_DIR = BASE_DIR / "inputs"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
EXPERIMENTS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")

QUANTITATIVE_METRICS = {
    "visual_groundedness": "그림 근거 충실도",
    "object_action_accuracy": "핵심 대상/행동 정확도",
    "low_hallucination": "환각 적음",
    "emotion_tone_alignment": "감정/분위기 반영도",
    "scene_linguistic_quality": "장면 문장 언어 품질",
}
SCORE_OPTIONS = tuple(range(1, 11))

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
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
            if not candidate.is_absolute():
                candidate = BASE_DIR / candidate
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


def _a_result_sort_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem.replace("_experiment_a", "")
    if stem.isdigit():
        return (0, int(stem), path.name.casefold())
    return (1, stem.casefold(), path.name.casefold())


def _case_from_a_records(case_id: str, result_files: list[Path]) -> EvaluationCase:
    scenes = []
    story_units = []
    for index, result_file in enumerate(sorted(result_files, key=_a_result_sort_key), start=1):
        record = _read_json(result_file)
        image_id = str(record.get("image_id", "")).strip()
        vision = record.get("vision") or {}
        story_final = str(record.get("story_final", "")).strip()
        scene_payload = {
            **((record.get("steps") or {}).get("01_image_input") or {}),
            "image_path": record.get("image_path") or ((record.get("steps") or {}).get("01_image_input") or {}).get("image_path"),
        }
        story_units.append(story_final)
        scenes.append(
            SceneView(
                scene_index=index,
                image_id=image_id,
                image_path=_resolve_image(image_id, scene_payload),
                scene_summary=_compose_a_scene_summary(vision),
                generated_sentence=story_final,
            )
        )
    full_story = "\n\n".join(part for part in story_units if part)
    return EvaluationCase(case_id, "A", result_files[0], scenes, full_story)


def _case_from_record(case_id: str, result_file: Path, experiment: str) -> EvaluationCase:
    record = _read_json(result_file)

    if isinstance(record.get("story"), dict):
        full_story, scene_sentences = _story_from_cd_result(record)
        scenes_payload = record.get("scenes") or []
        scenes = []
        for index, scene in enumerate(scenes_payload, start=1):
            image_id = str(scene.get("image_id", "")).strip()
            summary = (
                str(scene.get("scene_summary") or "").strip()
                or _compose_sequence_scene_summary(scene)
                or "장면 설명 없음"
            )
            scenes.append(
                SceneView(
                    scene_index=int(scene.get("scene_index") or index),
                    image_id=image_id,
                    image_path=_resolve_image(image_id, scene),
                    scene_summary=summary,
                    generated_sentence=scene_sentences[index - 1] if index - 1 < len(scene_sentences) else "",
                )
            )
        return EvaluationCase(case_id, experiment, result_file, scenes, full_story)

    if experiment == "A":
        image_id = str(record.get("image_id", "")).strip()
        vision = record.get("vision") or {}
        full_story = str(record.get("story_final", "")).strip()
        scene_payload = {
            **((record.get("steps") or {}).get("01_image_input") or {}),
            "image_path": record.get("image_path") or ((record.get("steps") or {}).get("01_image_input") or {}).get("image_path"),
        }
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
                    image_path=_resolve_image(image_id, scene),
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
                image_path=_resolve_image(image_id, scene),
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
            standard_path = directory / "experiment_a_result.json"
            paths = [standard_path] if standard_path.exists() else sorted(directory.glob("*_experiment_a.json"))
        elif experiment == "B":
            standard_path = directory / "experiment_b_result.json"
            paths = [standard_path] if standard_path.exists() else sorted(directory.glob("sequence_story.json"))
        else:
            paths = sorted(directory.glob(f"experiment_{experiment.lower()}_result.json"))
            if not paths:
                paths = sorted(directory.glob("*_result.json"))
        results.extend(ResultFile(experiment, path) for path in paths)
    return results


def create_blind_mapping() -> dict[str, dict[str, Any]]:
    result_files = discover_result_files()
    grouped_items: list[dict[str, Any]] = []
    a_paths = sorted(
        (
            result.path
            for result in result_files
            if result.experiment == "A" and result.path.name != "experiment_a_result.json"
        ),
        key=_a_result_sort_key,
    )
    if a_paths:
        grouped_items.append(
            {
                "experiment": "A",
                "result_files": [_relative(path) for path in a_paths],
            }
        )
    grouped_items.extend(
        {
            "experiment": result.experiment,
            "result_file": _relative(result.path),
        }
        for result in result_files
        if result.experiment != "A" or result.path.name == "experiment_a_result.json"
    )
    random.SystemRandom().shuffle(grouped_items)
    mapping = {
        f"case_{index:03d}": item
        for index, item in enumerate(grouped_items, start=1)
    }
    _write_json(MAPPING_FILE, mapping)
    return mapping


def load_or_create_mapping() -> dict[str, dict[str, Any]]:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    if MAPPING_FILE.exists():
        mapping = _read_json(MAPPING_FILE)
        if isinstance(mapping, dict) and mapping:
            return mapping
    return create_blind_mapping()


def load_cases(mapping: dict[str, dict[str, Any]]) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    errors: list[str] = []
    for case_id in sorted(mapping):
        item = mapping[case_id]
        experiment = str(item.get("experiment", "")).upper()
        result_file_values = item.get("result_files")
        if experiment == "A" and isinstance(result_file_values, list):
            result_files = [_resolve_result_path(str(value)) for value in result_file_values]
            missing = [path for path in result_files if not path.exists()]
            if missing:
                errors.append(f"{case_id}: A 결과 파일 일부를 읽을 수 없습니다.")
                continue
            try:
                cases.append(_case_from_a_records(case_id, result_files))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"{case_id}: {exc}")
            continue
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


def _case_by_id(cases: list[EvaluationCase]) -> dict[str, EvaluationCase]:
    return {case.case_id: case for case in cases}


def _ensure_quantitative_scene_selection(
    mapping: dict[str, dict[str, Any]],
    cases: list[EvaluationCase],
) -> None:
    """Persist one random scene per case for quantitative scoring."""
    cases_by_id = _case_by_id(cases)
    changed = False
    randomizer = random.SystemRandom()
    for case_id, item in mapping.items():
        case = cases_by_id.get(case_id)
        if not case or not case.scenes:
            continue
        valid_indices = {scene.scene_index for scene in case.scenes}
        selected = item.get("quantitative_scene_index")
        try:
            selected_index = int(selected)
        except (TypeError, ValueError):
            selected_index = -1
        if selected_index not in valid_indices:
            item["quantitative_scene_index"] = randomizer.choice(sorted(valid_indices))
            changed = True
    if changed:
        _write_json(MAPPING_FILE, mapping)


def _selected_quantitative_scenes(
    case: EvaluationCase,
    mapping: dict[str, dict[str, Any]],
) -> list[SceneView]:
    item = mapping.get(case.case_id) or {}
    try:
        selected_index = int(item.get("quantitative_scene_index"))
    except (TypeError, ValueError):
        selected_index = case.scenes[0].scene_index if case.scenes else -1
    selected = [scene for scene in case.scenes if scene.scene_index == selected_index]
    if selected:
        return selected
    return case.scenes[:1]


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
    mapping: dict[str, dict[str, Any]],
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

        selected_scene_index = item.get("quantitative_scene_index")
        try:
            selected_scene_index = int(selected_scene_index)
        except (TypeError, ValueError):
            selected_scene_index = None
        for scene in record.get("scene_quantitative", []):
            if selected_scene_index is not None:
                try:
                    scene_index = int(scene.get("scene_index"))
                except (TypeError, ValueError):
                    scene_index = -1
                if scene_index != selected_scene_index:
                    continue
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
    if all(case_id in latest_records for case_id in mapping):
        _write_excel_results(summary, mapping, latest_records)
    return summary


def _excel_column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _excel_cell_xml(row_index: int, column_index: int, value: Any) -> str:
    reference = f"{_excel_column_name(column_index)}{row_index}"
    if value is None:
        value = ""
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = xml_escape(str(value))
    return f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'


def _excel_sheet_xml(rows: list[list[Any]]) -> str:
    row_xml = []
    for row_index, row in enumerate(rows, start=1):
        cells = "".join(_excel_cell_xml(row_index, column_index, value) for column_index, value in enumerate(row))
        row_xml.append(f'<row r="{row_index}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        "</worksheet>"
    )


def _write_xlsx(path: Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook_sheets = []
    relationships = []
    content_overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index, (sheet_name, _rows) in enumerate(sheets, start=1):
        safe_name = xml_escape(sheet_name[:31] or f"Sheet{index}")
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{index}" r:id="rId{index}"/>')
        relationships.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    relationships.append(
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f"{''.join(content_overrides)}"
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{''.join(workbook_sheets)}</sheets>"
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{''.join(relationships)}"
            "</Relationships>",
        )
        archive.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<fonts count=\"1\"><font><sz val=\"11\"/><name val=\"Calibri\"/></font></fonts>"
            "<fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>"
            "<borders count=\"1\"><border/></borders>"
            "<cellStyleXfs count=\"1\"><xf/></cellStyleXfs>"
            "<cellXfs count=\"1\"><xf/></cellXfs>"
            "</styleSheet>",
        )
        for index, (_sheet_name, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _excel_sheet_xml(rows))


def _summary_quantitative_rows(summary: dict[str, Any], mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_experiment = {
        str(item.get("experiment", "")).upper(): item.get("quantitative_scene_index")
        for item in mapping.values()
    }
    rows = []
    for experiment in EXPERIMENTS:
        metrics = summary.get("quantitative_by_experiment", {}).get(experiment)
        if not metrics:
            continue
        row = {
            "experiment": experiment,
            "selected_scene_index": selected_by_experiment.get(experiment, ""),
            "scene_average_score": metrics.get("scene_average_score", ""),
        }
        row.update({metric: metrics.get(metric, "") for metric in QUANTITATIVE_METRICS})
        rows.append(row)
    return rows


def _write_excel_results(
    summary: dict[str, Any],
    mapping: dict[str, dict[str, Any]],
    latest_records: dict[str, dict[str, Any]],
) -> None:
    quantitative_rows = _summary_quantitative_rows(summary, mapping)
    quantitative_sheet = [
        ["experiment", "selected_scene_index", "scene_average_score", *QUANTITATIVE_METRICS.keys()]
    ]
    quantitative_sheet.extend(
        [
            [
                row.get("experiment", ""),
                row.get("selected_scene_index", ""),
                row.get("scene_average_score", ""),
                *(row.get(metric, "") for metric in QUANTITATIVE_METRICS),
            ]
            for row in quantitative_rows
        ]
    )

    qualitative_sheet = [["experiment", "field", "option", "count"]]
    for experiment in EXPERIMENTS:
        fields = summary.get("qualitative_by_experiment", {}).get(experiment, {})
        for field, counts in fields.items():
            for option, count in counts.items():
                qualitative_sheet.append([experiment, field, option, count])

    case_sheet = [["case_id", "experiment", "result_file", "selected_scene_index", "scene_average_score"]]
    for case_id in sorted(mapping):
        item = mapping[case_id]
        record = latest_records.get(case_id, {})
        case_sheet.append(
            [
                case_id,
                item.get("experiment", ""),
                item.get("result_file", ""),
                item.get("quantitative_scene_index", ""),
                record.get("scene_average_score", ""),
            ]
        )

    _write_xlsx(
        EXCEL_RESULTS_FILE,
        [
            ("quantitative_scores", quantitative_sheet),
            ("qualitative_counts", qualitative_sheet),
            ("case_records", case_sheet),
        ],
    )


def _existing_scene_scores(
    existing_record: dict[str, Any] | None,
    scene_index: int,
) -> dict[str, int]:
    if not existing_record:
        return {}
    for scene in existing_record.get("scene_quantitative", []):
        if int(scene.get("scene_index", -1)) == scene_index:
            return {
                metric: _coerce_score(scene.get(metric), default=5)
                for metric in QUANTITATIVE_METRICS
            }
    return {}


def _coerce_score(value: Any, default: int = 5) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = default
    return min(max(score, SCORE_OPTIONS[0]), SCORE_OPTIONS[-1])


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


def _render_story_image_overview(case: EvaluationCase) -> None:
    st.markdown("## 전체 그림 흐름")
    st.caption("전체 이야기 평가를 위해 10장 그림의 순서를 먼저 확인하세요.")
    column_count = 5
    for row_start in range(0, len(case.scenes), column_count):
        columns = st.columns(column_count)
        for offset, scene in enumerate(case.scenes[row_start : row_start + column_count]):
            with columns[offset]:
                if scene.image_path and scene.image_path.exists():
                    st.image(
                        str(scene.image_path),
                        caption=f"{scene.scene_index}번째",
                        width="stretch",
                    )
                else:
                    st.info(f"{scene.scene_index}번째 이미지 없음")


def _render_scene(scene: SceneView, existing_record: dict[str, Any] | None, case_id: str) -> None:
    st.markdown(f"#### {scene.scene_index}번째 장면")
    image_col, eval_col = st.columns([1.2, 1])
    with image_col:
        if scene.image_path and scene.image_path.exists():
            st.image(str(scene.image_path), width="stretch")
        else:
            st.info("이미지 파일을 찾을 수 없습니다.")
        st.markdown("**장면 설명**")
        st.write(scene.scene_summary or "장면 설명 없음")
        st.markdown("**생성 문장**")
        st.write(scene.generated_sentence or "이 장면에 직접 대응되는 문장이 없습니다. 전체 이야기를 함께 참고하세요.")

    with eval_col:
        existing_scores = _existing_scene_scores(existing_record, scene.scene_index)
        for metric, label in QUANTITATIVE_METRICS.items():
            score = _coerce_score(existing_scores.get(metric), default=5)
            st.radio(
                label,
                SCORE_OPTIONS,
                index=score - 1,
                key=f"{case_id}_{scene.scene_index}_{metric}",
                horizontal=True,
            )


def _collect_scene_quantitative(case: EvaluationCase, scenes_to_score: list[SceneView]) -> list[dict[str, int]]:
    scenes = []
    for scene in scenes_to_score:
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


def _render_results_dashboard(
    mapping: dict[str, dict[str, Any]],
    latest_records: dict[str, dict[str, Any]],
) -> None:
    summary = build_summary(mapping, latest_records)
    st.success("모든 평가가 완료되었습니다.")
    st.markdown("## 평가 결과")
    st.caption("모든 case 평가가 저장된 뒤 실제 실험 버전 기준으로 공개되는 결과 화면입니다.")

    quantitative_rows = _summary_quantitative_rows(summary, mapping)
    if quantitative_rows:
        display_rows = [
            {
                "버전": row.get("experiment", ""),
                "평가 장면": row.get("selected_scene_index", ""),
                "장면 평균": row.get("scene_average_score", ""),
                "그림 근거": row.get("visual_groundedness", ""),
                "대상/행동 정확도": row.get("object_action_accuracy", ""),
                "환각 적음": row.get("low_hallucination", ""),
                "감정/분위기 반영": row.get("emotion_tone_alignment", ""),
                "문장 언어 품질": row.get("scene_linguistic_quality", ""),
            }
            for row in quantitative_rows
        ]
        st.markdown("### 버전별 정량 점수")
        st.dataframe(display_rows, width="stretch", hide_index=True)

        chart_data = pd.DataFrame(
            [
                {
                    "버전": row["experiment"],
                    "장면 평균": float(row.get("scene_average_score") or 0),
                }
                for row in quantitative_rows
            ]
        ).set_index("버전")
        st.markdown("### A-J 점수 변화")
        st.line_chart(chart_data)

    qualitative = summary.get("qualitative_by_experiment", {})
    if qualitative:
        st.markdown("### 버전별 정성 평가 분포")
        for experiment in EXPERIMENTS:
            fields = qualitative.get(experiment)
            if not fields:
                continue
            rows = []
            for key, config in QUALITATIVE_OPTIONS.items():
                counts = fields.get(key, {})
                if not counts:
                    continue
                distribution = ", ".join(str(label) for label in counts)
                rows.append({"항목": config["label"], "분포": distribution})
            if rows:
                st.markdown(f"#### {experiment}")
                st.dataframe(rows, width="stretch", hide_index=True)

    if EXCEL_RESULTS_FILE.exists():
        st.markdown("### 엑셀 결과 파일")
        st.code(str(EXCEL_RESULTS_FILE.relative_to(BASE_DIR)), language="text")
        st.download_button(
            "엑셀 파일 다운로드",
            data=EXCEL_RESULTS_FILE.read_bytes(),
            file_name=EXCEL_RESULTS_FILE.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )


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
        st.error("평가할 결과 파일이 없습니다. 먼저 A/B/C/D/E/F/G/H/I/J 결과를 생성하세요.")
        st.code("python run.py all", language="bash")
        return

    _ensure_quantitative_scene_selection(mapping, cases)

    records = load_records()
    latest_records = latest_records_by_case(records)
    all_completed = all(case.case_id in latest_records for case in cases)
    if all_completed and st.session_state.get("view_mode", "results") == "results":
        _render_results_dashboard(mapping, latest_records)
        if st.button("평가 화면으로 돌아가기", width="stretch"):
            st.session_state.view_mode = "evaluate"
            st.rerun()
        return

    current_case = _render_case_navigation(cases)
    existing_record = latest_records.get(current_case.case_id)

    completed = len([case for case in cases if case.case_id in latest_records])
    st.progress(completed / len(cases), text=f"저장된 평가: {completed} / {len(cases)}")

    _render_story_image_overview(current_case)
    quantitative_scenes = _selected_quantitative_scenes(current_case, mapping)

    st.markdown("## 장면별 정량 평가")
    st.caption("각 장면은 1점부터 10점까지 숫자를 클릭해 평가합니다.")
    st.info("정량 평가는 이 버전에서 무작위로 선택된 한 장면만 진행합니다. 전체 이야기 정성 평가는 아래에서 그대로 진행합니다.")
    for scene in quantitative_scenes:
        with st.expander(f"{scene.scene_index}번째 장면 평가", expanded=True):
            _render_scene(scene, existing_record, current_case.case_id)

    st.divider()
    _render_story_image_overview(current_case)
    st.divider()
    st.markdown("## 전체 이야기")
    st.write(current_case.full_story or "전체 이야기 본문이 없습니다.")
    _render_qualitative_form(current_case, existing_record)

    submitted = st.button("현재 case 평가 저장", width="stretch")

    if submitted:
        scene_quantitative = _collect_scene_quantitative(current_case, quantitative_scenes)
        story_qualitative = _collect_story_qualitative(current_case)
        record = {
            "case_id": current_case.case_id,
            "quantitative_scene_indices": [scene.scene_index for scene in quantitative_scenes],
            "scene_quantitative": scene_quantitative,
            "story_qualitative": story_qualitative,
            "scene_average_score": _score_average(scene_quantitative),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_record(record)
        latest_records[current_case.case_id] = record
        build_summary(mapping, latest_records)
        if all(case.case_id in latest_records for case in cases):
            st.session_state.view_mode = "results"
            st.rerun()
        st.success(f"{current_case.case_id} 평가를 저장했습니다.")

    if st.button("다음 case로 이동", disabled=st.session_state.case_position >= len(cases) - 1, width="stretch"):
        st.session_state.case_position = min(len(cases) - 1, st.session_state.case_position + 1)
        st.rerun()

    if all(case.case_id in latest_records for case in cases):
        st.info("모든 case 평가가 저장되었습니다. 결과 화면에서 점수표와 그래프를 확인할 수 있습니다.")
        if st.button("결과 화면 보기", width="stretch"):
            st.session_state.view_mode = "results"
            st.rerun()
    else:
        st.info("모든 case 평가를 저장하면 실험명 기준 분석 요약이 표시됩니다.")


if __name__ == "__main__":
    main()
