"""Evaluation metrics for Experiment A."""

from __future__ import annotations

from generators import CONCEPT_KO, translate_en_ko


def evaluate(
    vision: dict,
    story_ko: str,
    translate_objects: bool = True,
) -> dict[str, float | int | str]:
    """Calculate object coverage, Korean character count, and paragraph count."""
    # vision 단계에서 찾은 object가 최종 한국어 이야기 안에 얼마나 반영됐는지 계산합니다.
    objects = vision.get("objects", [])
    if objects:
        hits = 0
        for word in objects:
            # 기본 경로에서는 영어 object 후보를 NLLB로 한국어 표기로 바꾼 뒤 포함 여부를 봅니다.
            # 한국어 직접 생성 경로에서는 NLLB를 다시 로드하지 않도록 내장 개념 사전과 원문 후보를 검사합니다.
            if translate_objects:
                candidates = [translate_en_ko(str(word)).strip()]
            else:
                raw_word = str(word).lower().strip()
                candidates = [CONCEPT_KO.get(raw_word, ""), raw_word]
            if any(candidate and candidate.lower() in story_ko.lower() for candidate in candidates):
                hits += 1
        object_coverage = hits / len(objects)
    else:
        object_coverage = 0.0

    # 빈 줄 두 개를 문단 구분으로 보고, 공백 문단은 평가에서 제외합니다.
    paragraphs = [part for part in story_ko.split("\n\n") if part.strip()]
    return {
        "object_coverage": object_coverage,
        "object_coverage_mode": "nllb_translation" if translate_objects else "lexicon_or_raw_match",
        "char_count": len(story_ko),
        "paragraph_count": len(paragraphs),
    }
