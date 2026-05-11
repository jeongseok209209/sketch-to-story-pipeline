"""Evaluation metrics for Experiment A."""

from __future__ import annotations

from generators import translate_en_ko


def evaluate(vision: dict, story_ko: str) -> dict[str, float | int]:
    """Calculate object coverage, Korean character count, and paragraph count."""
    # vision 단계에서 찾은 object가 최종 한국어 이야기 안에 얼마나 반영됐는지 계산합니다.
    objects = vision.get("objects", [])
    if objects:
        hits = 0
        for word in objects:
            # object 후보는 영어이므로, 같은 번역기를 통해 한국어 표기로 바꾼 뒤 포함 여부를 봅니다.
            translated = translate_en_ko(str(word)).strip()
            if translated and translated in story_ko:
                hits += 1
        object_coverage = hits / len(objects)
    else:
        object_coverage = 0.0

    # 빈 줄 두 개를 문단 구분으로 보고, 공백 문단은 평가에서 제외합니다.
    paragraphs = [part for part in story_ko.split("\n\n") if part.strip()]
    return {
        "object_coverage": object_coverage,
        "char_count": len(story_ko),
        "paragraph_count": len(paragraphs),
    }
