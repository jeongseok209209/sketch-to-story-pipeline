"""Evaluation metrics for Experiment A."""

from __future__ import annotations

from generators import translate_en_ko


def evaluate(vision: dict, story_ko: str) -> dict[str, float | int]:
    """Calculate object coverage, Korean character count, and paragraph count."""
    objects = vision.get("objects", [])
    if objects:
        hits = 0
        for word in objects:
            translated = translate_en_ko(str(word)).strip()
            if translated and translated in story_ko:
                hits += 1
        object_coverage = hits / len(objects)
    else:
        object_coverage = 0.0

    paragraphs = [part for part in story_ko.split("\n\n") if part.strip()]
    return {
        "object_coverage": object_coverage,
        "char_count": len(story_ko),
        "paragraph_count": len(paragraphs),
    }
