"""[담당 2 · 스토리] 실험 A 베이스라인 언어 생성 + 한국어 개념 사전.

GPT-2 영어 초안 → NLLB 한국어 번역(실험 A). CONCEPT_KO/PHRASE_KO/STOPWORDS는 vision 개념을
한국어로 매핑하는 자원으로, EXAONE 구조화 플랜(exaone_runtime)과 평가(evaluate)에서도 쓴다.
"""

from __future__ import annotations

from storypipe.common.logging import timed_step
from storypipe.common.runtime import get_device
from storypipe.story.loaders import get_gpt2_components, get_nllb_components


# ─────────────────────────────────────────────────────────────────────────────
# 아래 본문은 기존 generators.py에서 이동한 코드.
# ─────────────────────────────────────────────────────────────────────────────
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
    "cactus": "선인장",
    "cat": "고양이",
    "child": "아이",
    "children": "아이들",
    "cloud": "구름",
    "dog": "강아지",
    "drawing": "그림",
    "family": "가족",
    "flower": "꽃",
    "flying": "나는 모습",
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
    "stork": "황새",
    "sun": "해",
    "sunlight": "햇살",
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
    "calm and magical": "차분하고 신비로운",
    "wonder": "신비로운",
    "park": "공원",
    "in front of house": "집 앞",
    "in front of a house": "집 앞",
    "house in front": "집 앞",
    "door": "문",
    "friendship under the stars": "별빛 아래 나누는 우정",
    "family bonding under the stars": "별빛 아래 나누는 다정한 마음",
    "family fun under the stars": "별빛 아래 나누는 즐거운 마음",
    "family joy under the stars": "별빛 아래 가족이 나누는 기쁨",
    "nature exploration": "자연을 살피는 모험",
}


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

    with timed_step(8, "GPT-2 English story generation", model="gpt2-medium"):
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
    with timed_step(9, "NLLB English-to-Korean translation", model="facebook/nllb-200-distilled-600M"):
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
