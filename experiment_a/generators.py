"""Story generation and translation for Experiment A."""

from __future__ import annotations

from utils import get_device, get_gpt2_components, get_nllb_components, timed_step


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
