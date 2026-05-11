# Experiment A

손그림 스케치를 입력받아 BLIP/BLIP-VQA/OpenCLIP으로 시각 단서를 만들고, GPT-2로 영문 동화를 생성한 뒤 NLLB로 한국어 동화로 번역하는 실험 파이프라인입니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
cd experiment_a
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

모든 모델은 HuggingFace Hub에서 처음 실행 시 다운로드됩니다. CUDA가 가능하면 자동으로 CUDA를 사용하고, 아니면 CPU를 사용합니다.

## 실행

입력 이미지는 기본적으로 `input/` 폴더에 넣습니다. 파일명은 `1.png`, `2.jpg`, `3.jpeg`처럼 숫자로 두면 됩니다.

에디터의 플레이 버튼으로 `pipeline_a.py`를 바로 실행하면 기본으로 `input/1.png`를 처리합니다.

단일 이미지:

```bash
python pipeline_a.py --image 1
```

배치 실행:

```bash
python pipeline_a.py --batch
```

출력 폴더 변경과 OpenCLIP 임계값 조정:

```bash
python pipeline_a.py --image 1 --output-dir results/A --clip-threshold 0.22
```

직접 경로를 넘기는 방식도 사용할 수 있습니다.

```bash
python pipeline_a.py --image ~/Desktop/kid_001.png
```

## 출력 JSON 구조

각 실행 결과는 `results/A/{image_stem}_experiment_a.json`로 저장됩니다.
사진별 단계 산출물은 `results/A/{image_stem}_steps/` 아래에 따로 저장됩니다.

단계별 파일:

```text
results/A/1_steps/
├── 01_image_input.json
├── 02_preprocessing.json
├── 03_blip_captioning.json
├── 04_blip_vqa.json
├── 05_openclip_concept_scoring.json
├── 06_vision_json.json
├── 07_gpt2_story_en.json
├── 08_nllb_translation_ko.json
├── 09_evaluation.json
└── run_record.json
```

```json
{
  "image_id": "kid_001.png",
  "experiment": "A",
  "timestamp": "ISO8601 timestamp",
  "step_output_dir": "results/A/kid_001_steps",
  "steps": {
    "01_image_input": {"step": 1, "name": "손그림 이미지 입력"},
    "02_preprocessing": {"step": 2, "name": "전처리"},
    "03_blip_captioning": {"step": 3, "raw_caption": "..."},
    "04_blip_vqa": {"step": 4, "answers": {"who": "..."}},
    "05_openclip_concept_scoring": {"step": 5, "scores": {"house": 0.91}}
  },
  "vision": {
    "objects": ["house", "tree", "sun", "child"],
    "object_scores": {"house": 0.91, "tree": 0.84},
    "who": "a child",
    "actions": "playing outside",
    "scene": "outside near a house",
    "mood": "happy",
    "raw_caption": "a children's drawing of a house and a tree with a sun",
    "confidence": "high"
  },
  "story_en": "GPT-2 output English story",
  "story_final": "NLLB translated Korean story",
  "metrics": {
    "object_coverage": 0.85,
    "char_count": 1240,
    "paragraph_count": 7
  }
}
```
