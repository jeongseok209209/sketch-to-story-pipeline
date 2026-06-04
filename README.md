# A-to-F 3B 실행 패키지

이 폴더는 손그림 입력을 A/B/C/D/E/F 실험 방식으로 실행하는 패키지입니다.
통합 실행 진입점은 `run.py`입니다.

## 통합 실행

```bash
python run.py a
python run.py b
python run.py c
python run.py d
python run.py e
python run.py f
python run.py all
python run.py evaluate
```

기본 입력 폴더는 `inputs/`입니다.
기본 결과 저장 위치는 `outputs/`입니다.

- `a`: BLIP/BLIP-VQA/OpenCLIP + GPT-2/NLLB 또는 선택 backend
- `b`: 여러 장 입력을 순서형 이야기로 묶는 sequence/B-style 실행
- `c`: Qwen2.5-VL-3B로 전체 장면을 새로 인식한 뒤 C 결과 생성
- `d`: Qwen2.5-VL-3B로 전체 장면을 새로 인식한 뒤 D 결과 생성
- `e`: Qwen2.5-VL-3B로 전체 장면을 새로 인식한 뒤 E 결과 생성
- `f`: Qwen2.5-VL-3B로 전체 장면을 새로 인식한 뒤 현재 그림 기준 grounding 검사를 포함한 F 결과 생성

모드별 기본 저장 폴더:

```text
outputs/A
outputs/B
outputs/C
outputs/D
outputs/E
outputs/F
outputs/qwen25_vl_3b_story
```

중요: `c`, `d`, `e`, `f`는 실행할 때마다 기존 장면 캐시를 재사용하지 않고 Qwen2.5-VL-3B로 `inputs/` 전체를 다시 인식합니다. 결과 캐시는 확인용으로 다시 저장될 뿐입니다. `f`는 현재 그림 기준 grounding 보정을 별도 실험으로 기록한 모드입니다.

예시:

```bash
python run.py a --image 1 --story-backend structured_template
python run.py a --batch --story-backend gpt2_nllb
python run.py b --story-backend prompt_twostep_short --output-dir outputs/B_prompt_twostep
python run.py c
python run.py f
```

## 블라인드 평가 대시보드

생성 결과를 사람이 평가할 때는 별도 평가 모드를 사용합니다.

```bash
streamlit run evaluation_dashboard.py
```

또는 통합 실행 진입점으로 실행할 수 있습니다.

```bash
python run.py evaluate
```

대시보드는 `outputs/A`~`outputs/F`의 결과 JSON을 읽고, 평가 화면에서는 실험명과 모델명을 숨긴 익명 case만 보여줍니다. 처음 실행할 때 평가 대상 순서를 무작위로 섞어 `outputs/evaluations/blind_mapping.json`에 저장합니다. 평가 기록은 실험명을 포함하지 않고 `outputs/evaluations/evaluation_records.jsonl`에 누적 저장됩니다.

평가 방식:

- 장면별 정량 평가: 이미지 충실도, 감정 반영도, 환각 적음, 장면 문장 자연성을 1~5점으로 입력
- 전체 이야기 정성 평가: 이야기 구조, 장면 연결 방식, 감정 흐름, 이야기 톤, 주요 실패 유형, 동화 적합성을 선택형으로 입력
- 모든 case 평가가 저장되면 `outputs/evaluations/evaluation_summary.json`에 실제 실험명 기준 집계가 생성되고, 대시보드 하단에 분석 요약이 표시됩니다.

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

EXAONE으로 한국어 동화를 직접 생성:

```bash
python pipeline_a.py --image 1 --story-backend exaone
```

구조화 파이프라인으로 빠르게 한국어 동화 생성:

```bash
python pipeline_a.py --image 1 --story-backend structured_template
```

구조화 파이프라인 초안을 EXAONE으로 한 번만 보정:

```bash
python pipeline_a.py --image 1 --story-backend structured_exaone --story-max-new-tokens 60
```

BLIP/OpenCLIP 다음에 EXAONE으로 한국어 구조화/기획을 한 번 생성하고 템플릿으로 동화 작성:

```bash
python pipeline_a.py --image 1 --story-backend exaone_structured --story-max-new-tokens 180
```

EXAONE 출력 길이 조정:

```bash
python pipeline_a.py --image 1 --story-backend exaone --story-max-new-tokens 120
```

기본값은 기존 방식인 `gpt2_nllb`입니다. 로컬 CPU에서는 `structured_template`이 가장 빠르고 안정적입니다. `exaone_structured`는 BLIP/OpenCLIP 결과를 EXAONE이 한국어 `structured_json`과 `plan_json`으로 정리하고, 최종 동화 문장은 템플릿으로 안정적으로 만듭니다. `structured_exaone`은 구조화/기획/초안은 규칙 기반으로 만들고 마지막 문체 보정만 EXAONE에 맡깁니다. EXAONE 실행은 `LGAI-EXAONE/EXAONE-4.0-1.2B` 모델을 HuggingFace Hub에서 내려받으며, EXAONE 라이선스 조건을 확인해야 합니다.

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
├── 07_gpt2_story_en.json 또는 07_structured_json.json
├── 08_nllb_translation_ko.json 또는 08_plan_json.json
├── 09_story_draft.json
├── 10_story_final.json
├── 11_evaluation.json
└── run_record.json
```

```json
{
  "image_id": "kid_001.png",
  "experiment": "A",
  "story_backend": "gpt2_nllb",
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
