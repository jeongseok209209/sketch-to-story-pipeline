# Architecture / 구조

손그림 → 한국어 동화 파이프라인. 단일 패키지 `storypipe/`, 도메인별 서브패키지.

## 의존성 레이어 (순환 없음)

```
                    storypipe/common/        (토대: config, runtime, logging, models, images, io, jsonparse)
                          ▲
        ┌─────────────────┼──────────────────┐
   storypipe/vision/                    storypipe/story/
   (BLIP·CLIP, Qwen 장면)               (EXAONE GGUF, 베이스라인, 실험 C~J)
        ▲                                     ▲
        └─────────────────┬──────────────────┘
                  storypipe/pipeline/         (cli, doctor, experiment_a, runner, outputs, evaluate)
                          │
                  storypipe/evaluation/       (dashboard — 독립 streamlit, outputs/ 읽음)
```

- `vision`과 `story`는 서로 import하지 않는다. `story.experiments`는 장면(scenes)을 **인자**로 받는다.
- `pipeline/runner.py`가 vision(장면 추출) + story(스토리 생성)를 묶는 유일한 통합 지점이다.
- `evaluation/dashboard.py`는 `cli`가 streamlit 하위프로세스로 띄우며, `outputs/`를 읽는 독립 앱이다.

## 모듈 지도

| 패키지 | 모듈 | 역할 |
| --- | --- | --- |
| common | config / runtime / logging / models / images / io / jsonparse | 상수·경로, 디바이스·CUDA 감지, 단계 로그, HF/GGUF 다운로드, 이미지·JSON·파일 유틸 |
| vision | blip_clip / qwen_scenes / loaders | 실험 A 인식 / Qwen 장면·콜라주 추출 / 비전 모델 로더 |
| story | exaone_runtime / baseline / experiments / loaders | EXAONE GGUF(llama-cpp-python)·구조화 플랜 / GPT-2·NLLB·개념사전 / 실험 C~J / LLM 로더 |
| pipeline | cli / doctor / experiment_a / runner / outputs / evaluate | 4-커맨드 CLI / 환경 점검·설치 / 실험 A·B / C~J 통합 / 결과 작성 / 정량 평가 |
| evaluation | dashboard | 블라인드 평가 Streamlit 앱 |

## 실험 개요

| 실험 | 비전 입력 | 언어 모델 | 특징 |
| --- | --- | --- | --- |
| A | BLIP/OpenCLIP 단일 이미지 | GPT-2 + NLLB | 단일 이미지 베이스라인 |
| B | BLIP/OpenCLIP 순서 장면 | EXAONE GGUF | 순서 장면 연결 스토리 |
| C~F | Qwen2.5-VL 장면 JSON | EXAONE GGUF | 프롬프트 전략 비교(단순/플랜/연속성/장면윈도우) |
| G~J | Qwen2.5-VL 장면 JSON (+캡션/콜라주) | EXAONE GGUF | 페르소나·캡션·콜라주 유도 변형. H/I/J는 `caption.txt` 필요, I/J는 콜라주 필요 |

## EXAONE GGUF 런타임 (재현성 핵심)

EXAONE GGUF는 **llama-cpp-python(in-process)**으로 실행한다(`story/exaone_runtime.py`). 과거의 llama-cli
바이너리 다운로드(하드코딩 Windows-Vulkan zip)/소스 빌드 의존을 제거해, `pip install`만으로 어느 OS에서도
동작한다. 기본은 CPU(`LLAMA_GPU_LAYERS=0`), NVIDIA GPU 가속은 환경변수로 opt-in.

## 진입점

- `storypipe <cmd>` (설치 후 콘솔 스크립트) / `python -m storypipe <cmd>` / `python run.py <cmd>` (무설치 폴백)
- 4-커맨드: `doctor`, `run <story> <exp>`, `run-all <story>`, `demo` — [README.md](README.md) 참고.
