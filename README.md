# Sketch to Story Pipeline

아이 손그림을 순서대로 읽어 오픈소스 비전/언어 모델로 한국어 동화를 생성하고, 블라인드 평가까지 하는 파이프라인입니다.

흐름: **이미지 -> 장면 인식(BLIP/OpenCLIP / Qwen2.5-VL) -> 한국어 동화(EXAONE) -> 평가**

## 설치 (크로스플랫폼: Windows / macOS / Linux 동일)

Python 3.10-3.12에서, 저장소 루트에서:

```bash
pip install -r requirements.txt
python kim_jeongseok_run.py doctor
```

- `pip install`이 **llama-cpp-python 포함 모든 의존성**을 PyPI에서 설치합니다(별도 바이너리 빌드/Vulkan 불필요).
- `python kim_jeongseok_run.py doctor`가 **필요한 오픈소스 모델(~20GB)을 자동 다운로드**하고 점검합니다. 디스크 30GB+ 권장.
- GPU 없이 CPU로 동작합니다(기본). PyTorch 기반 비전 모델의 NVIDIA 가속은 [ADVANCED.md](ADVANCED.md)를 참고하십시오.

## 4가지 명령

- 0) python kim_jeongseok_run.py doctor
  환경 점검 + 모델 자동 다운로드 + 스모크 추론. 최초 실행 시 1회 수행합니다.
- 1) python kim_jeongseok_run.py run <story> <exp>
  이야기 + 실험버전(a~j) 1개. 예: python kim_jeongseok_run.py run 1 e
- 2) python kim_jeongseok_run.py run-all <story>
  이야기의 전체 실험(A~J). 예: python kim_jeongseok_run.py run-all 7
- 3) python kim_jeongseok_run.py demo
  "7. 새로운 이야기" 전체 실험 후 블라인드 평가 대시보드까지

- `<story>`는 번호 사용을 권장합니다(예: `1`, `7`). 한글 폴더명 직접 입력을 피할 수 있습니다.
- 실험 H/I/J와 `demo`는 `caption.txt`가 있는 이야기가 필요합니다(예제는 story 7).
- 다운로드 없이 점검만 실행: `python kim_jeongseok_run.py doctor --check-only`

### 재현 환경 (2단계)

별도 설정 없이 아래 두 단계만으로 동작하도록 구성했습니다. **GPU는 필요하지 않으며**,
CPU만으로 전체 실행과 평가가 완료됩니다.

```
pip install -r requirements.txt
python kim_jeongseok_run.py doctor      # 모델 자동 다운로드 + 점검
python kim_jeongseok_run.py demo        # 전체 실행 + 평가
```

- 첫 실행 시 모델(~20GB)을 자동으로 내려받으므로 시간이 다소 걸릴 수 있습니다.
- CPU 실행 시간을 단축하고자 하는 경우에 한해 BLIP/OpenCLIP/Qwen2.5-VL용 PyTorch GPU 가속을
  [ADVANCED.md](ADVANCED.md)에서 안내합니다.

## 파일 구성 (3인 작업 분담)

코드는 파이프라인 단계별로 3명이 나누어 작업했습니다.

[김기홍 · 비전 — 보고서·발표]
- kim_gihong_vision.py
  BLIP/OpenCLIP 인식(실험 A), Qwen2.5-VL 장면/콜라주 추출(C~J)
- 보고서 작성
  프로젝트 정리, 실험 A~J 결과 분석, 표·그림 작성
- 발표
  발표 자료 구성, 대본 작성

[김정석 · 스토리 — 코딩]
- kim_jeongseok_story_runtime.py
  EXAONE GGUF(llama-cpp-python) 런타임, GPT-2/NLLB 베이스라인, 구조화 플랜
- kim_jeongseok_story_experiments.py
  실험 C~J 프롬프트/품질 게이트/빌더
- kim_jeongseok_common.py
  설정/런타임/로깅/모델 다운로드/이미지/IO/JSON 유틸
- kim_jeongseok_run.py
  4-커맨드 디스패처 (진입점)

[박정우 · 파이프라인 · 평가 — PPT 제작]
- park_jeongwoo_experiment_a.py
  실험 A/B 오케스트레이션 + 정량 평가
- park_jeongwoo_pipeline.py
  4-커맨드 CLI, doctor, C~J 통합 러너, 출력 작성
- park_jeongwoo_dashboard.py
  블라인드 평가 Streamlit 대시보드
  PPT 제작

데이터: `inputs/`(이야기별 그림 10장 + story 7의 `caption.txt`/콜라주). 결과는 `outputs/`에 생성(자동, Git/제출 제외).

## 환경변수 (선택)

- EXAONE_GGUF_MODEL_PATH (기본: 자동)
  직접 받은 GGUF 파일 경로 지정(자동 다운로드 대신).
- HF_TOKEN (기본: 없음)
  로그인 필요(게이트) 모델 다운로드용 Hugging Face 토큰.

NVIDIA GPU 가속(선택): PyTorch를 CUDA 빌드로 설치하면 BLIP/OpenCLIP/Qwen2.5-VL 등 비전 모델이 GPU를 사용합니다. 상세 설치·검증·CPU 복구 절차는 [ADVANCED.md](ADVANCED.md)에서 확인할 수 있습니다. (CPU가 정본/재현 경로이며 GPU는 개발 가속용 opt-in입니다.)

## 문제 해결

**`llama-cpp-python` 설치 실패 (Windows, "Failed building wheel" / C4819 / C2001)**
미리 빌드된 wheel을 받지 못해 소스 C++ 빌드로 전환된 경우입니다. `requirements.txt`에 프리빌트 인덱스가
이미 포함되어 있어 일반적으로는 `pip install -r requirements.txt`로 해결되지만, 그래도 빌드를 시도하면:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary "llama-cpp-python>=0.3.2,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

그래도 소스 빌드로 전환되면(드문 경우) MSVC에 UTF-8 옵션을 지정한 뒤 다시 시도합니다:

```powershell
$env:CL="/utf-8"
python -m pip install --no-cache-dir --force-reinstall "llama-cpp-python>=0.3.2,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

- Python은 **3.10-3.12** 사용을 권장합니다(3.13+는 프리빌트 wheel이 없을 수 있어 소스 빌드로 빠집니다).
- 모델이 게이트면 `huggingface-cli login` 또는 `HF_TOKEN` 설정 후 `python kim_jeongseok_run.py doctor`.
