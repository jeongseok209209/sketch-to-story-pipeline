# Sketch to Story Pipeline

아이 손그림을 순서대로 읽어 오픈소스 비전/언어 모델로 한국어 동화를 생성하고, 블라인드 평가까지 하는 파이프라인입니다.

흐름: **이미지 → 장면 인식(BLIP·OpenCLIP / Qwen2.5-VL) → 한국어 동화(EXAONE) → 평가**

## 설치 (크로스플랫폼: Windows / macOS / Linux 동일)

Python 3.10–3.12에서, 저장소 루트에서:

```bash
pip install -r requirements.txt
python run.py doctor
```

- `pip install`이 **llama-cpp-python 포함 모든 의존성**을 PyPI에서 설치합니다(별도 바이너리 빌드·Vulkan 불필요).
- `python run.py doctor`가 **필요한 오픈소스 모델(~20GB)을 자동 다운로드**하고 점검합니다. 디스크 30GB+ 권장.
- GPU 없이 CPU로 동작합니다(기본). NVIDIA 가속은 아래 "환경변수" 참고.

## 4가지 명령

| # | 명령 | 설명 |
| --- | --- | --- |
| 0 | `python run.py doctor` | **환경 점검 + 모델 자동 다운로드 + 스모크 추론.** 처음에 한 번. |
| 1 | `python run.py run <story> <exp>` | 이야기 + 실험버전(a~j) 1개. 예: `python run.py run 1 e` |
| 2 | `python run.py run-all <story>` | 이야기의 전체 실험(A~J). 예: `python run.py run-all 7` |
| 3 | `python run.py demo` | "7. 새로운 이야기" 전체 실험 후 블라인드 평가 대시보드까지 |

- `<story>`는 번호 권장(예: `1`, `7`) — 한글 폴더명 입력 회피.
- 실험 H/I/J와 `demo`는 `caption.txt`가 있는 이야기 필요(예제는 story 7).
- 점검만(다운로드 없이): `python run.py doctor --check-only`

### 다른 컴퓨터/채점자 환경 (2단계)
```
pip install -r requirements.txt
python run.py doctor      # 모델 자동 다운로드 + 점검
python run.py demo        # 전체 실행 + 평가
```

## 파일 구성 (3인 작업 분담)

코드는 파이프라인 단계별로 3명이 나누어 작업했습니다.

| 파일 | 담당 | 내용 |
| --- | --- | --- |
| `vision.py` | **담당 1 · 비전** | BLIP/OpenCLIP 인식(실험 A), Qwen2.5-VL 장면/콜라주 추출(C~J) |
| `story_runtime.py` | **담당 2 · 스토리** | EXAONE GGUF(llama-cpp-python) 런타임, GPT-2/NLLB 베이스라인, 구조화 플랜 |
| `story_experiments.py` | **담당 2 · 스토리** | 실험 C~J 프롬프트·품질 게이트·빌더 |
| `experiment_a.py` | **담당 3 · 파이프라인** | 실험 A/B 오케스트레이션 + 정량 평가 |
| `pipeline.py` | **담당 3 · 파이프라인** | 4-커맨드 CLI, doctor(점검·설치), C~J 통합 러너, 출력 작성 |
| `dashboard.py` | **담당 3 · 평가** | 블라인드 평가 Streamlit 대시보드 |
| `common.py` | (공유) | 설정·런타임·로깅·모델 다운로드·이미지·IO·JSON 유틸 |
| `run.py` | (진입점) | 4-커맨드 디스패처 |

데이터: `inputs/`(이야기별 그림 10장 + story 7의 `caption.txt`·콜라주). 결과는 `outputs/`에 생성(자동, Git/제출 제외).

## 환경변수 (선택)

| 변수 | 기본 | 설명 |
| --- | --- | --- |
| `LLAMA_GPU_LAYERS` | `0` | EXAONE GGUF GPU offload(0=CPU). NVIDIA 가속 시 `999`. |
| `EXAONE_GGUF_MODEL_PATH` | (자동) | 직접 받은 GGUF 파일 경로 지정(자동 다운로드 대신). |
| `HF_TOKEN` | — | 로그인 필요(게이트) 모델 다운로드용 Hugging Face 토큰. |

NVIDIA GPU 가속: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` 후 `LLAMA_GPU_LAYERS=999`.

## 문제 해결

**`llama-cpp-python` 설치 실패 (Windows, "Failed building wheel" / C4819 / C2001)**
미리 빌드된 wheel을 못 받아 소스 C++ 빌드로 빠진 경우입니다. `requirements.txt`에 프리빌트 인덱스가
이미 포함돼 있어 보통은 `pip install -r requirements.txt`로 해결되지만, 그래도 빌드를 시도하면:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install --prefer-binary "llama-cpp-python>=0.3.2,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

그래도 소스 빌드로 빠지면(드묾) MSVC에 UTF-8 옵션을 주고 한 번 더:

```powershell
$env:CL="/utf-8"
python -m pip install --no-cache-dir --force-reinstall "llama-cpp-python>=0.3.2,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

- Python은 **3.10–3.12**를 쓰세요(3.13+는 프리빌트 wheel이 없을 수 있어 소스 빌드로 빠집니다).
- 모델이 게이트면 `huggingface-cli login` 또는 `HF_TOKEN` 설정 후 `python run.py doctor`.
