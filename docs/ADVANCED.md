# Advanced / 고급 설정

기본 사용은 [README.md](../README.md)의 4-커맨드면 충분합니다. 이 문서는 수동 설정·GPU·환경변수·문제 해결을 모읍니다.

## 수동 설치 (setup 스크립트 대신)

```bash
# macOS / Linux
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .
./.venv/bin/storypipe doctor
```

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run.py doctor
```

요구사항: Python 3.10–3.12 (x64), 디스크 30GB+ 권장, 인터넷. GPU는 선택.

## 모델 자동 다운로드

`storypipe doctor`가 다음을 Hugging Face에서 자동으로 받습니다(첫 실행 ~20GB):
BLIP captioning/VQA, GPT-2 medium, NLLB-200-distilled-600M, OpenCLIP ViT-H-14, Qwen2.5-VL-3B, EXAONE-4.0-1.2B GGUF.

- 게이트(로그인 필요) 모델이면 `huggingface-cli login` 또는 `HF_TOKEN` 환경변수를 설정하세요.
- 이미 받은 EXAONE GGUF가 있으면 `EXAONE_GGUF_MODEL_PATH`로 직접 지정할 수 있습니다.

### (선택) 오프라인/저속망 대비

인터넷이 약한 채점 환경이면, 동작하는 머신에서 모델을 받은 뒤 `.local_models/` 폴더를 통째로 복사해
대상 머신의 저장소 루트에 넣으면 `doctor`가 그대로 인식합니다.

## GPU / CUDA (선택)

기본은 CPU로 어디서나 동작합니다. NVIDIA GPU 가속을 원하면:

```powershell
# 1) CUDA용 PyTorch 교체
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu124
# 2) EXAONE GGUF GPU offload 켜기
$env:LLAMA_GPU_LAYERS="999"
```

llama-cpp-python의 CUDA/Metal 가속 빌드가 필요하면 해당 프로젝트 설치 가이드를 따르세요(CPU 휠로도 동작).

## 환경변수

| 변수 | 기본 | 설명 |
| --- | --- | --- |
| `LLAMA_GPU_LAYERS` | `0` | EXAONE GGUF GPU offload 레이어 수. CPU=0, GPU 가속 시 999 등. |
| `EXAONE_N_CTX` | `8192` | EXAONE GGUF 컨텍스트 길이. |
| `EXAONE_GGUF_MODEL_PATH` | (자동) | 직접 받은 GGUF 파일 경로. |
| `HF_TOKEN` | — | 게이트 모델 다운로드용 Hugging Face 토큰. |
| `HF_REVISION_<모델>` | (최신) | 모델 리비전 고정(재현성). 키: 모델 ID의 `/ . -` 를 `_`로 치환. |

`.env.example`를 복사해 쓰면 편합니다.

## 정확 재현 (lockfile)

상한 핀(requirements.txt의 `<`)으로 주요 드리프트는 막았습니다. 바이트 단위 재현이 필요하면:

```bash
./.venv/bin/storypipe doctor          # 성공 확인 후
./.venv/bin/python -m pip freeze > requirements.lock   # 잠그고 커밋
```

## 문제 해결

- `Python 3.10-3.12 not found`: 해당 버전을 설치하고 PATH 확인 후 setup 재실행.
- 패키지 누락 경고: `pip install -r requirements.txt` (또는 `pip install -e .`).
- `caption.txt` 오류: 실험 H/I/J와 `demo`는 `caption.txt`가 있는 이야기 필요(예제는 story 7).
- 콜라주 오류: 실험 I/J는 입력 트리에 `collage_2x5_scene_order.png` 필요.
- EXAONE import 오류: `pip install llama-cpp-python` 확인 후 `storypipe doctor`.
- 한글 경로 문제: `--story`에 숫자(예: `7`) 사용 권장.
