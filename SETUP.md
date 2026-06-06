# Setup Guide

이 문서는 GitHub에서 clone한 프로젝트를 Windows 컴퓨터에서 처음 세팅하는 방법을 설명합니다.

## Requirements

- Windows 10/11 x64
- Python 3.12 x64
- Git
- 인터넷 연결
- 디스크 여유 공간 30GB 이상 권장

GPU는 필수가 아닙니다. NVIDIA GPU가 있으면 더 빠르게 실행될 수 있고, 없으면 CPU 모드로 실행됩니다.

## 1. Clone

```powershell
git clone https://github.com/jeongseok209209/sketch-to-story-pipeline.git
cd sketch-to-story-pipeline
```

## 2. One-Command Setup

VS Code, Cursor, PowerShell, 또는 Windows Terminal에서 프로젝트 폴더를 열고 아래 명령 하나만 실행하세요.

```powershell
.\setup.bat
```

CMD 터미널에서는 `setup.bat`라고 입력해도 됩니다.

이 명령은 내부적으로 다음 작업을 합니다.

- Python 3.12 확인
- `.venv` 생성
- pip 업그레이드
- `requirements.txt` 설치
- NVIDIA GPU가 있으면 `nvidia-smi` 출력
- `run.py check`로 환경 점검

완료 후 작은 실행 테스트 명령을 화면에 보여줍니다.

## Manual Setup

자동 세팅이 아니라 직접 단계별로 실행하고 싶을 때만 아래 명령을 사용하세요.

### 1. Create Virtual Environment

```powershell
py -3.12 -m venv .venv
```

PowerShell 스크립트 활성화는 필요하지 않습니다. 아래처럼 프로젝트 가상환경 Python을 직접 호출합니다.

```powershell
.\.venv\Scripts\python.exe --version
```

### 2. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Check Environment

```powershell
.\.venv\Scripts\python.exe run.py check
```

`check`는 모델 다운로드나 자동 설치를 하지 않고 현재 상태만 확인합니다. Python 패키지, 입력 이미지, GPU/CPU 상태, 로컬 모델 캐시, llama.cpp 위치를 점검합니다.

## PowerShell Script Directly

`setup.bat` 대신 PowerShell 스크립트를 직접 실행할 수도 있습니다.

```powershell
.\setup_windows.ps1
```

PowerShell 실행 정책 때문에 막히면 다음 명령으로 한 번만 실행할 수 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

## GPU and CPU

NVIDIA GPU가 있고 드라이버가 정상 설치되어 있으면 `run.py`가 CUDA 사용 가능 여부를 감지합니다.

확인:

```powershell
nvidia-smi
```

NVIDIA GPU가 없거나 CPU로 강제 실행하고 싶으면:

```powershell
$env:LLAMA_GPU_LAYERS="0"
$env:AUTO_INSTALL_TORCH_CUDA="0"
```

CPU 모드는 가능하지만 Qwen2.5-VL, BLIP, NLLB 모델 실행이 매우 느릴 수 있습니다.

## First Run Downloads

첫 실제 실행에서는 다음 파일들이 `.local_models/`와 `.local_tools/` 아래에 준비됩니다.

- BLIP captioning model
- BLIP VQA model
- GPT-2 medium
- NLLB translation model
- Qwen2.5-VL 3B
- EXAONE GGUF model
- llama.cpp `llama-cli`

네트워크 상태에 따라 오래 걸릴 수 있습니다. 중간에 실패하면 다시 같은 명령을 실행해도 됩니다.

## Common Hints

- `py` 명령을 찾지 못하면 Python 3.12를 설치하고 "Add python.exe to PATH" 옵션을 확인하세요.
- `git` 명령을 찾지 못하면 Git for Windows를 설치하세요.
- pip 설치가 실패하면 인터넷 연결, 회사/학교 프록시, 디스크 여유 공간을 확인하세요.
- `caption.txt` 오류가 나면 H/I 또는 `all` 실행에서 선택한 story 폴더에 `caption.txt`가 있는지 확인하세요. 현재 예제에서는 `--story 7`에 `caption.txt`가 있습니다.
- `llama-cli` 오류가 나면 `EXAONE_GGUF_SETUP.md`를 참고하거나 `AUTO_INSTALL_LLAMA_CPP=0`과 `LLAMA_CLI_PATH`를 설정하세요.

## Next Step

세팅이 끝나면 [USAGE.md](USAGE.md)의 실행 명령어를 사용하세요.
