# Sketch to Story Pipeline

아이 손그림 이미지를 순서대로 읽고, 오픈소스 비전/언어 모델로 한국어 동화를 생성하는 실험용 파이프라인입니다.

이 저장소는 GitHub에서 clone한 뒤 바로 세팅할 수 있게 구성되어 있습니다. `inputs/`의 예제 이미지는 Git에 포함되며, 모델 파일과 실행 결과는 각 컴퓨터에서 새로 만들어집니다.

## What Is Included

Git에 포함되는 파일:

- Python 실행 코드: `run.py`, `pipeline_a.py`, `generators.py`, `utils.py`, `vision.py`, `run_experiments_cd_qwen3b.py`
- 문서: `README.md`, `SETUP.md`, `USAGE.md`
- 의존성 목록: `requirements.txt`
- Windows 세팅 도우미: `setup.bat`, `setup_windows.ps1`
- 예제 입력 데이터: `inputs/`

Git에 포함하지 않는 로컬 생성물:

- `.venv/`: Python 가상환경
- `.local_models/`: Hugging Face 모델과 EXAONE GGUF 모델 캐시
- `.local_tools/`: llama.cpp 실행 파일과 빌드 산출물
- `outputs/`: 실험 결과와 평가 결과

## Quick Start

Windows PowerShell에서:

```powershell
git clone https://github.com/jeongseok209209/sketch-to-story-pipeline.git
cd sketch-to-story-pipeline
.\setup.bat
```

VS Code/Cursor에서 저장소 폴더를 연 뒤 터미널에 `.\setup.bat` 한 줄만 입력해도 같은 세팅이 진행됩니다.
Python이 없는 컴퓨터에서는 `setup.bat`가 `winget`으로 Python 3.12 설치를 먼저 시도합니다.

가장 작은 실행 테스트:

```powershell
.\.venv\Scripts\python.exe run.py a --story 1 --image 1 --story-max-new-tokens 20 --output-dir outputs\smoke_A
```

전체 실험 실행 예시:

```powershell
.\.venv\Scripts\python.exe run.py all --story 7
```

처음 실제 실험을 실행하면 모델 파일이 `.local_models/` 아래로 다운로드됩니다. 디스크 여유 공간은 30GB 이상을 권장합니다.

## Inputs and Outputs

`inputs/`에는 기본 예제 이야기 이미지가 들어 있습니다. 이야기 폴더가 여러 개 있으므로 실행할 때는 `--story 1`, `--story 7`처럼 숫자로 선택하는 방식을 권장합니다.

`outputs/`에는 실행 결과 JSON, 텍스트, HTML, 평가 파일이 저장됩니다. 이 폴더는 Git에 올리지 않습니다.

`.local_models/`와 `.local_tools/`는 첫 실행 중 자동으로 준비되는 로컬 캐시입니다. 다른 컴퓨터에서 clone하면 처음에는 없어도 정상입니다.

## Experiments

| Experiment | Vision input | Language model | Behavior |
| --- | --- | --- | --- |
| A | BLIP/OpenCLIP single image | GPT-2+NLLB or EXAONE HF | Single-image baseline |
| B | BLIP/OpenCLIP ordered scene records | EXAONE GGUF via llama.cpp | Sequence story from ordered scenes |
| C | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Simple whole-story prompt |
| D | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Structure, plan, draft, self-check in one prompt |
| E | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Global continuity and emotion-first prompt |
| F | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Whole-scene overview plus previous/current/next scene windows |
| G/H/I | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Additional prompt and caption-guided variants |

H와 I는 선택한 이야기 폴더 안에 `caption.txt`가 있어야 합니다.

## More Docs

- [SETUP.md](SETUP.md): Windows 설치와 환경 점검
- [USAGE.md](USAGE.md): 실험별 실행 명령어
- [EXAONE_GGUF_SETUP.md](EXAONE_GGUF_SETUP.md): EXAONE GGUF와 llama.cpp 세부 설정
