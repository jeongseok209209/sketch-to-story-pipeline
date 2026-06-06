# Usage Guide

모든 명령은 저장소 루트에서 PowerShell로 실행합니다.

## Check

다운로드 없이 환경만 점검합니다.

```powershell
.\.venv\Scripts\python.exe run.py check
```

특정 입력 루트를 점검하려면:

```powershell
.\.venv\Scripts\python.exe run.py check --input-dir inputs
```

## Story Selection

`inputs/`에는 여러 이야기 폴더가 있습니다. `--story`에는 폴더 번호나 폴더명을 넣을 수 있습니다.

권장:

```powershell
--story 1
--story 7
```

폴더명을 직접 사용할 수도 있지만 한글과 공백이 있으므로 따옴표가 필요합니다.

```powershell
--story "7. 새로운 이야기"
```

H/I 실험과 `all` 실행은 `caption.txt`가 필요한 단계가 포함됩니다. 현재 예제에서는 `inputs/7. 새로운 이야기/caption.txt`가 있습니다.

## Smoke Test

가장 작은 실행 테스트입니다.

```powershell
.\.venv\Scripts\python.exe run.py a --story 1 --image 1 --story-max-new-tokens 20 --output-dir outputs\smoke_A
```

## Experiment A

단일 이미지:

```powershell
.\.venv\Scripts\python.exe run.py a --story 1 --image 1 --output-dir outputs\A
```

한 이야기 폴더의 모든 이미지:

```powershell
.\.venv\Scripts\python.exe run.py a --story 1 --batch --output-dir outputs\A
```

EXAONE HF backend를 쓰려면:

```powershell
.\.venv\Scripts\python.exe run.py a --story 1 --image 1 --story-backend exaone --output-dir outputs\A_exaone
```

## Experiment B

순서가 있는 이미지 전체를 BLIP/OpenCLIP으로 읽고 EXAONE GGUF로 연결된 이야기를 만듭니다.

```powershell
.\.venv\Scripts\python.exe run.py b --story 1 --output-dir outputs\B
```

## Experiments C-I

Qwen2.5-VL로 장면 JSON을 만들고 EXAONE GGUF로 이야기를 생성합니다.

```powershell
.\.venv\Scripts\python.exe run.py c --story 1 --output-root outputs
.\.venv\Scripts\python.exe run.py d --story 1 --output-root outputs
.\.venv\Scripts\python.exe run.py e --story 1 --output-root outputs
.\.venv\Scripts\python.exe run.py f --story 1 --output-root outputs
.\.venv\Scripts\python.exe run.py g --story 1 --output-root outputs
```

H/I는 `caption.txt`가 있는 story를 사용하세요.

```powershell
.\.venv\Scripts\python.exe run.py h --story 7 --output-root outputs
.\.venv\Scripts\python.exe run.py i --story 7 --output-root outputs
```

## All Experiments

전체 A-I를 실행합니다. H/I 때문에 `caption.txt`가 있는 story를 권장합니다.

```powershell
.\.venv\Scripts\python.exe run.py all --story 7
```

## All Experiments + Evaluation Dashboard

전체 실행 후 성공 결과만 모아 블라인드 평가 대시보드를 실행합니다.

```powershell
.\.venv\Scripts\python.exe run.py all-evaluate --story 7 --port 8501
```

브라우저에서 다음 주소를 엽니다.

```text
http://localhost:8501
```

## Evaluation Dashboard Only

이미 결과가 있을 때 대시보드만 실행합니다.

```powershell
.\.venv\Scripts\python.exe run.py evaluate --port 8501
```

## Useful Environment Variables

CPU로 강제 실행:

```powershell
$env:LLAMA_GPU_LAYERS="0"
$env:AUTO_INSTALL_TORCH_CUDA="0"
```

llama.cpp 자동 준비 끄기:

```powershell
$env:AUTO_INSTALL_LLAMA_CPP="0"
```

직접 받은 EXAONE GGUF 모델 사용:

```powershell
$env:EXAONE_GGUF_MODEL_PATH="C:\path\to\EXAONE-4.0-1.2B-IQ4_XS.gguf"
```

직접 준비한 llama-cli 사용:

```powershell
$env:LLAMA_CLI_PATH="C:\path\to\llama-cli.exe"
```
