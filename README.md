# Sketch to Story Pipeline

Open-source vision and language models turn ordered child drawings into Korean fairy-tale stories.

## Experiments

| Experiment | Vision input | Language model | Behavior |
| --- | --- | --- | --- |
| A | BLIP/OpenCLIP single image | GPT-2+NLLB or EXAONE HF | Single-image baseline |
| B | BLIP/OpenCLIP ordered scene records | EXAONE GGUF via llama.cpp | Sequence story from ordered scenes |
| C | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Simple whole-story prompt |
| D | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Structure, plan, draft, self-check in one prompt |
| E | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Global continuity and emotion-first prompt |
| F | Qwen2.5-VL scene JSON | EXAONE GGUF via llama.cpp | Whole-scene overview plus previous/current/next scene windows |

B, C, D, E, and F are independent experiment paths. C/D/E/F share Qwen vision recognition, but their EXAONE prompts do not consume another experiment's story output.

## Quick Run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe run.py all
```

If `.venv` does not exist yet:

```powershell
& "C:\Users\AEM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py all
```

PowerShell script activation is not required. You can call `.\.venv\Scripts\python.exe` directly.

## Automatic Runtime Setup

At startup, `run.py` performs a runtime preflight:

- downloads Hugging Face models into `.local_models/huggingface`
- downloads the EXAONE GGUF file into `.local_models/exaone`
- prepares `llama-cli.exe` under `.local_tools/llama.cpp/build/bin` when possible
- detects NVIDIA CUDA support and installs CUDA PyTorch wheels when appropriate
- uses CPU mode on machines without NVIDIA CUDA

Default model paths:

```powershell
$env:EXAONE_GGUF_MODEL_PATH=".local_models\exaone\EXAONE-4.0-1.2B-IQ4_XS.gguf"
$env:LLAMA_CLI_PATH=".local_tools\llama.cpp\build\bin\llama-cli.exe"
```

Useful overrides:

```powershell
$env:LLAMA_GPU_LAYERS="0"       # force CPU llama.cpp
$env:LLAMA_CUDA_RETRY_CPU="0"   # fail instead of retrying CPU after GPU llama.cpp error
$env:AUTO_INSTALL_LLAMA_CPP="0" # skip llama.cpp auto preparation
```

## Output Policy

The pipeline does not replace failed model output with a hardcoded story or canned structure. If EXAONE returns malformed JSON, the code asks EXAONE to repair the JSON once. If that also fails, execution raises an error so invalid generated output is not saved as a real result.
