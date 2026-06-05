# Automatic EXAONE GGUF setup

B/C/D/E/F use EXAONE GGUF through llama.cpp by default. On first use, the pipeline automatically prepares:

```text
.local_models/exaone/EXAONE-4.0-1.2B-IQ4_XS.gguf
.local_tools/llama.cpp/build/bin/llama-cli.exe
```

Default EXAONE GGUF source:

```text
LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF
EXAONE-4.0-1.2B-IQ4_XS.gguf
```

Run:

```powershell
python run.py all
```

For `python run.py all`, the pipeline now runs preflight before any story generation:

```text
1. Prepare Python runtime compatibility.
2. Download/verify BLIP, BLIP-VQA, GPT-2, NLLB, OpenCLIP, and Qwen.
3. Download/verify EXAONE GGUF.
4. Prepare and execute-check llama-cli.
5. Start A/B/C/D/E/F generation only after preflight succeeds.
```

Optional model overrides:

```powershell
$env:EXAONE_GGUF_REPO_ID="LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"
$env:EXAONE_GGUF_FILENAME="EXAONE-4.0-1.2B-IQ4_XS.gguf"
$env:EXAONE_GGUF_MODEL_PATH="C:\path\to\already-downloaded-model.gguf"
```

Optional llama.cpp overrides:

```powershell
$env:LLAMA_CLI_PATH="C:\path\to\llama-cli.exe"
$env:AUTO_INSTALL_LLAMA_CPP="0" # disable automatic llama.cpp install/build
```

CUDA-related options:

```powershell
$env:LLAMA_GPU_LAYERS="0"       # force CPU llama.cpp
$env:LLAMA_CUDA_RETRY_CPU="0"   # disable GPU-to-CPU retry
```

The `.gguf` model and `llama-cli.exe` binary are intentionally ignored by Git because they are local large/generated files. They are restored by the first run when network access and platform tooling are available.
