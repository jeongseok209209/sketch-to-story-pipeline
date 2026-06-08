# 고급: NVIDIA GPU 가속 (선택)

이 파이프라인은 **CPU 기본**으로 동작하도록 구성되어 있으며, 이 경로가 재현의 기준입니다.
GPU 가속은 **선택 사항**으로, CPU 실행 시간이 길어 더 빠르게 확인하고자 하실 때에만 사용하시면 됩니다.

> ## 시작하기 전에
>
> - **기본 설정(CPU) 그대로 실행하시면 됩니다.** 아래 GPU 설정은 필요하지 않습니다.
>   `pip install -r requirements.txt` → `python run.py doctor` → `python run.py demo` 만으로
>   동작하도록 구성되었습니다.
> - 이 문서는 **CPU 실행이 느리게 느껴지실 때를 위한 선택 안내**입니다. GPU가 없으셔도
>   실행에는 전혀 지장이 없습니다.
>
GPU 가속은 **두 엔진을 각각** 설정해야 적용됩니다. 한쪽만 설정하면 절반만 가속됩니다.

- PyTorch (대상: BLIP / OpenCLIP / Qwen2.5-VL)
  CUDA 빌드 torch 설치 → 이후 자동으로 GPU 사용
- llama.cpp (대상: EXAONE GGUF)
  CUDA 휠 llama-cpp-python 설치 + LLAMA_GPU_LAYERS 설정

---

## 0. 드라이버가 지원하는 CUDA 버전 확인

```powershell
nvidia-smi
```

출력 오른쪽 위 헤더의 `CUDA Version: 12.x` 가 **드라이버가 지원하는 최대 CUDA 버전**입니다
(설치된 CUDA Toolkit이 아니라 드라이버의 상한값입니다). 이후 단계에서 설치할 휠의 `cuXXX`
태그는 **이 값 이하**여야 합니다. 예를 들어 `CUDA Version: 12.6` 이면 `cu124`(또는 그 이하)
휠을 사용하시면 됩니다.

> `nvidia-smi` 명령이 없거나 NVIDIA GPU가 없는 환경이라면 GPU 가속을 사용할 수 없습니다.
> 이 경우 CPU 기본 설정 그대로 사용하시면 됩니다.

---

## 1. PyTorch를 CUDA 빌드로 설치 (비전 모델용)

현재 설치된 CPU 빌드를 CUDA 빌드로 교체합니다. 아래 `cu124` 부분은 0단계에서 확인하신
버전에 맞추면 됩니다.

```powershell
pip uninstall -y torch
pip install "torch>=2.6,<2.8" --index-url https://download.pytorch.org/whl/cu124
```

- `--index-url` 은 PyPI 대신 PyTorch 공식 인덱스에서 내려받도록 지정합니다.
- 버전 범위(`>=2.6,<2.8`)는 `requirements.txt` 의 고정 범위와 동일하게 유지합니다.
- PyTorch는 별도 코드 변경 없이 자동으로 GPU를 사용합니다([common.py](common.py)의
  `get_device()` 가 CUDA 가용 시 자동 선택하고, 없으면 CPU로 되돌아갑니다).

---

## 2. llama-cpp-python을 CUDA 휠로 설치 (EXAONE용)

이 단계를 생략하면 다음 3단계에서 `LLAMA_GPU_LAYERS` 를 설정하셔도 EXAONE은 계속 CPU로
동작합니다(CPU 휠에는 CUDA 코드가 포함되어 있지 않기 때문입니다). EXAONE 생성이 가장 큰
실행 시간을 차지하므로, 체감 속도를 위해서는 이 단계가 특히 중요합니다.

```powershell
pip install --force-reinstall --no-cache-dir --prefer-binary "llama-cpp-python>=0.3.2,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

- `--force-reinstall --no-cache-dir` 는 이미 설치된 CPU 휠을 확실히 교체합니다.
- `cu124` 태그는 1단계의 torch와 가능한 한 동일하게 맞추는 것이 좋습니다.
- 제공되는 태그 목록은 <https://abetlen.github.io/llama-cpp-python/whl/> 에서 확인하실 수
  있습니다. 드라이버 지원 버전과 정확히 일치하는 태그가 없으면 가장 가까운 하위 버전을
  사용하면 됩니다.

---

## 3. 환경변수로 GPU 오프로드 활성화 (EXAONE)

llama.cpp는 휠 설치만으로는 자동 가속되지 않으며, 오프로드할 레이어 수를 지정해야 합니다.

PowerShell (현재 세션):
```powershell
$env:LLAMA_GPU_LAYERS = "999"
```

bash / zsh:
```bash
export LLAMA_GPU_LAYERS=999
```

- `999` 는 "전체 레이어를 GPU로 오프로드"한다는 의미입니다(1.2B 모델이라 실제 레이어 수보다
  크면 전부 올라갑니다).
- 반드시 **양수**로 설정해야 합니다. 코드가 `max(int(값), 0)` 으로 처리하므로 `-1` 등 음수는
  0(CPU)으로 무시됩니다([common.py](common.py)의 `configured_llama_gpu_layers()`).
- 위 `$env:` 방식은 **해당 터미널 세션에만** 적용됩니다. 영구 적용을 원하시면 시스템
  환경변수로 등록하면 됩니다.

---

## 4. GPU 적용 여부 확인

**(a) PyTorch 확인:**
```powershell
python -c "import torch; print('torch CUDA available:', torch.cuda.is_available())"
```
`True` 가 출력되어야 합니다.

**(b) 실행 로그 확인** — `python run.py doctor`(또는 `run` / `demo`) 실행 시 다음 로그가
보이면 정상입니다:
```
[runtime] CUDA ready: <GPU 이름>
[runtime] EXAONE llama.cpp mode: gpu (gpu_layers=999)
```

아래 로그가 보인다면 아직 GPU가 적용되지 않은 상태입니다:
```
[runtime] NVIDIA GPU detected but PyTorch CUDA unavailable; ...   ← 1단계 미완료
[runtime] EXAONE llama.cpp mode: cpu (gpu_layers=0)               ← 2단계 또는 3단계 미완료
```

---

## 5. CPU로 되돌리기

GPU 설정을 되돌려 프로젝트 기본 구성인 CPU로 복구하는 방법입니다.

```powershell
pip install --force-reinstall "torch>=2.6,<2.8" --index-url https://download.pytorch.org/whl/cpu
pip install --force-reinstall --no-cache-dir --prefer-binary "llama-cpp-python>=0.3.2,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
$env:LLAMA_GPU_LAYERS = "0"   # 또는 환경변수 자체를 제거
```

확인 방법은 4단계와 동일하며, 이번에는 `torch CUDA available: False` /
`EXAONE llama.cpp mode: cpu (gpu_layers=0)` 가 출력되면 정상입니다.

---

## 6. 문제 해결

**`torch.cuda.is_available()` 가 계속 False인 경우**
- CPU 빌드 torch가 남아 있을 수 있습니다. `pip uninstall -y torch` 후 1단계를 다시
  진행하면 됩니다.
- `nvidia-smi` 가 정상 동작하는지(드라이버 설치 여부), 설치하신 `cuXXX` 태그가 드라이버
  지원 버전 **이하**인지 확인이 필요합니다.

**EXAONE이 계속 `mode: cpu` 인 경우**
- 2단계(CUDA 휠 재설치)를 생략했거나, 3단계 `LLAMA_GPU_LAYERS` 가 설정되지 않은 경우입니다.
- 새 터미널을 여시면 `$env:LLAMA_GPU_LAYERS` 가 초기화되므로 다시 설정하면 됩니다.

**GPU 메모리 부족(OOM) 또는 로드 실패**
- 전체 레이어를 올리지 못하는 경우입니다. `LLAMA_GPU_LAYERS` 값을 낮추면 **일부만 오프로드**
  됩니다(예: `20`). 나머지 레이어는 CPU에서 처리됩니다.

**`llama-cpp-python` 이 소스 빌드로 진행되거나 해당 `cuXXX` 휠이 없는 경우**
- 그 CUDA 태그의 프리빌트 휠이 제공되지 않을 수 있습니다. 0단계에서 확인한 범위 안에서
  다른 `cuXXX` 태그를 시도하거나, 프리빌트가 제공되는 가장 가까운 하위 버전을 사용하면
  됩니다.
- 일반적인 `llama-cpp-python` 설치 문제(Windows 빌드 오류 등)는
  [README의 "문제 해결"](README.md#문제-해결) 항목을 참고하시면 됩니다.
