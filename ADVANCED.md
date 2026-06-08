# 고급: PyTorch NVIDIA GPU 가속 (선택)

이 파이프라인은 **CPU 기본**으로 동작하도록 구성되어 있으며, 해당 경로를 재현 기준으로 삼습니다.
GPU 가속은 **선택 사항**이며, CPU 실행 시간이 길어 실행 시간을 단축하고자 할 때 사용할 수 있습니다.

> ## 시작하기 전에
>
> - **기본 설정(CPU) 그대로 실행하셔도 됩니다.** 아래 GPU 설정은 필수 사항이 아닙니다.
>   `pip install -r requirements.txt` -> `python run.py doctor` -> `python run.py demo` 만으로
>   동작하도록 구성되었습니다.
> - 이 문서는 **BLIP / OpenCLIP / Qwen2.5-VL 등 PyTorch 기반 비전 모델**을 GPU로
>   실행하고자 할 때 참고하는 선택 안내입니다. NVIDIA GPU가 없는 환경에서도 CPU로 실행할 수 있습니다.

PyTorch CUDA 빌드가 설치되면 별도의 코드 변경 없이 CUDA를 자동으로 사용합니다.

대상:

- BLIP captioning
- BLIP-VQA
- OpenCLIP
- Qwen2.5-VL

---

## 0. 드라이버가 지원하는 CUDA 버전 확인

```powershell
nvidia-smi
```

출력 오른쪽 위 헤더의 `CUDA Version: 12.x` 가 **드라이버가 지원하는 최대 CUDA 버전**입니다
(설치된 CUDA Toolkit이 아니라 드라이버의 상한값입니다). 이후 단계에서 설치할 휠의 `cuXXX`
태그는 **이 값 이하**여야 합니다. 예를 들어 `CUDA Version: 12.6` 이면 `cu124`(또는 그 이하)
휠을 사용하시면 됩니다.

> `nvidia-smi` 명령이 없거나 NVIDIA GPU가 없는 환경에서는 GPU 가속을 사용할 수 없습니다.
> 이 경우 CPU 기본 설정을 그대로 사용하시면 됩니다.

---

## 1. PyTorch를 CUDA 빌드로 설치

현재 설치된 CPU 빌드의 PyTorch를 CUDA 빌드로 교체합니다. 아래 `cu124` 부분은 0단계에서
확인한 드라이버 지원 버전에 맞추어 선택합니다.

```powershell
pip uninstall -y torch
pip install "torch>=2.6,<2.8" --index-url https://download.pytorch.org/whl/cu124
```

- `--index-url` 은 PyPI 대신 PyTorch 공식 인덱스에서 내려받도록 지정합니다.
- 버전 범위(`>=2.6,<2.8`)는 `requirements.txt` 의 고정 범위와 동일하게 유지합니다.
- PyTorch는 별도의 코드 변경 없이 GPU를 사용합니다([common.py](common.py)의
  `get_device()` 가 CUDA 가용 시 자동 선택하며, CUDA를 사용할 수 없는 경우 CPU로 실행합니다).

---

## 2. GPU 적용 여부 확인

**(a) PyTorch 확인:**

```powershell
python -c "import torch; print('torch CUDA available:', torch.cuda.is_available())"
```

`True` 가 출력되면 PyTorch CUDA 빌드가 정상적으로 인식된 것입니다.

**(b) 실행 로그 확인** - `python run.py doctor`(또는 `run` / `demo`) 실행 시 다음 로그가
표시되면 정상입니다:

```text
[runtime] CUDA ready: <GPU 이름>
```

아래 로그가 표시되는 경우에는 아직 PyTorch GPU 가속이 적용되지 않은 상태입니다:

```text
[runtime] NVIDIA GPU detected but PyTorch CUDA unavailable; ...
```

---

## 3. CPU로 되돌리기

GPU 설정을 되돌려 프로젝트 기본 구성인 CPU 실행 환경으로 복구하는 방법입니다.

```powershell
pip install --force-reinstall "torch>=2.6,<2.8" --index-url https://download.pytorch.org/whl/cpu
```

확인 방법은 2단계와 동일하며, 이 경우에는 `torch CUDA available: False` 가 출력되면 정상입니다.

---

## 4. 문제 해결

**`torch.cuda.is_available()` 가 계속 False인 경우**

- CPU 빌드 torch가 남아 있을 수 있습니다. `pip uninstall -y torch` 후 1단계를 다시
  진행합니다.
- `nvidia-smi` 가 정상 동작하는지(드라이버 설치 여부), 설치하신 `cuXXX` 태그가 드라이버
  지원 버전 **이하**인지 확인이 필요합니다.

**GPU 메모리 부족(OOM) 또는 로드 실패**

- 비전 모델을 GPU 메모리에 올리지 못하는 경우입니다. 다른 GPU 작업을 종료한 뒤 다시 실행하거나,
  CPU 기본 경로로 되돌려 실행합니다.
