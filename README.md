# Sketch to Story Pipeline (storypipe)

아이 손그림을 순서대로 읽어 오픈소스 비전/언어 모델로 한국어 동화를 생성하고, 블라인드 평가까지 하는 파이프라인입니다.

흐름: **이미지 → 장면 인식(BLIP·OpenCLIP / Qwen2.5-VL) → 한국어 동화(EXAONE) → 평가**

## 설치 (한 번)

저장소를 clone한 뒤, 운영체제에 맞게 한 줄:

```bash
# macOS / Linux
./setup.sh
```

```powershell
# Windows
.\setup.bat
```

내부적으로 `.venv` 생성 → `pip install -e .`(llama-cpp-python 포함 모든 의존성 자동 설치)를 수행합니다.
Python 3.10–3.12, 디스크 30GB+ 권장. GPU 없이 CPU로 동작합니다.

## 4가지 명령

설치 후에는 `storypipe <명령>`을 씁니다. (무설치 폴백: `python run.py <명령>`)

| # | 명령 | 설명 |
| --- | --- | --- |
| 0 | `storypipe doctor` | **환경 점검 + 필요한 모델 자동 다운로드 + 스모크 추론.** 처음에 한 번 실행. |
| 1 | `storypipe run <story> <exp>` | 이야기 + 실험버전(a~j) 1개 실행. 예: `storypipe run 1 e` |
| 2 | `storypipe run-all <story>` | 이야기의 전체 실험(A~J) 실행. 예: `storypipe run-all 7` |
| 3 | `storypipe demo` | "7. 새로운 이야기" 전체 실험 후 블라인드 평가 대시보드까지 |

- `<story>`는 번호 권장(예: `1`, `7`) — 한글 폴더명 입력을 피합니다.
- 실험 H/I/J와 `demo`는 `caption.txt`가 있는 이야기가 필요합니다(예제는 story 7).
- 처음 `doctor` 실행 시 모델(~20GB)이 `.local_models/`로 자동 다운로드됩니다.

### 채점자/다른 컴퓨터에서 (2단계)

```bash
./setup.sh            # 1) 파이썬 의존성 자동 설치
storypipe doctor      # 2) 모델 자동 다운로드 + 점검   (Windows: python run.py doctor)
storypipe demo        #    전체 실행 + 평가
```

## 입력 / 출력

- `inputs/`: 이야기별 폴더에 순서 이미지(`1.png` … `10.png`). Git에 포함된 예제 제공.
- `outputs/`: 실험 결과 JSON/TXT/HTML + 평가 파일. Git에 올리지 않습니다.
- `.local_models/`: 모델 캐시(자동 생성, Git 제외).

## 더 보기

- [ARCHITECTURE.md](ARCHITECTURE.md) — 패키지 구조·의존성·실험 개요
- [CONTRIBUTORS.md](CONTRIBUTORS.md) — 3인 작업 분담
- [docs/ADVANCED.md](docs/ADVANCED.md) — 수동 설치·GPU/CUDA·환경변수·문제 해결
