# Colab 실행 안내

이 브랜치는 Google Colab에서 바로 실행하기 위한 버전입니다.

## 1. Colab 런타임

Colab 메뉴에서 `런타임 > 런타임 유형 변경 > GPU`를 선택하세요.
C, D, E, F는 Qwen2.5-VL-3B를 사용하므로 CPU 실행은 권장하지 않습니다.

## 2. 프로젝트 열기

Colab에서 GitHub 저장소를 clone한 뒤 `sketch_to_story_colab.ipynb`를 실행합니다.

```python
!git clone -b Colab git@github.com:jeongseok209209/sketch-to-story-pipeline.git
%cd sketch-to-story-pipeline
```

SSH 인증이 없는 Colab에서는 GitHub의 HTTPS clone 주소를 사용하세요. 단, private 저장소는 인증 없는 HTTPS clone이 실패하므로 GitHub 토큰 등 인증된 clone 주소가 필요합니다. 노트북에서 `ERROR: Could not open requirements file: [Errno 2] No such file or directory: 'requirements-colab.txt'` 또는 `Project root not found`가 나오면 저장소 루트가 아닌 `/content` 같은 다른 폴더에서 셀을 실행한 것입니다. 먼저 `Colab` 브랜치를 clone한 뒤 `%cd sketch-to-story-pipeline`을 실행하고, 의존성 설치 셀부터 순서대로 다시 실행하세요. 이어서 `FileNotFoundError: ... '/content/inputs'`가 나오면 이야기 선택 셀이 저장소 루트를 못 찾은 상태이므로 동일하게 `%cd /content/sketch-to-story-pipeline` 후 다시 실행하세요.

## 3. 그림 폴더 구조

그림은 `inputs/` 아래 이야기별 폴더에 들어 있습니다.

```text
inputs/
├── 1. 토끼와 거북이/1.png ... 10.png
├── 2. 여우와 두루미/1.png ... 10.png
├── 3. 개미와 베짱이/1.png ... 10.png
├── 4. 금도끼 은도끼/1.png ... 10.png
├── 5. 해와 달이 된 오누이/1.png ... 10.png
└── 6. 은혜 갚은 까치/1.png ... 10.png
```

노트북에서 이야기 폴더 하나를 선택하면 그 폴더의 `1.png`부터 `10.png`까지를 입력으로 사용합니다.

## 4. 실행 방식

노트북은 A, B, C, D, E, F를 각각 별도 셀로 실행합니다.
C-F는 묶음 실행하지 않습니다. 필요한 실험만 따로 실행하세요.

결과는 다음 위치에 저장됩니다.

```text
outputs/<선택한_이야기명>/A
outputs/<선택한_이야기명>/B
outputs/<선택한_이야기명>/C
outputs/<선택한_이야기명>/D
outputs/<선택한_이야기명>/E
outputs/<선택한_이야기명>/F
outputs/<선택한_이야기명>/feedback
```

## 5. 피드백

노트북의 `피드백 기록` 섹션에서 장면별 피드백과 전체 이야기 피드백을 저장할 수 있습니다.
피드백은 재생성에 사용하지 않고 평가/기록용으로만 저장됩니다.

저장 파일:

```text
outputs/<선택한_이야기명>/feedback/feedback_records.jsonl
outputs/<선택한_이야기명>/feedback/feedback_summary.json
```
