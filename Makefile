# storypipe — 크로스플랫폼 단축 명령 (macOS/Linux). Windows는 setup.bat 사용.
# 사용 예:  make setup && make doctor && make demo
#          make run STORY=1 EXP=e
#          make run-all STORY=7
PY ?= ./.venv/bin/python
STORY ?= 7
EXP ?= e

.PHONY: setup setup-cuda doctor check run run-all demo clean clean-venv

setup:            ## venv 생성 + 패키지 설치(editable)
	./setup.sh

setup-cuda: setup ## (선택) NVIDIA CUDA용 torch 설치
	$(PY) -m pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu124

doctor:           ## 환경 점검 + 모델 자동 다운로드 + 스모크 추론
	$(PY) -m storypipe doctor

check:            ## 다운로드 없이 점검만
	$(PY) -m storypipe doctor --check-only

run:              ## 단일 실험: make run STORY=1 EXP=e
	$(PY) -m storypipe run $(STORY) $(EXP)

run-all:          ## 전체 실험: make run-all STORY=7
	$(PY) -m storypipe run-all $(STORY)

demo:             ## story 7 전체 실험 + 평가 대시보드
	$(PY) -m storypipe demo

clean:            ## 결과물/캐시 삭제 (.venv는 유지)
	rm -rf outputs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-venv:       ## 가상환경까지 삭제
	rm -rf .venv
