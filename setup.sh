#!/usr/bin/env bash
# storypipe macOS/Linux 셋업: venv 생성 → pip install -e . → 점검
# (모델 다운로드는 설치 후 `storypipe doctor`가 담당)
set -euo pipefail
cd "$(dirname "$0")"

echo "storypipe setup (macOS/Linux)"

# Python 3.10-3.12 찾기
PYBIN=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")"
    case "$ver" in
      3.10|3.11|3.12) PYBIN="$cand"; break;;
    esac
  fi
done
if [ -z "$PYBIN" ]; then
  echo "ERROR: Python 3.10-3.12 가 필요합니다. 설치 후 다시 실행하세요." >&2
  echo "  macOS:  brew install python@3.12" >&2
  echo "  Linux:  sudo apt install python3.12 python3.12-venv" >&2
  exit 1
fi
echo "Using $PYBIN ($("$PYBIN" --version))"

[ -d .venv ] || "$PYBIN" -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e .

echo ""
echo "셋업 완료. 다음으로 환경 점검 + 모델 자동 다운로드:"
echo "  ./.venv/bin/storypipe doctor"
echo "또는 무설치 폴백:  ./.venv/bin/python run.py doctor"
