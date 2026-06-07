"""진입점: `python run.py <doctor|run|run-all|demo>`.

먼저 의존성 설치:  pip install -r requirements.txt
그다음:
    python run.py doctor          # 환경 점검 + 모델 자동 다운로드
    python run.py run 1 e         # 이야기 1, 실험 e
    python run.py run-all 7       # 이야기 7 전체 실험
    python run.py demo            # 이야기 7 전체 실험 + 평가 대시보드
"""

from pipeline import main

if __name__ == "__main__":
    main()
