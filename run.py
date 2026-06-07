"""무설치 폴백 진입점: `python run.py <doctor|run|run-all|demo>`.

저장소를 clone한 뒤 `pip install -e .`로 설치하면 `storypipe <명령>`을 바로 쓸 수 있다.
설치 없이도 저장소 루트에서 이 shim으로 동일하게 실행할 수 있다:

    python run.py doctor
    python run.py run 1 e
    python run.py run-all 7
    python run.py demo
"""

from storypipe.pipeline.cli import main

if __name__ == "__main__":
    main()
