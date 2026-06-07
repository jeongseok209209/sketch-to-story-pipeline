# Contributors / 작업 분담

이 프로젝트는 3명이 파이프라인 단계별로 도메인을 나누어 작업했습니다. 각 담당은 아래 패키지를 소유합니다.
(코드 각 모듈 상단 docstring에도 `[담당 N · 도메인]` 표기가 있습니다.)

| 담당 | 도메인 | 소유 패키지 | 핵심 책임 |
| --- | --- | --- | --- |
| **담당 1** | 비전 / 장면 인식 | `storypipe/vision/` | BLIP 캡션·BLIP-VQA·OpenCLIP 인식(실험 A), Qwen2.5-VL 장면/콜라주 추출(실험 C~J), 비전 모델 로더 |
| **담당 2** | 스토리 생성 (LLM) | `storypipe/story/` | EXAONE GGUF 런타임(llama-cpp-python), 구조화 플랜, GPT-2/NLLB 베이스라인, 실험 C~J 프롬프트·품질 게이트·빌더 |
| **담당 3** | 파이프라인 · 평가 | `storypipe/pipeline/`, `storypipe/evaluation/` | 4-커맨드 CLI·`doctor`, 실험 A/B 오케스트레이션, C~J 통합 러너, 출력 작성, 블라인드 평가 대시보드 |
| (공유) | 공통 토대 | `storypipe/common/` | 설정·경로 상수, 런타임/디바이스 감지, 로깅, 모델 다운로드, 이미지/IO/JSON 파싱 유틸 |

## 데이터 흐름과 담당 경계

```
이미지 ──[담당1 vision]──> 장면 JSON ──[담당2 story]──> 한국어 동화 ──[담당3 pipeline]──> 저장/평가
                                  (scenes 인자 전달; story는 vision을 import하지 않음)
```

- 담당 1과 담당 2는 서로 직접 의존하지 않습니다. 통합은 담당 3(`pipeline/runner.py`)이 담당합니다.
- 공통 토대(`common/`)는 세 담당이 함께 사용합니다.

자세한 모듈 지도와 의존성 레이어는 [ARCHITECTURE.md](ARCHITECTURE.md) 참고.
