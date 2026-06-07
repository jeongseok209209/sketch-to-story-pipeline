"""[공통 토대] 모델 ID, 경로, 버전 등 프로젝트 전역 상수.

세 도메인(vision / story / pipeline)이 공유하는 단일 진실 공급원(single source of truth).
모델 리비전을 고정하고 싶으면 ``MODEL_REVISIONS``에 정확한 revision 해시를 넣거나
``HF_REVISION_<별칭>`` 환경변수로 덮어쓸 수 있다(재현성용).
"""

from __future__ import annotations

from pathlib import Path

# ── 모델 ID ────────────────────────────────────────────────────────────────
BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"
GPT2_MODEL = "gpt2-medium"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
OPENCLIP_MODEL = "ViT-H-14"
OPENCLIP_PRETRAINED = "laion2b_s32b_b79k"
OPENCLIP_HF_CACHE_REPO = "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
OPENCLIP_WEIGHTS_FILENAME = "open_clip_model.safetensors"
EXAONE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
EXAONE_GGUF_REPO_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"
EXAONE_GGUF_FILENAME = "EXAONE-4.0-1.2B-IQ4_XS.gguf"
QWEN25_VL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
VISION_MODEL_ID = QWEN25_VL_MODEL

# 재현성: 비우면 최신(main). 정확한 revision 해시를 넣으면 그 버전으로 고정 다운로드.
MODEL_REVISIONS: dict[str, str] = {}

HF_PREFLIGHT_IGNORE_PATTERNS = (
    "onnx/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "tf_model.*",
    "flax_model.*",
    "rust_model.*",
)

# ── 경로 ──────────────────────────────────────────────────────────────────
# config.py = storypipe/common/config.py → parents[2] = 저장소 루트.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
COMMON_OUTPUT_DIR = OUTPUT_ROOT / "qwen25_vl_3b_story"
SHARED_DIR = COMMON_OUTPUT_DIR / "scene_descriptions"
RESIZED_DIR = COMMON_OUTPUT_DIR / "_resized_input"
LOCAL_HF_MODEL_DIR = PROJECT_ROOT / ".local_models" / "huggingface"
DEFAULT_EXAONE_GGUF_PATH = str(PROJECT_ROOT / ".local_models" / "exaone" / EXAONE_GGUF_FILENAME)

# ── 입력 규칙 ──────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
STORY_CAPTION_FILENAME = "caption.txt"
COLLAGE_FILENAME = "collage_2x5_scene_order.png"

# ── 런타임 호환성(진단/안내 메시지용 참고값) ──────────────────────────────
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
PYTORCH_CUDA_PACKAGES = ("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0")
TRANSFORMERS_VERSION_SPEC = "transformers>=4.54.0,<5"
TORCH_MIN_SAFE_VERSION = (2, 6)

# ── 버전 태그 ──────────────────────────────────────────────────────────────
PIPELINE_VERSION = "2026.06.07-storypipe"
STEP_LOG_VERSION = "step-log-v1"


def experiment_dirs(output_root: Path) -> dict[str, Path]:
    """실험 키(c~j) → 출력 디렉터리 매핑. vision/pipeline이 공유하므로 common에 둔다."""
    return {key: output_root / key.upper() for key in ("c", "d", "e", "f", "g", "h", "i", "j")}


def hf_revision(model_id: str) -> str | None:
    """모델 ID에 고정된 revision을 반환(없으면 None=최신)."""
    import os

    env_key = "HF_REVISION_" + model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    return os.environ.get(env_key) or MODEL_REVISIONS.get(model_id)
