"""[공통 토대] 이미지 로드/정규화 유틸."""

from __future__ import annotations

from typing import Any


def load_and_normalize_image(image_path: str) -> Any:
    """이미지를 로드해 흰 배경 정규화 + 대비 보정한다(손그림 선을 또렷하게)."""
    from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(image_path)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to load image: {image_path}") from exc

    # 투명 배경 그림은 흰 배경 위에 합성해 RGB 입력으로 안정화한다.
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")

    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    return image


def resize_square(image: Any, size: int) -> Any:
    """중앙 crop 기반 정사각형 캔버스로 리사이즈한다."""
    from PIL import Image, ImageOps

    return ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
