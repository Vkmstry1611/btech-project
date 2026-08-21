"""Optional preprocessing steps applied after PDF page rendering."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import pvariance
from typing import Dict, List, Optional

from PIL import Image, ImageFilter, ImageOps

from catalogbank_ocr.preprocessing.pdf_to_image import pdf_first_page_to_image

logger = logging.getLogger(__name__)


@dataclass
class ImagePreprocessingConfig:
    """Configurable optional preprocessing for rendered pages."""

    dpi: int = 200
    first_page_only: bool = True
    deskew: bool = False
    denoise: bool = False
    enhance_resolution: bool = False
    resolution_scale: float = 1.5
    deskew_max_angle: float = 5.0
    deskew_step: float = 0.5
    threshold: int = 230
    save_intermediate_steps: bool = True


@dataclass
class ImagePreprocessingResult:
    """Result metadata for image preprocessing."""

    source_pdf: Path
    rendered_image: Path
    processed_image: Path
    steps: List[str] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_pdf": str(self.source_pdf),
            "rendered_image": str(self.rendered_image),
            "processed_image": str(self.processed_image),
            "steps": list(self.steps),
            "metadata": dict(self.metadata),
        }


def _score_deskew_candidate(image: Image.Image, threshold: int) -> float:
    gray = image.convert("L")
    bw = gray.point(lambda p: 0 if p < threshold else 255, mode="1")
    pixels = bw.load()
    width, height = bw.size
    row_counts = []
    for y in range(height):
        count = 0
        for x in range(width):
            if pixels[x, y] == 0:
                count += 1
        row_counts.append(count)
    if not row_counts:
        return 0.0
    return float(pvariance(row_counts))


def estimate_skew_angle(image: Image.Image, max_angle: float = 5.0, step: float = 0.5, threshold: int = 230) -> float:
    """Estimate a small skew angle using projection variance.

    The method is deterministic and only searches a narrow angle band.
    If no strong signal is found, zero is returned.
    """

    best_angle = 0.0
    best_score = -1.0
    candidate = -max_angle
    while candidate <= max_angle + 1e-9:
        rotated = image.rotate(candidate, expand=True, fillcolor=255)
        score = _score_deskew_candidate(rotated, threshold)
        if score > best_score:
            best_score = score
            best_angle = candidate
        candidate += step
    logger.debug("Estimated deskew angle %.2f with score %.4f", best_angle, best_score)
    return best_angle


def apply_optional_preprocessing(image: Image.Image, config: ImagePreprocessingConfig) -> Image.Image:
    """Apply configured optional preprocessing steps to a page image."""

    result = image.convert("RGB")

    if config.denoise:
        result = result.filter(ImageFilter.MedianFilter(size=3))

    if config.enhance_resolution:
        scale = max(1.0, float(config.resolution_scale))
        new_size = (max(1, int(round(result.width * scale))), max(1, int(round(result.height * scale))))
        result = result.resize(new_size, Image.Resampling.LANCZOS)

    if config.deskew:
        angle = estimate_skew_angle(result, max_angle=config.deskew_max_angle, step=config.deskew_step, threshold=config.threshold)
        if abs(angle) > 0.1:
            result = result.rotate(-angle, expand=True, fillcolor=255)

    result = ImageOps.autocontrast(result)
    return result


def render_and_preprocess_pdf_page(
    pdf_path: Path,
    rendered_dir: Path,
    processed_dir: Path,
    config: Optional[ImagePreprocessingConfig] = None,
) -> ImagePreprocessingResult:
    """Render the first page of a PDF and optionally preprocess the image."""

    if config is None:
        config = ImagePreprocessingConfig()

    rendered_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rendered_image = pdf_first_page_to_image(pdf_path, rendered_dir, dpi=config.dpi)
    processed_image_path = processed_dir / rendered_image.name

    with Image.open(rendered_image) as image:
        processed = apply_optional_preprocessing(image, config)
        processed.save(processed_image_path)

    steps = ["render_first_page"]
    if config.denoise:
        steps.append("denoise")
    if config.enhance_resolution:
        steps.append("enhance_resolution")
    if config.deskew:
        steps.append("deskew")
    steps.append("autocontrast")

    return ImagePreprocessingResult(
        source_pdf=pdf_path,
        rendered_image=rendered_image,
        processed_image=processed_image_path,
        steps=steps,
        metadata={"dpi": float(config.dpi)},
    )
