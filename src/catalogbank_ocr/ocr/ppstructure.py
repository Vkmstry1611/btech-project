"""Run PaddleOCR PP-StructureV3 on rendered page images."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Force CPU-only mode — PaddlePaddle GPU builds require CUDA <=12.x.
# CUDA 13.x (RTX 50-series) is not yet supported by any PaddlePaddle wheel.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPU from paddle entirely

logger = logging.getLogger(__name__)


@dataclass()
class PPStructureRunResult:
    """Metadata for a single PP-StructureV3 image run."""

    image_path: Path
    output_dir: Path
    prediction_count: int
    json_files: list[Path]
    markdown_files: list[Path]
    image_files: list[Path]


def initialize_ppstructurev3(**kwargs: Any) -> Any:
    """Initialize the PaddleOCR PPStructureV3 engine lazily."""

    try:
        from paddleocr import PPStructureV3
    except Exception as exc:  # pragma: no cover - depends on external package
        raise RuntimeError("Unable to import paddleocr.PPStructureV3") from exc

    logger.info("Initializing PPStructureV3")
    return PPStructureV3(device="cpu", **kwargs)


def run_ppstructurev3(image_path: Path, output_dir: Path, engine: Any | None = None) -> PPStructureRunResult:
    """Process an image with PP-StructureV3 and persist its outputs."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or initialize_ppstructurev3()

    prediction_count = 0
    for result in engine.predict(str(image_path)):
        result.save_to_json(save_path=str(output_dir))
        result.save_to_markdown(save_path=str(output_dir))
        result.save_to_img(save_path=str(output_dir))
        prediction_count += 1

    if prediction_count == 0:
        raise RuntimeError(f"PPStructureV3 returned no results for {image_path}")

    json_files = sorted(output_dir.glob("*.json"))
    markdown_files = sorted(output_dir.glob("*.md"))
    image_files = sorted(output_dir.glob("*.png"))
    logger.info(
        "PPStructureV3 completed for %s with %d result(s)", image_path, prediction_count
    )
    return PPStructureRunResult(
        image_path=image_path,
        output_dir=output_dir,
        prediction_count=prediction_count,
        json_files=json_files,
        markdown_files=markdown_files,
        image_files=image_files,
    )
