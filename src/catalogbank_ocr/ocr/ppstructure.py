"""Run PaddleOCR PP-StructureV3 on rendered page images."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Preserve the existing oneDNN workaround without hiding CUDA devices.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")

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


def initialize_ppstructurev3(device: str = 'auto', recognition_batch_size: int | None = None, **kwargs: Any) -> Any:
    """Initialize the PaddleOCR PPStructureV3 engine lazily."""

    try:
        from paddleocr import PPStructureV3
    except Exception as exc:  # pragma: no cover - depends on external package
        raise RuntimeError("Unable to import paddleocr.PPStructureV3") from exc

    import paddle
    if recognition_batch_size is not None and recognition_batch_size < 1:
        raise ValueError('recognition_batch_size must be positive')
    available = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    if device == 'auto':
        device = 'gpu:0' if available else 'cpu'
    elif device.startswith('gpu') and not available:
        raise RuntimeError('GPU requested but installed PaddlePaddle has no usable CUDA device. Install a compatible GPU build or use --device cpu.')
    if recognition_batch_size is None and device.startswith('gpu'):
        # Dense catalog lines can exhaust an 8 GB GPU at the model's batch of 8.
        recognition_batch_size = 1
    logger.info('Initializing PPStructureV3 on %s', device)
    if recognition_batch_size is not None:
        kwargs['text_recognition_batch_size'] = recognition_batch_size
    engine = PPStructureV3(device=device, **kwargs)
    if hasattr(engine, '__dict__'):
        engine._catalogbank_runtime = {'device': device, 'recognition_batch_size': recognition_batch_size}
    return engine


def run_ppstructurev3(image_path: Path, output_dir: Path, engine: Any | None = None, debug_artifacts: bool = True) -> PPStructureRunResult:
    """Process an image with PP-StructureV3 and persist its outputs."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = engine or initialize_ppstructurev3()

    prediction_count = 0
    for result in engine.predict(str(image_path)):
        result.save_to_json(save_path=str(output_dir))
        if debug_artifacts:
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


def run_ppstructurev3_batch(images: list[Path], output_dirs: list[Path], engine: Any,
                          debug_artifacts: bool = True) -> list[PPStructureRunResult]:
    """Submit a bounded list of page images through the same model pipeline."""
    if not images or len(images) != len(output_dirs):
        raise ValueError('Batch images and output directories must be nonempty and aligned')
    for path in images:
        if not path.is_file(): raise FileNotFoundError(path)
    predictions = list(engine.predict([str(path) for path in images]))
    if len(predictions) != len(images):
        raise RuntimeError('OCR page count differs from submitted batch; refusing to misalign pages')
    runs = []
    for path, directory, prediction in zip(images, output_dirs, predictions):
        directory.mkdir(parents=True, exist_ok=True)
        prediction.save_to_json(save_path=str(directory))
        if debug_artifacts:
            prediction.save_to_markdown(save_path=str(directory))
            prediction.save_to_img(save_path=str(directory))
        runs.append(PPStructureRunResult(path, directory, 1, sorted(directory.glob('*.json')),
                                        sorted(directory.glob('*.md')), sorted(directory.glob('*.png'))))
    return runs
