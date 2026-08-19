"""Display and save PP-StructureV3 annotated images."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_image_files(output_dir: Path) -> list[Path]:
    """Return annotated image files under an output directory."""

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    return sorted(output_dir.rglob("*.png"))


def display_image(image_path: Path, figure_path: Path | None = None, show: bool = True) -> None:
    """Display an image with matplotlib and optionally save the figure."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("matplotlib and Pillow are required for visualization") from exc

    image = Image.open(image_path)
    plt.figure(figsize=(10, 14))
    plt.imshow(image)
    plt.axis("off")
    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, bbox_inches="tight", pad_inches=0)
        logger.info("Saved visualization to %s", figure_path)
    if show:
        plt.show()
    else:
        plt.close()
