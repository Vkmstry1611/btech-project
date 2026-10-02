"""Convert PDF pages to PNG images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def pdf_first_page_to_image(pdf_path: Path, output_dir: Path, dpi: int = 200) -> Path:
    """Render the first page of a PDF to a PNG file.

    Args:
        pdf_path: Path to the input PDF.
        output_dir: Directory where the PNG will be written.
        dpi: Rendering resolution.

    Returns:
        Path to the generated PNG image.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PyMuPDF is required to render PDF pages") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    try:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        page = document[0]
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pixmap = page.get_pixmap(matrix=matrix)
        output_path = output_dir / f"{pdf_path.stem}.png"
        logger.info("Rendering first page of %s to %s", pdf_path, output_path)
        pixmap.save(str(output_path))
        return output_path
    finally:
        document.close()


def pdf_all_pages_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    max_pages: Optional[int] = None,
) -> List[Path]:
    """Render all pages of a PDF to individual PNG files.

    Args:
        pdf_path: Path to the input PDF.
        output_dir: Directory where PNGs will be written.
        dpi: Rendering resolution.
        max_pages: If set, render only the first N pages.

    Returns:
        List of paths to the generated PNG images, one per page, in order.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PyMuPDF is required to render PDF pages") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    try:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {pdf_path}")

        total = document.page_count
        if max_pages is not None:
            total = min(total, max_pages)

        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        output_paths: List[Path] = []

        for page_idx in range(total):
            page = document[page_idx]
            pixmap = page.get_pixmap(matrix=matrix)
            # Zero-padded page number suffix: stem_p001.png, stem_p002.png …
            output_path = output_dir / f"{pdf_path.stem}_p{page_idx + 1:03d}.png"
            pixmap.save(str(output_path))
            logger.info("Rendered page %d/%d of %s → %s", page_idx + 1, total, pdf_path.name, output_path.name)
            output_paths.append(output_path)

        return output_paths
    finally:
        document.close()
