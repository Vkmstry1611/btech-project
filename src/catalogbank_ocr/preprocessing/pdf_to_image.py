"""Convert the first page of a PDF to a PNG image."""

from __future__ import annotations

import logging
from pathlib import Path

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
