"""CatalogBank dataset discovery, validation, and sample selection."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass()
class VendorPdfs:
    """Collection of PDF paths for each vendor."""

    thorlabs: list[Path]
    mcmaster: list[Path]


def find_vendor_pdfs(repo_dir: Path, vendor_relative_dir: str) -> list[Path]:
    """Find all PDFs for a given vendor directory."""

    pdf_dir = repo_dir / vendor_relative_dir
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    logger.info("Found %d PDFs in %s", len(pdfs), pdf_dir)
    return pdfs


def load_vendor_pdfs(repo_dir: Path, thorlabs_relative_dir: str, mcmaster_relative_dir: str) -> VendorPdfs:
    """Load vendor PDF lists from the CatalogBank repository."""

    return VendorPdfs(
        thorlabs=find_vendor_pdfs(repo_dir, thorlabs_relative_dir),
        mcmaster=find_vendor_pdfs(repo_dir, mcmaster_relative_dir),
    )


def select_sample_pdfs(
    thorlabs_pdfs: list[Path],
    mcmaster_pdfs: list[Path],
    samples_per_vendor: int = 3,
    seed: int = 42,
) -> list[Path]:
    """Select a deterministic sample of PDFs from both vendors.

    The selection order matches the notebook: Thorlabs samples first, then
    McMaster-Carr samples.
    """

    if samples_per_vendor < 0:
        raise ValueError("samples_per_vendor must be non-negative")

    rng = random.Random(seed)
    thorlabs_count = min(samples_per_vendor, len(thorlabs_pdfs))
    mcmaster_count = min(samples_per_vendor, len(mcmaster_pdfs))
    selected = rng.sample(sorted(thorlabs_pdfs), thorlabs_count) + rng.sample(
        sorted(mcmaster_pdfs), mcmaster_count
    )
    logger.info(
        "Selected %d PDFs total (%d Thorlabs, %d McMaster-Carr)",
        len(selected),
        thorlabs_count,
        mcmaster_count,
    )
    return selected


def validate_pdf_paths(pdf_paths: list[Path]) -> None:
    """Ensure every PDF path exists before processing."""

    if not pdf_paths:
        raise ValueError("No PDF paths were provided")

    missing = [path for path in pdf_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PDF files: {missing}")
