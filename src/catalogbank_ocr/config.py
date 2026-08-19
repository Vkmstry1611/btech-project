"""Configuration loading and path resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root directory."""

    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Return the default configuration file path."""

    return project_root() / "configs" / "config.yaml"


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


@dataclass()
class DatasetConfig:
    """Dataset-related configuration."""

    repo_url: str
    repo_dir: Path
    thorlabs_pdf_dir: Path
    mcmaster_pdf_dir: Path
    samples_per_vendor: int = 3
    seed: int = 42


@dataclass()
class PreprocessingConfig:
    """PDF preprocessing configuration."""

    dpi: int = 200


@dataclass()
class OutputConfig:
    """Output directory configuration."""

    base_dir: Path
    json_dir: Path
    markdown_dir: Path
    images_dir: Path
    visualizations_dir: Path
    ocr_dir: Path


@dataclass()
class OCRConfig:
    """OCR engine configuration."""

    ppstructurev3: dict[str, Any] = field(default_factory=dict)


@dataclass()
class ProjectConfig:
    """Fully resolved project configuration."""

    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    outputs: OutputConfig
    ocr: OCRConfig


def load_config(config_path: Path | None = None) -> ProjectConfig:
    """Load configuration from YAML and resolve relative paths.

    Args:
        config_path: Optional path to a YAML config file.

    Returns:
        Resolved project configuration.
    """

    root = project_root()
    path = config_path or default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PyYAML is required to load config.yaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset_raw = raw.get("dataset", {})
    preprocessing_raw = raw.get("preprocessing", {})
    outputs_raw = raw.get("outputs", {})
    ocr_raw = raw.get("ocr", {})

    dataset = DatasetConfig(
        repo_url=str(dataset_raw.get("repo_url", "https://github.com/bankh/CatalogBank.git")),
        repo_dir=_resolve_path(root, dataset_raw.get("repo_dir", "data/raw/CatalogBank")),
        thorlabs_pdf_dir=_resolve_path(
            root,
            dataset_raw.get(
                "thorlabs_pdf_dir",
                "data/raw/CatalogBank/Catalogs/Sample/Thorlabs/OptoMechanics_v21/_pdfs",
            ),
        ),
        mcmaster_pdf_dir=_resolve_path(
            root,
            dataset_raw.get(
                "mcmaster_pdf_dir",
                "data/raw/CatalogBank/Catalogs/Sample/McMasterCarr/_pdfs",
            ),
        ),
        samples_per_vendor=int(dataset_raw.get("samples_per_vendor", 3)),
        seed=int(dataset_raw.get("seed", 42)),
    )

    preprocessing = PreprocessingConfig(dpi=int(preprocessing_raw.get("dpi", 200)))

    base_dir = _resolve_path(root, outputs_raw.get("base_dir", "outputs"))
    outputs = OutputConfig(
        base_dir=base_dir,
        json_dir=_resolve_path(root, outputs_raw.get("json_dir", "outputs/json")),
        markdown_dir=_resolve_path(root, outputs_raw.get("markdown_dir", "outputs/markdown")),
        images_dir=_resolve_path(root, outputs_raw.get("images_dir", "outputs/images")),
        visualizations_dir=_resolve_path(
            root, outputs_raw.get("visualizations_dir", "outputs/visualizations")
        ),
        ocr_dir=_resolve_path(root, outputs_raw.get("ocr_dir", "outputs/ppstructurev3")),
    )

    ocr = OCRConfig(ppstructurev3=dict(ocr_raw.get("ppstructurev3", {})))

    return ProjectConfig(dataset=dataset, preprocessing=preprocessing, outputs=outputs, ocr=ocr)


def ensure_project_directories(config: ProjectConfig) -> None:
    """Create the standard directory structure used by the project."""

    directories = [
        config.dataset.repo_dir.parent,
        config.outputs.base_dir,
        config.outputs.json_dir,
        config.outputs.markdown_dir,
        config.outputs.images_dir,
        config.outputs.visualizations_dir,
        config.outputs.ocr_dir,
        project_root() / "data" / "interim",
        project_root() / "data" / "processed",
        project_root() / "data" / "samples",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
