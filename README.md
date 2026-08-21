# CatalogBank OCR with PaddleOCR PP-StructureV3

## Overview

This project refactors a Jupyter notebook into a modular Python research workflow for document-structure extraction from CatalogBank sample PDFs using PaddleOCR PP-StructureV3.

The implementation preserves the notebook methodology:

- CatalogBank dataset usage
- deterministic sample selection with seed $42$
- Thorlabs OptoMechanics v21 and McMaster-Carr sample PDFs
- first-page PDF rendering with PyMuPDF
- PaddleOCR PP-StructureV3 processing
- JSON, Markdown, and annotated image generation
- schema inspection of the generated JSON
- annotated image visualization

## Research objective

The goal is to reproduce the notebook's document-structure extraction workflow in a cleaner, maintainable, and reproducible project layout suitable for research use and academic reporting.

## Dataset

The current implementation uses a small sample from the [CatalogBank](https://github.com/bankh/CatalogBank) repository.

Only the following subset is used:

- Thorlabs OptoMechanics v21 PDFs
- McMaster-Carr PDFs
- a deterministic small sample from each vendor
- only the first page of each selected PDF

This project does **not** process the full CatalogBank dataset.

## Methodology

The pipeline is intentionally kept close to the notebook:

1. Clone or locate the CatalogBank repository
2. Locate vendor PDF folders
3. Reproducibly select a small sample of PDFs per vendor
4. Render the first page of each selected PDF to PNG
5. Run PaddleOCR PP-StructureV3 on each rendered page
6. Save JSON, Markdown, and annotated image outputs
7. Inspect the JSON schema of the generated results
8. Visualize the annotated output image

## System / pipeline workflow

$$
\text{CatalogBank PDFs} \rightarrow \text{sample selection} \rightarrow \text{first-page rendering} \rightarrow \text{PP-StructureV3} \rightarrow \text{saved outputs} \rightarrow \text{JSON inspection} \rightarrow \text{visualization}
$$

## Project structure

- `configs/` — configuration files
- `data/` — raw, interim, processed, and sampled data locations
- `outputs/` — generated JSON, Markdown, images, and visualizations
- `notebooks/` — notebook reference copy
- `src/catalogbank_ocr/` — reusable Python package
- `scripts/` — command-line entry points
- `tests/` — lightweight tests for core utilities

## Installation

Install the Python dependencies from `requirements.txt` or the dependency list in `pyproject.toml`.

Recommended steps:

1. Create a Python environment
2. Install dependencies
3. Ensure `git` and `git-lfs` are available for dataset download

## Environment requirements

The notebook included environment repair steps for CUDA-related NVIDIA packages. In this refactor, package repair is **not** performed automatically.

You should verify compatibility for:

- PyTorch
- PaddlePaddle
- PaddleOCR
- CUDA-capable NVIDIA libraries when using GPU acceleration

The project includes a script that reports the active environment and CUDA availability.

## Dataset setup

Use the dataset download script to clone the repository and fetch the sample PDF assets:

`python scripts/download_dataset.py`

The default dataset location is `data/raw/CatalogBank`.

## Running the pipeline

Run the end-to-end workflow with defaults matching the notebook:

`python scripts/run_pipeline.py`

Optional overrides:

`python scripts/run_pipeline.py --samples-per-vendor 3 --dpi 200 --seed 42`

## Stage-1 research pipeline

The first research extension keeps the v0.1.0 baseline frozen and adds three deterministic components:

1. PDF preprocessing
2. Semantic block detection
3. Hierarchy reconstruction

Run the stage-1 pipeline with:

`python scripts/run_stage1_pipeline.py --samples-per-vendor 3 --dpi 200 --seed 42`

Optional preprocessing flags:

- `--deskew`
- `--denoise`
- `--enhance-resolution`

Stage-1 outputs are written under `outputs/`:

- `outputs/raw/` — copied PP-StructureV3 JSON
- `outputs/semantic/` — `semantic_blocks.json`
- `outputs/hierarchy/` — `hierarchy.json`
- `outputs/visualizations/` — semantic block overlays and hierarchy diagrams

## Output description

The pipeline writes generated artifacts under `outputs/`:

- `outputs/json/` — JSON outputs from PP-StructureV3
- `outputs/markdown/` — Markdown summaries
- `outputs/images/` — rendered first-page PNGs
- `outputs/visualizations/` — annotated PP-StructureV3 images
- `outputs/ppstructurev3/` — intermediate PP-StructureV3 per-sample output folders

## PP-StructureV3 explanation

PaddleOCR PP-StructureV3 is used as the document structure extraction engine. The current project does not add a new model or alter the notebook's OCR methodology; it only wraps the existing workflow in reusable functions.

## Reproducibility

Reproducibility is preserved through:

- deterministic random seed $42$
- fixed sample-per-vendor defaults
- fixed first-page selection
- fixed PDF rendering DPI

## Current limitations

- only a small sample of CatalogBank is processed
- only the first page of each selected PDF is rendered and analyzed
- no product extraction pipeline is implemented
- no normalization, embeddings, retrieval, or evaluation layer is added
- JSON schema handling is intentionally exploratory and does not assume a fixed legacy format

## Future work

- expand sample coverage while preserving reproducibility
- add structured result analysis if a stable PP-StructureV3 schema is established
- add optional batch processing for larger experiments
- add experiment tracking and quantitative evaluation

## Notebook reference

The original notebook is preserved in the repository root and also copied to `notebooks/` for reference.
