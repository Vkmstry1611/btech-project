# CatalogBank OCR — Stage 1 Document Structure Pipeline

## Overview

This project builds a modular, reproducible document-structure pipeline on top of PaddleOCR PP-StructureV3 for CatalogBank sample PDFs.

It has two layers:

| Layer | Tag | Status |
|---|---|---|
| Baseline OCR pipeline | `v0.1.0` | Frozen — faithful port of the reference notebook |
| Stage 1 research pipeline | `main` | Active — adds semantic block detection and hierarchy reconstruction |

The original notebook (`notebooks/Catalog_bank_with_paddleocr_and_ppstruct_v3.ipynb`) is preserved unchanged as the proof-of-concept baseline.

---

## What Stage 1 Does

```
Input PDF
    ↓
PDF preprocessing (PyMuPDF, configurable DPI)
    ↓
PP-StructureV3 / OCR inference (PaddleOCR pretrained models)
    ↓
Semantic block detection (deterministic rules on model output)
    ↓
Hierarchy reconstruction (reading-order + spatial geometry)
    ↓
JSON + visualization output
```

**Stage 1 does NOT:**
- Train or fine-tune any ML model
- Use embeddings, LLMs, or retrieval
- Normalize product data or build a database
- Claim to produce semantically perfect hierarchies

---

## Architecture

### Models used (all pretrained, no training)

| Model | Role |
|---|---|
| `PP-DocLayout_plus-L` | Layout region detection (bounding boxes + region type) |
| `PP-OCRv5_server_det/rec` | Text detection and recognition |
| `SLANeXt_wired` / `SLANet_plus` | Table structure reconstruction |
| `PP-FormulaNet_plus-L` | Formula detection |

### Post-processing (deterministic, no ML)

| Component | File | What it does |
|---|---|---|
| Block detector | `semantic/block_detector.py` | Reads `parsing_res_list` from PP-StructureV3 JSON; joins confidence scores from `layout_det_res.boxes` |
| Block type mapping | `semantic/block_types.py` | Maps PP-StructureV3 labels (`paragraph_title`, `text`, `image`, `table`, …) to our `SemanticType` enum using a label lookup table + text heuristics |
| Hierarchy builder | `hierarchy/hierarchy_builder.py` | Sorts blocks by reading order, builds a heading-stack tree, attaches non-heading blocks as children |
| Visualization | `visualization/research_visualization.py` | Bbox overlay on page image; matplotlib tree diagram |

### Semantic type mapping (heuristic, not trained)

PP-StructureV3 label → our SemanticType:

| PP-StructureV3 label | SemanticType |
|---|---|
| `paragraph_title` | `heading` |
| `text` | `paragraph` |
| `image` | `figure` |
| `figure_title` | `figure` |
| `table` | `table` |
| `vision_footnote` | `specification` |
| `header`, `footer`, `number` | `paragraph` (page chrome, not structural) |

Heading level is inferred from text length and label type — there is **no font-size signal** from PP-StructureV3, so all `paragraph_title` blocks initially receive the same level unless they differ in word count.

---

## Installation

```bash
# Dependencies (see pyproject.toml for exact versions)
pip install paddlepaddle paddleocr paddlex PyMuPDF Pillow matplotlib PyYAML
```

Python ≥ 3.9 required.

---

## Dataset setup

```bash
python scripts/download_dataset.py
```

This clones the [CatalogBank](https://github.com/bankh/CatalogBank) repository to `data/raw/CatalogBank`.

---

## Running the pipeline

### Process a single PDF

```bash
python scripts/run_stage1_pipeline.py --input path/to/file.pdf
```

### Process all PDFs in a directory

```bash
python scripts/run_stage1_pipeline.py --input path/to/pdf_folder/
```

### Deterministic dataset sample (default: 1 per vendor, seed 42)

```bash
python scripts/run_stage1_pipeline.py --samples-per-vendor 1 --seed 42 --dpi 200
```

### Full option reference

```
--input / -i       PDF file or directory of PDFs
--output / -o      Output root directory (default: outputs/)
--dpi              Rendering DPI (default: 200)
--samples-per-vendor  PDFs per vendor for dataset sampling (default: 1)
--seed             Sampling seed (default: 42)
--deskew           Enable deskew preprocessing
--denoise          Enable denoising preprocessing
--enhance-resolution  Enable resolution upscaling
--config           Path to config.yaml
```

### Original baseline pipeline (unchanged)

```bash
python scripts/run_pipeline.py
```

---

## Output structure

For each processed PDF, Stage 1 writes under `outputs/`:

```
outputs/
  images/{stem}/           rendered first-page PNG
  processed/{stem}/        preprocessed first-page PNG
  ocr/{stem}/              PP-StructureV3 raw outputs
    {stem}_res.json          → raw layout+OCR JSON (primary input to Stage 1)
    {stem}.md                → Markdown summary
    {stem}_layout_det_res.png → annotated layout detection image
    {stem}_overall_ocr_res.png → annotated OCR image
    imgs/                    → cropped region images
  raw/{stem}/              copy of _res.json used by semantic layer
  semantic/{stem}/
    {stem}_semantic_blocks.json  → all blocks with type, text, bbox, confidence
  hierarchy/{stem}/
    {stem}_hierarchy.json        → reconstructed document tree
  canonical/{stem}/
    {stem}_document.json         → unified schema_version 1.0 document
  visualizations/{stem}/
    {stem}_semantic_blocks.png   → bbox overlay on page image
    {stem}_hierarchy.png         → matplotlib hierarchy tree diagram
```

### Canonical JSON schema (schema_version: 1.0)

```json
{
  "schema_version": "1.0",
  "source": {
    "pdf_path": "...",
    "page_number": 1,
    "rendered_image_path": "...",
    "preprocessed_image_path": "...",
    "raw_json_path": "..."
  },
  "pages": [
    {
      "page": 1,
      "blocks": [
        {
          "id": "p1_b001",
          "type": "heading",
          "source_type": "paragraph_title",
          "text": "Stainless Steel Pipe Fittings",
          "bbox": [81, 13, 702, 59],
          "confidence": 0.593,
          "order": 1,
          "heading_level": 2
        }
      ]
    }
  ],
  "hierarchy": {
    "name": "document",
    "semantic_type": "document",
    "children": [...]
  }
}
```

---

## Limitations

1. **Single page only.** Only the first page of each PDF is processed.
2. **Heading level inference has no font-size signal.** PP-StructureV3 does not output typographic hierarchy; all `paragraph_title` blocks are assigned the same level unless text length differs.
3. **Garbled OCR on low-quality scans.** Section headings on noisy catalog scans may be unreadable. The structural nesting (correct bbox, correct children) is still produced even when the heading text is garbled.
4. **No product extraction.** Stage 1 identifies layout structure only — it does not extract product names, prices, specifications, or any catalog-specific entities.
5. **Heuristic `specification` detection.** A `text` block is classified as `specification` if it contains ≥ 2 `key: value` patterns. This is a regex heuristic, not a trained classifier.

---

## Architectural boundary — what Stage 1 does NOT do

The following are explicitly **future research stages** and are not part of Stage 1:

- Product / entity extraction
- Attribute / specification normalization
- Database or catalog ingestion
- Evaluation framework or metrics
- LLM-based semantic reasoning
- Model training or fine-tuning
- Embeddings or retrieval

Stage 1 scope is strictly: PDF preprocessing → PP-StructureV3 inference → semantic block detection → hierarchy reconstruction → JSON + visualization output.

The semantic block types and hierarchy relationships are produced by deterministic post-processing rules applied to PP-StructureV3 output. **This is not a trained semantic model.**

---

## Tests

```bash
python -m pytest tests/ -v
```

All tests run without requiring the OCR engine (integration tests use cached JSON fixtures from `outputs/raw/`).

---

## Reproducibility

- Deterministic seed: 42
- Fixed DPI: 200
- First-page only
- PP-StructureV3 pretrained models — no custom training
