# HierarchIQ catalog extraction

The current entry point processes multi-page PDFs into a product catalog following
`sampleoutput.json`: Product Category → Model → Sellable Item → List Price.
It is a command-line application; there is no upload web interface.

## Run the full pipeline

Use Python 3.10 or newer. For the existing CPU environment:

```powershell
python -m pip install -r requirements.txt
ollama pull qwen3:8b
python scripts/run_final_pipeline.py --input pdfs --backend ollama --model qwen3:8b
```

Keep Ollama running. To process one PDF, pass its quoted path to `--input`.
Ollama is selected explicitly by default; the pipeline does not silently switch
production runs to the empty mock backend.

```powershell
# Keep the OCR engine loaded and accept further PDF paths in the same process.
python scripts/run_final_pipeline.py --input pdfs --interactive

# Auto-select a usable Paddle CUDA build, or fall back to CPU.
python scripts/run_final_pipeline.py --input pdfs --device auto

# Require GPU OCR; fail clearly if the installed Paddle build cannot use it.
python scripts/run_final_pipeline.py --input pdfs --device gpu:0

# Optional recognition batching. Default: 1 on GPU; model default on CPU.
python scripts/run_final_pipeline.py --input pdfs --recognition-batch-size 8

# Optional bounded page batching; use a size that fits available memory.
python scripts/run_final_pipeline.py --input pdfs --device gpu:0 --page-batch-size 2

# Include optional annotated images, Markdown and graph visualizations.
python scripts/run_final_pipeline.py --input pdfs --debug-artifacts

# Repeat extraction from saved OCR for the same input PDF.
python scripts/run_final_pipeline.py --input pdfs --skip-stage1

# Revalidate/export saved semantic extractions after a parser or validation fix.
python scripts/run_final_pipeline.py --input "pdfs/industrial_fittings_5_pages.pdf" --skip-stage1 --reuse-extractions
```

GPU OCR requires a compatible **paddlepaddle-gpu** installation, not only an NVIDIA
driver. Use a separate environment when changing Paddle builds. See the
[official installation guide](https://www.paddlepaddle.org.cn/install/quick).

The local `.venv-gpu` environment has Paddle GPU 3.2.2 with CUDA 12.9 installed.
On this laptop, run it explicitly to use the RTX 5060:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/run_final_pipeline.py --input pdfs --device gpu:0 --interactive
```

Keep recognition and page batch sizes at 1 on this 8 GB GPU unless a representative
dense catalog has been measured with larger batches. Batching does not change the
recognition models, but larger batches can increase memory pressure and slow a run.

To recreate the GPU overlay after setting up the ordinary dependencies:

```powershell
python -m venv --system-site-packages .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

## Output and review

- `outputs/final/<name>/<name>_final.json`: validated catalog; model and SKU are
  separate, and explicit metric/imperial variants are preserved.
- `outputs/final/<name>/status.json`: running, complete, complete_with_review or failed status. Earlier
  successful exports are archived before a new run, so a failed run cannot leave
  an old file masquerading as the new final result.
- `outputs/canonical/<name>/`: every page, complete blocks, page classification,
  source metadata and document hierarchy.
- `outputs/stage2/<name>/`: full extraction evidence, normalized records, validation
  issues and correction attempts. Unsupported or ambiguous identifiers block export.
  Ambiguous prices are retained in normalized evidence and listed for review;
  they are not converted into guessed numeric prices.
- `outputs/metrics/<name>/`: per-page OCR/rendering/preprocessing timings and
  pipeline timings. Startup metrics are saved at the output root. Memory metrics
  cover the Python process, not the separate Ollama process or total GPU VRAM.

Covers and indexes provide metadata; section dividers carry context forward.
Mixed and uncertain pages retain their content for extraction. Page geometry is
never used to attach a block to a product card on another page. Long sections
are split into bounded requests without silently dropping rows or text. Explicit
SKU table rows are read directly; mixed text and unresolved structure use the LLM.
When a row includes a distributor stock/order code and a manufacturer part code,
the stock/order code identifies the sellable item; the manufacturer code remains a
property. Metric and imperial SKU columns remain separate sellable variants.
Uncertain continuation tables retain their original cells and a continuation hint;
the pipeline does not guess missing headers or repair ambiguous OCR cell alignment.
For digital PDFs, exact SKU/price pairs with printed dotted leaders in the PDF
text layer can supplement items lost when OCR fuses dense table cells. Prices
without an explicit currency symbol keep a null currency. Unsupported family
headings and material grades are not exported as product models. Malformed
table fragments are recorded for review when source text recovers their SKUs.

The two five-page evaluation PDFs are `pdfs/optical_hardware_5_pages.pdf` and
`pdfs/industrial_fittings_5_pages.pdf`. Their extracted catalogs and review
artifacts are under `outputs/evaluations/two_domains/pipeline/`.

## Benchmark a new PDF

```powershell
python scripts/benchmark_catalog.py --input "path/to/catalog.pdf" --max-pages 60 --device auto
```

Add `--compare` to measure sequential preparation with debug artifacts against
streaming preparation without them. Both runs retain the same DPI and recognition
models. This compares execution settings, not extraction accuracy: check the
resulting data against an annotated reference as well. A PDF shorter than 60 pages
does not constitute a 60-page benchmark. `--skip-stage1` is for repeat runs and is
not used by this benchmark.

Run regression tests with `python -m pytest tests`. Implementation and outstanding
evaluation work are tracked in [TODO.md](TODO.md).

---

The following sections document the original Stage 1 pipeline and its legacy commands.

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
