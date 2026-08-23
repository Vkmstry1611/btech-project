# CatalogBank Document Structure Pipeline — Architecture

> Describes the **actually implemented** system as of Stage 2 integration.
> All module paths are relative to `src/catalogbank_ocr/`.

---

## End-to-End Data Flow

```
Input PDF
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  PDF Preprocessing                                  │
│  preprocessing/pdf_to_image.py                      │
│  preprocessing/image_preprocessing.py               │
│  • Render first page → PNG (PyMuPDF, configurable   │
│    DPI, default 200)                                │
│  • Optional: deskew, denoise, resolution enhance,   │
│    autocontrast                                     │
└────────────────────┬────────────────────────────────┘
                     │ rendered + preprocessed page image (PNG)
                     ▼
┌─────────────────────────────────────────────────────┐
│  PP-StructureV3 / OCR Engine                        │
│  ocr/ppstructure.py                                 │
│  • PaddleOCR PPStructureV3                          │
│  • Layout detection (PP-DocLayout_plus-L)           │
│  • OCR (PP-OCRv5_server_det/rec)                   │
│  • Table reconstruction (SLANeXt / SLANet_plus)    │
│  • Formula detection (PP-FormulaNet_plus-L)         │
│  • Outputs: _res.json, .md, visualisation PNGs     │
└────────────────────┬────────────────────────────────┘
                     │ raw JSON (_res.json)
                     │  {
                     │    "parsing_res_list": [          ← primary structured source
                     │      { block_label, block_content,
                     │        block_bbox, block_order }, …
                     │    ],
                     │    "layout_det_res": { boxes },
                     │    "overall_ocr_res": { rec_texts, rec_boxes, … },
                     │    "table_res_list": [ … ]
                     │  }
                     ▼
┌─────────────────────────────────────────────────────┐
│  Semantic Block Detection                           │
│  semantic/block_detector.py                         │
│  • Reads parsing_res_list as primary source         │
│    (avoids duplication from raw OCR line arrays)    │
│  • Normalises bbox to [x1,y1,x2,y2]                │
│  • Falls back to recursive walk for legacy shapes   │
└────────────────────┬────────────────────────────────┘
                     │ List[BlockCandidate]
                     ▼
┌─────────────────────────────────────────────────────┐
│  Block Classification / Mapping                     │
│  semantic/block_mapper.py                           │
│  semantic/block_types.py                            │
│                                                     │
│  PP-StructureV3 label → SemanticType mapping:       │
│    paragraph_title → HEADING                        │
│    text            → PARAGRAPH                      │
│    image           → FIGURE                         │
│    figure_title    → FIGURE                         │
│    table           → TABLE                          │
│    vision_footnote → SPECIFICATION                  │
│    footer/header   → PARAGRAPH (kept, low priority) │
│                                                     │
│  Text heuristics (secondary):                       │
│    Markdown # prefix → HEADING                      │
│    key:value density → SPECIFICATION                │
│    price+dim+model   → PRODUCT_CARD                 │
└────────────────────┬────────────────────────────────┘
                     │ SemanticDocument
                     │  { pages: [ { blocks: [SemanticBlock] } ] }
                     ▼
┌─────────────────────────────────────────────────────┐
│  Hierarchy Reconstruction                           │
│  hierarchy/hierarchy_builder.py                     │
│  hierarchy/relationship_rules.py                    │
│                                                     │
│  Algorithm (deterministic, reading-order):          │
│    1. Sort blocks by (order, bbox_top, bbox_left)   │
│    2. HEADING → push onto heading stack with level  │
│    3. PRODUCT_CARD → attach to current heading,     │
│       becomes active container                      │
│    4. Other blocks → attach to active container if  │
│       spatially close (vertical gap + column check) │
│       else attach to current heading                │
│    5. Gap > 250 px → reset active container         │
└────────────────────┬────────────────────────────────┘
                     │ HierarchyDocument { root: HierarchyNode }
                     ▼
┌─────────────────────────────────────────────────────┐
│  Output Generation                                  │
│  research_stage1_pipeline.py                        │
│                                                     │
│  Writes per-PDF under outputs/                      │
│    raw/{stem}/*_res.json         ← PP-StructureV3   │
│    semantic/{stem}/*_semantic_blocks.json           │
│    hierarchy/{stem}/*_hierarchy.json                │
│    canonical/{stem}/*_document.json  ← unified      │
│    visualizations/{stem}/*_semantic_blocks.png      │
│    visualizations/{stem}/*_hierarchy.png            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
             Canonical JSON + Visualizations
```

---

## Canonical JSON Schema (schema_version: "1.0")

```json
{
  "schema_version": "1.0",
  "source": {
    "pdf_path":                  "absolute/path/to/file.pdf",
    "page_number":               1,
    "rendered_image_path":       "outputs/images/stem/stem.png",
    "preprocessed_image_path":   "outputs/processed/stem/stem.png",
    "raw_json_path":             "outputs/raw/stem/stem_res.json"
  },
  "pages": [
    {
      "page": 1,
      "blocks": [
        {
          "id":           "p1_b001",
          "page":         1,
          "type":         "heading",        // SemanticType enum value
          "source_type":  "paragraph_title",// raw PP-StructureV3 label
          "text":         "Stainless Steel Pipe Fittings",
          "bbox":         [x1, y1, x2, y2],
          "confidence":   0.97,
          "order":        1,
          "source_ref":   "parsing_res_list[0]",
          "heading_level": 1
        }
      ]
    }
  ],
  "hierarchy": {
    "name":          "document",
    "semantic_type": "document",
    "children": [
      {
        "name":          "Stainless Steel Pipe Fittings",
        "semantic_type": "heading",
        "bbox":          [x1, y1, x2, y2],
        "children": [
          {
            "name":          "Pipe Nipples and Pipe: ...",
            "semantic_type": "specification",
            "children": []
          }
        ]
      }
    ]
  }
}
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Load `configs/config.yaml`, resolve paths |
| `preprocessing/pdf_to_image.py` | PDF first-page → PNG via PyMuPDF |
| `preprocessing/image_preprocessing.py` | Optional deskew / denoise / enhance |
| `ocr/ppstructure.py` | Thin wrapper around PaddleOCR PPStructureV3 |
| `postprocessing/json_inspection.py` | Load and inspect raw OCR JSON |
| `semantic/block_detector.py` | Extract `BlockCandidate` list from raw JSON |
| `semantic/block_types.py` | Label→SemanticType mapping + text heuristics |
| `semantic/block_mapper.py` | `BlockCandidate` → `SemanticBlock` + `SemanticDocument` |
| `hierarchy/hierarchy_builder.py` | `SemanticDocument` → `HierarchyDocument` tree |
| `hierarchy/relationship_rules.py` | Spatial containment helpers (gaps, overlap) |
| `schemas/semantic_block.py` | `SemanticBlock`, `PageSemanticBlocks`, `SemanticDocument` |
| `schemas/hierarchy.py` | `HierarchyNode`, `HierarchyDocument` |
| `schemas/document.py` | `Stage1Document`, `DocumentSource`, `DocumentPage` |
| `visualization/research_visualization.py` | Bbox overlay + matplotlib tree diagram |
| `research_stage1_pipeline.py` | End-to-end orchestration; writes all outputs |
| `pipeline.py` | Original baseline OCR-only pipeline (preserved) |

---

## Key Integration Decision

PP-StructureV3 produces **two parallel representations** in its JSON:

- `parsing_res_list` — clean, de-duplicated, ordered layout blocks (24 blocks per page typical)
- `overall_ocr_res.rec_texts` — individual OCR text lines (100–500 per page)

The Stage 1 detector now uses `parsing_res_list` as the **exclusive primary source** for the
structured pipeline. This eliminates the 6× duplication that was producing 156 flat blocks
instead of 24 structured blocks. The generic recursive walker is retained as a fallback for
non-standard JSON shapes.

---

## CLI Entry Points

```bash
# Full end-to-end pipeline on a single PDF
python3 scripts/run_stage1_pipeline.py \
    --single-pdf path/to/file.pdf \
    --dpi 200

# Dataset sampling (N PDFs per vendor, deterministic seed)
python3 scripts/run_stage1_pipeline.py \
    --samples-per-vendor 3 \
    --seed 42 \
    --dpi 200

# Original baseline OCR-only pipeline
python3 scripts/run_pipeline.py
```

---

## Output Directory Structure

```
outputs/
  images/{stem}/            rendered page PNGs
  processed/{stem}/         preprocessed page PNGs
  ocr/{stem}/               PP-StructureV3 raw outputs (_res.json, .md, viz PNGs)
  raw/{stem}/               copy of _res.json used by semantic layer
  semantic/{stem}/          *_semantic_blocks.json
  hierarchy/{stem}/         *_hierarchy.json
  canonical/{stem}/         *_document.json  (unified schema_version 1.0)
  visualizations/{stem}/    *_semantic_blocks.png
                            *_hierarchy.png
```
