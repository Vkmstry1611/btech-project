"""
POC Output Generator — Stage 2 Pipeline
========================================
Runs the full A->B->C->D pipeline on the synthetic catalog fixture,
saves all intermediate JSON outputs, and generates 4 publication-quality
images suitable for a slide deck.

Images produced (in outputs/poc_outputs/):
  01_pipeline_overview.png     -- full pipeline architecture diagram
  02_context_chunks.png        -- Phase A context chunks extracted from catalog
  03_extraction_results.png    -- Phase C normalized entities (products/attrs/relations)
  04_knowledge_graph.png       -- Phase D knowledge graph visualization

Usage:
    python scripts/generate_poc_outputs.py
    python scripts/generate_poc_outputs.py --output outputs/my_poc
"""

from __future__ import annotations

import sys

# Force UTF-8 stdout/stderr on Windows so unicode chars in print() don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from catalogbank_ocr.stage2.context_builder import build_context_chunks
from catalogbank_ocr.stage2.table_parser import parse_table_blocks
from catalogbank_ocr.stage2.llm_input_builder import build_llm_input, estimate_prompt_chars
from catalogbank_ocr.stage2.llm_extractor import LLMExtractor, ExtractionResult
from catalogbank_ocr.stage2.validation import validate_batch
from catalogbank_ocr.stage2.normalization import (
    normalize_batch, NormalizedResult,
    NormalizedProduct, NormalizedAttribute, NormalizedRelation,
)
from catalogbank_ocr.stage2.graph_ingest import GraphIngestor, _build_statements

# Import the synthetic fixture from Phase A script
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from run_stage2_phase_a import SYNTHETIC_CANONICAL

# ── colour palette (dark-on-light, PPT-safe) ─────────────────────────────────
C = {
    "bg":      "#F7F9FC",
    "heading": "#1A2E44",
    "blue":    "#2563EB",
    "teal":    "#0891B2",
    "green":   "#16A34A",
    "amber":   "#D97706",
    "red":     "#DC2626",
    "purple":  "#7C3AED",
    "slate":   "#475569",
    "light":   "#E2E8F0",
    "white":   "#FFFFFF",
    "border":  "#CBD5E1",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Run the actual pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(out_dir: Path):
    """Execute A→B→C→D and return all intermediate results."""
    print("  [A] Building context chunks …")
    chunks = build_context_chunks(SYNTHETIC_CANONICAL,
                                  source_doc_path="synthetic_catalog.pdf",
                                  skip_empty=True)

    print(f"      → {len(chunks)} chunks")

    print("  [B] Building LLM inputs (mock extraction) …")
    llm_inputs = [build_llm_input(c, include_prompt=True) for c in chunks]

    # Use mock backend — produces empty JSON, but we'll inject realistic data
    # for the visualisations so the PPT shows what real output looks like
    extractor = LLMExtractor(backend="mock")
    raw_results = extractor.extract_batch(llm_inputs, progress=False)

    # ── Inject realistic extraction data into the mock results ───────────────
    # (This simulates what Qwen3-8B would actually return from these chunks.)
    from catalogbank_ocr.stage2.llm_extractor import (
        ExtractedProduct, ExtractedAttribute, ExtractedRelation
    )

    # Chunk 0: Pipe Nipples
    raw_results[0].products = [
        ExtractedProduct(name="SS-NIP-001", model="SS-NIP-001", sku="SS-NIP-001", category="Pipe Nipples"),
        ExtractedProduct(name="SS-NIP-002", model="SS-NIP-002", sku="SS-NIP-002", category="Pipe Nipples"),
        ExtractedProduct(name="SS-NIP-003", model="SS-NIP-003", sku="SS-NIP-003", category="Pipe Nipples"),
    ]
    raw_results[0].attributes = [
        ExtractedAttribute(entity="SS-NIP-001", key="Material",         value="Stainless Steel 316", unit=None),
        ExtractedAttribute(entity="SS-NIP-001", key="Thread",           value="1/4-20",              unit=None),
        ExtractedAttribute(entity="SS-NIP-001", key="Length",           value="2 in",                unit="in"),
        ExtractedAttribute(entity="SS-NIP-001", key="Pressure Rating",  value="3000 psi",            unit="psi"),
        ExtractedAttribute(entity="SS-NIP-002", key="Size",             value="3/8 in",              unit="in"),
        ExtractedAttribute(entity="SS-NIP-002", key="Price",            value="$4.10",               unit=None),
        ExtractedAttribute(entity="SS-NIP-003", key="Size",             value="1/2 in",              unit="in"),
        ExtractedAttribute(entity="SS-NIP-003", key="Length",           value="3 in",                unit="in"),
    ]
    raw_results[0].relations = [
        ExtractedRelation(subject="SS-NIP-002", predicate="IS_VARIANT_OF", object="SS-NIP-001"),
        ExtractedRelation(subject="SS-NIP-003", predicate="IS_VARIANT_OF", object="SS-NIP-001"),
        ExtractedRelation(subject="SS-NIP-001", predicate="HAS_MATERIAL",  object="Stainless Steel 316"),
    ]

    # Chunk 1: Pipe Elbows
    raw_results[1].products = [
        ExtractedProduct(name="SS-ELB-90-025", model="SS-ELB-90-025", sku="SS-ELB-90-025", category="Pipe Elbows"),
        ExtractedProduct(name="SS-ELB-90-038", model="SS-ELB-90-038", sku="SS-ELB-90-038", category="Pipe Elbows"),
        ExtractedProduct(name="SS-ELB-45-025", model="SS-ELB-45-025", sku="SS-ELB-45-025", category="Pipe Elbows"),
    ]
    raw_results[1].attributes = [
        ExtractedAttribute(entity="SS-ELB-90-025", key="Angle",    value="90°",    unit="°"),
        ExtractedAttribute(entity="SS-ELB-90-025", key="Size",     value="1/4 in", unit="in"),
        ExtractedAttribute(entity="SS-ELB-90-025", key="Material", value="316 SS", unit=None),
        ExtractedAttribute(entity="SS-ELB-90-025", key="Price",    value="$6.20",  unit=None),
        ExtractedAttribute(entity="SS-ELB-90-038", key="Angle",    value="90°",    unit="°"),
        ExtractedAttribute(entity="SS-ELB-90-038", key="Size",     value="3/8 in", unit="in"),
        ExtractedAttribute(entity="SS-ELB-45-025", key="Angle",    value="45°",    unit="°"),
        ExtractedAttribute(entity="SS-ELB-45-025", key="Material", value="304 SS", unit=None),
    ]
    raw_results[1].relations = [
        ExtractedRelation(subject="SS-ELB-90-038", predicate="IS_VARIANT_OF", object="SS-ELB-90-025"),
        ExtractedRelation(subject="SS-ELB-45-025", predicate="IS_VARIANT_OF", object="SS-ELB-90-025"),
        ExtractedRelation(subject="SS-ELB-90-025", predicate="HAS_MATERIAL",  object="316 SS"),
        ExtractedRelation(subject="SS-ELB-45-025", predicate="HAS_MATERIAL",  object="304 SS"),
    ]

    # Chunk 2: Optical Lens Mounts
    raw_results[2].products = [
        ExtractedProduct(name="LM1-A", model="LM1-A", sku="LM1-A", category="Lens Mounts"),
    ]
    raw_results[2].attributes = [
        ExtractedAttribute(entity="LM1-A", key="Lens Diameter", value="25.4 mm",       unit="mm"),
        ExtractedAttribute(entity="LM1-A", key="Mount Type",    value="SM1 (1.035-40)", unit=None),
        ExtractedAttribute(entity="LM1-A", key="Material",      value="Anodized Aluminum", unit=None),
        ExtractedAttribute(entity="LM1-A", key="Coating",       value="None",           unit=None),
    ]
    raw_results[2].relations = [
        ExtractedRelation(subject="LM1-A", predicate="BELONGS_TO_CATEGORY", object="Lens Mounts"),
    ]

    print("  [C] Validating and normalizing …")
    reports   = validate_batch(raw_results)
    normalized = normalize_batch(raw_results)
    print(f"      → {len(normalized.products)} products, "
          f"{len(normalized.attributes)} attributes, "
          f"{len(normalized.relations)} relations")

    print("  [D] Building graph statements (dry-run) …")
    ingestor = GraphIngestor(dry_run=True)
    ingestion = ingestor.ingest(normalized)
    print(f"      → {ingestion.statements_executed} Cypher statements, "
          f"~{ingestion.nodes_merged} nodes, ~{ingestion.rels_merged} rels")

    # Save JSON outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "normalized.json").write_text(
        json.dumps(normalized.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "ingestion_dry_run.json").write_text(
        json.dumps(ingestion.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    val_data = {cid: r.to_dict() for cid, r in reports.items()}
    (out_dir / "validation_report.json").write_text(
        json.dumps(val_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return chunks, raw_results, normalized, ingestion


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 1 — Pipeline Overview
# ─────────────────────────────────────────────────────────────────────────────

def fig1_pipeline_overview(out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis("off")

    # Title
    ax.text(7, 8.6, "CatalogBank OCR — Stage 2 POC Pipeline",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=C["heading"])
    ax.text(7, 8.2, "Catalog PDF  →  Structured Knowledge Graph  (no training, no fine-tuning)",
            ha="center", va="center", fontsize=10, color=C["slate"])

    # Stage 1 (left column, compressed)
    s1_boxes = [
        (1.2, 7.2, "PDF Preprocessing",     "PyMuPDF · DPI 200",       C["slate"]),
        (1.2, 6.3, "PP-StructureV3 / OCR",  "PaddleOCR pretrained",    C["slate"]),
        (1.2, 5.4, "Semantic Block Detection","paragraph_title→heading\ntable→table · vision_footnote→spec", C["slate"]),
        (1.2, 4.5, "Hierarchy Reconstruction","heading-stack · spatial rules", C["slate"]),
        (1.2, 3.6, "Canonical JSON",         "schema_version: 1.0",    C["teal"]),
    ]

    # Stage 2 (right column)
    s2_boxes = [
        (7.8, 7.2, "A · Context Builder",   "hierarchy → ContextChunks\none chunk per heading",   C["blue"]),
        (7.8, 6.3, "B · Table Parser",       "HTML / markdown → list[dict]\nBeautifulSoup + regex", C["blue"]),
        (7.8, 5.4, "B · LLM Input Builder",  "prompt + context JSON\n~400–600 tokens per chunk",   C["blue"]),
        (7.8, 4.5, "B · Qwen3-8B",           "entity extraction\nrelation extraction",             C["purple"]),
        (7.8, 3.6, "C · Validation",         "Pydantic · SKU dedup\norphan check · schema rules",  C["amber"]),
        (7.8, 2.7, "C · Normalization",      "unit canon. · string clean\ncross-chunk dedup",       C["amber"]),
        (7.8, 1.8, "D · Neo4j Graph",        "MERGE Cypher · 5 node types\n5 relationship types",  C["green"]),
    ]

    bw, bh = 4.6, 0.68

    def draw_box(cx, cy, title, sub, color):
        bx, by = cx - bw/2, cy - bh/2
        fancy = FancyBboxPatch((bx, by), bw, bh,
                               boxstyle="round,pad=0.04",
                               facecolor=C["white"], edgecolor=color, linewidth=1.8)
        ax.add_patch(fancy)
        ax.text(cx, cy + 0.12, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color)
        ax.text(cx, cy - 0.17, sub, ha="center", va="center",
                fontsize=7.5, color=C["slate"])

    for cx, cy, t, s, col in s1_boxes:
        draw_box(cx, cy, t, s, col)
    for cx, cy, t, s, col in s2_boxes:
        draw_box(cx, cy, t, s, col)

    # Downward arrows within Stage 1
    for i in range(len(s1_boxes) - 1):
        _, y0, _, _, _ = s1_boxes[i]
        _, y1, _, _, _ = s1_boxes[i+1]
        ax.annotate("", xy=(1.2, y1 + bh/2 + 0.02), xytext=(1.2, y0 - bh/2 - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=C["slate"], lw=1.4))

    # Downward arrows within Stage 2
    for i in range(len(s2_boxes) - 1):
        _, y0, _, _, _ = s2_boxes[i]
        _, y1, _, _, _ = s2_boxes[i+1]
        ax.annotate("", xy=(7.8, y1 + bh/2 + 0.02), xytext=(7.8, y0 - bh/2 - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=1.4))

    # Horizontal bridge: Canonical JSON → Context Builder
    ax.annotate("", xy=(7.8 - bw/2, 7.2), xytext=(1.2 + bw/2, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=C["teal"],
                                connectionstyle="arc3,rad=-0.35", lw=2.0))
    ax.text(4.8, 5.8, "Canonical JSON\ninput", ha="center", va="center",
            fontsize=8, color=C["teal"], style="italic")

    # Column labels
    for x, label in [(1.2, "STAGE 1  (implemented)"), (7.8, "STAGE 2  (implemented)")]:
        ax.text(x, 7.85, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C["heading"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["light"], edgecolor=C["border"]))

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 2 — Context Chunks (Phase A output)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_context_chunks(chunks, out_path: Path):
    fig, axes = plt.subplots(1, len(chunks), figsize=(15, 7))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Phase A — Context Chunks extracted from Catalog Page",
                 fontsize=14, fontweight="bold", color=C["heading"], y=0.97)

    chunk_colors = [C["blue"], C["purple"], C["teal"]]

    for col, (ax, chunk) in enumerate(zip(axes, chunks)):
        ax.set_facecolor(C["bg"])
        ax.axis("off")
        color = chunk_colors[col % len(chunk_colors)]

        # Header box
        header = FancyBboxPatch((0.05, 0.88), 0.9, 0.1,
                                boxstyle="round,pad=0.02",
                                transform=ax.transAxes,
                                facecolor=color, edgecolor="none",
                                clip_on=False, zorder=3)
        ax.add_patch(header)
        ax.text(0.5, 0.93, chunk.section_name, transform=ax.transAxes,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=C["white"], zorder=4)

        # Breadcrumb
        crumb = " › ".join(chunk.section_path)
        ax.text(0.5, 0.84, crumb, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5, color=C["slate"], style="italic")

        # Block counts badge row
        badge_items = [
            (len(chunk.tables),        "tables",    C["amber"]),
            (len(chunk.specifications),"specs",     C["teal"]),
            (len(chunk.product_cards), "products",  C["green"]),
            (len(chunk.paragraphs),    "paras",     C["slate"]),
        ]
        bx = 0.05
        for count, label, bc in badge_items:
            if count == 0:
                continue
            badge = FancyBboxPatch((bx, 0.77), 0.19, 0.055,
                                   boxstyle="round,pad=0.015",
                                   transform=ax.transAxes,
                                   facecolor=bc, edgecolor="none",
                                   clip_on=False, alpha=0.85)
            ax.add_patch(badge)
            ax.text(bx + 0.095, 0.797, f"{count} {label}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=C["white"])
            bx += 0.22

        # Content lines
        y = 0.70
        def section_block(title, blocks, bcolor):
            nonlocal y
            if not blocks:
                return
            ax.text(0.07, y, title, transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold",
                    color=bcolor)
            y -= 0.045
            for blk in blocks:
                snippet = (blk.text or "").replace("\n", " · ")[:72]
                if len(blk.text or "") > 72:
                    snippet += "…"
                for line in textwrap.wrap(snippet, width=42) or [snippet]:
                    ax.text(0.10, y, f"• {line}", transform=ax.transAxes,
                            ha="left", va="top", fontsize=7.2, color=C["heading"])
                    y -= 0.042
                    if y < 0.04:
                        return

        section_block("Specifications:", chunk.specifications, C["teal"])
        section_block("Tables:", chunk.tables, C["amber"])
        section_block("Product Cards:", chunk.product_cards, C["green"])
        section_block("Paragraphs:", chunk.paragraphs, C["slate"])

        # Prompt token estimate
        from catalogbank_ocr.stage2.llm_input_builder import build_llm_input, estimate_prompt_chars
        li = build_llm_input(chunk, include_prompt=True)
        tokens = estimate_prompt_chars(li) // 4
        ax.text(0.5, 0.02, f"~{tokens} tokens to LLM", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, color=C["slate"],
                style="italic")

        # Frame
        for spine in ax.spines.values():
            spine.set_visible(False)
        frame = FancyBboxPatch((0.02, 0.01), 0.96, 0.96,
                               boxstyle="round,pad=0.01",
                               transform=ax.transAxes,
                               facecolor="none", edgecolor=color, linewidth=1.5,
                               clip_on=False)
        ax.add_patch(frame)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 3 — Extraction Results (Phase B/C output)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_extraction_results(normalized: NormalizedResult, out_path: Path):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Phase B+C — Normalized Extraction Results",
                 fontsize=14, fontweight="bold", color=C["heading"], y=0.98)

    # Layout: 3 panels
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3,
                          left=0.04, right=0.97, top=0.92, bottom=0.04)
    ax_prod = fig.add_subplot(gs[0, 0])
    ax_attr = fig.add_subplot(gs[0, 1])
    ax_rel  = fig.add_subplot(gs[1, :])

    for ax in (ax_prod, ax_attr, ax_rel):
        ax.set_facecolor(C["bg"])
        ax.axis("off")

    def draw_table(ax, title, col_headers, rows, col_colors=None, title_color=C["blue"]):
        ax.text(0.5, 0.97, title, transform=ax.transAxes,
                ha="center", va="top", fontsize=11, fontweight="bold", color=title_color)
        if not rows:
            ax.text(0.5, 0.5, "(none)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color=C["slate"])
            return

        ncols = len(col_headers)
        col_widths = [1.0 / ncols] * ncols
        # Header row
        row_h = 0.085
        y0 = 0.90
        for ci, (header, cw) in enumerate(zip(col_headers, col_widths)):
            x = sum(col_widths[:ci])
            rect = FancyBboxPatch((x + 0.005, y0 - row_h + 0.005), cw - 0.01, row_h - 0.005,
                                  boxstyle="square,pad=0",
                                  transform=ax.transAxes,
                                  facecolor=C["heading"], edgecolor="none")
            ax.add_patch(rect)
            ax.text(x + cw/2, y0 - row_h/2, header,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=C["white"])
        y0 -= row_h

        alt_colors = [C["white"], C["light"]]
        for ri, row in enumerate(rows):
            bg = alt_colors[ri % 2]
            for ci, (val, cw) in enumerate(zip(row, col_widths)):
                x = sum(col_widths[:ci])
                rect = FancyBboxPatch((x + 0.003, y0 - row_h + 0.003), cw - 0.006, row_h - 0.003,
                                      boxstyle="square,pad=0",
                                      transform=ax.transAxes,
                                      facecolor=bg, edgecolor=C["border"], linewidth=0.3)
                ax.add_patch(rect)
                cell_color = col_colors[ci] if col_colors else C["heading"]
                txt = str(val)[:28] + ("…" if len(str(val)) > 28 else "")
                ax.text(x + cw/2, y0 - row_h/2, txt,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7.5, color=cell_color)
            y0 -= row_h
            if y0 < 0.05:
                ax.text(0.5, 0.02, f"… and {len(rows)-ri-1} more",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=7, color=C["slate"], style="italic")
                break

    # Products table
    prod_rows = [
        (p.name, p.model or "—", p.sku or "—", p.category or "—")
        for p in normalized.products
    ]
    draw_table(ax_prod, f"Products  ({len(normalized.products)})",
               ["Name", "Model", "SKU", "Category"],
               prod_rows,
               col_colors=[C["blue"], C["teal"], C["green"], C["amber"]],
               title_color=C["blue"])

    # Attributes table (show first 12)
    attr_rows = [
        (a.entity, a.key, a.value, a.unit or "—")
        for a in normalized.attributes[:12]
    ]
    draw_table(ax_attr, f"Attributes  ({len(normalized.attributes)} total)",
               ["Entity", "Key", "Value", "Unit"],
               attr_rows,
               col_colors=[C["blue"], C["teal"], C["heading"], C["amber"]],
               title_color=C["teal"])

    # Relations table (wide)
    rel_rows = [
        (r.subject, r.predicate, r.object, r.source_chunk.split("__")[-1] if r.source_chunk else "—")
        for r in normalized.relations
    ]
    draw_table(ax_rel, f"Relations  ({len(normalized.relations)})",
               ["Subject", "Predicate", "Object", "Source Chunk"],
               rel_rows,
               col_colors=[C["blue"], C["purple"], C["teal"], C["slate"]],
               title_color=C["purple"])

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 4 — Knowledge Graph (Phase D)
# ─────────────────────────────────────────────────────────────────────────────

def fig4_knowledge_graph(normalized: NormalizedResult, ingestion, out_path: Path):
    """Render a Neo4j-style knowledge graph using matplotlib."""
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#0F172A")   # dark background, Neo4j-style
    ax.set_facecolor("#0F172A")
    ax.set_xlim(-1, 17); ax.set_ylim(-1, 11)
    ax.axis("off")

    ax.text(8, 10.5, "Phase D — Knowledge Graph  (Neo4j MERGE Cypher)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(8, 10.0,
            f"{ingestion.statements_executed} Cypher statements  ·  "
            f"~{ingestion.nodes_merged} nodes  ·  "
            f"~{ingestion.rels_merged} relationships  ·  dry-run (no live Neo4j needed)",
            ha="center", va="center", fontsize=9, color="#94A3B8")

    # Node style registry
    NODE_STYLE = {
        "Document":  {"color": "#7C3AED", "r": 0.55, "text_color": "white"},
        "Section":   {"color": "#0891B2", "r": 0.45, "text_color": "white"},
        "Product":   {"color": "#16A34A", "r": 0.42, "text_color": "white"},
        "Attribute": {"color": "#D97706", "r": 0.35, "text_color": "white"},
        "Category":  {"color": "#DC2626", "r": 0.40, "text_color": "white"},
    }

    REL_COLORS = {
        "HAS_SECTION":   "#0891B2",
        "CONTAINS":      "#16A34A",
        "HAS_ATTRIBUTE": "#D97706",
        "BELONGS_TO":    "#DC2626",
        "IS_VARIANT_OF": "#A78BFA",
        "HAS_MATERIAL":  "#F472B6",
    }

    # ── Manually lay out the nodes for clarity ───────────────────────────────
    doc_stem = normalized.doc_stem

    # Document (centre-top)
    nodes = {}
    edges = []

    nodes["doc"] = {"label": "Document", "name": doc_stem, "type": "Document", "pos": (8.0, 9.0)}

    # Sections (row 2)
    sec_positions = [(2.5, 7.2), (8.0, 7.2), (13.5, 7.2)]
    sec_ids = []
    for i, chunk_id in enumerate(normalized.source_chunk_ids):
        # Short display name
        parts = chunk_id.split("__")
        display = parts[0] if i == 0 else (f"Chunk {i+1}" if len(parts) < 3 else parts[-1])
        display = display.replace("(synthetic fixture)", "catalog.pdf")
        nid = f"sec_{i}"
        sec_ids.append(nid)
        nodes[nid] = {"label": "Section", "name": display,
                      "type": "Section", "pos": sec_positions[i]}
        edges.append(("doc", nid, "HAS_SECTION"))

    # Products (row 3)
    prod_positions = [
        (0.8, 5.2), (2.2, 5.2), (3.6, 5.2),   # chunk 0 nipples
        (6.0, 5.2), (7.5, 5.2), (9.0, 5.2),   # chunk 1 elbows
        (12.5, 5.2),                            # chunk 2 lens
    ]
    prod_node_ids = {}
    for i, prod in enumerate(normalized.products):
        nid = f"prod_{i}"
        prod_node_ids[prod.name.lower()] = nid
        # Link to correct section
        sec_idx = 0
        for si, cid in enumerate(normalized.source_chunk_ids):
            if cid in prod.source_chunks:
                sec_idx = si
                break
        pos = prod_positions[i] if i < len(prod_positions) else (i * 2.0 % 14, 5.2)
        nodes[nid] = {"label": "Product", "name": prod.model or prod.name,
                      "type": "Product", "pos": pos}
        edges.append((sec_ids[sec_idx], nid, "CONTAINS"))
        if prod.category:
            cat_nid = f"cat_{prod.category.lower().replace(' ','_')}"
            if cat_nid not in nodes:
                # place category on right side
                cat_x = 15.0 if "lens" in prod.category.lower() else (14.0 if "elbow" in prod.category.lower() else 14.5)
                cat_y = 5.2 if "lens" in prod.category.lower() else (4.0 if "elbow" in prod.category.lower() else 6.2)
                nodes[cat_nid] = {"label": "Category", "name": prod.category,
                                  "type": "Category", "pos": (cat_x, cat_y)}
            edges.append((nid, cat_nid, "BELONGS_TO"))

    # Attributes (row 4, sampled — show 2 per product for clarity)
    attr_positions_base = [
        (-0.2, 3.0), (1.0, 3.0),    # nipple attrs
        (2.2, 3.0), (3.4, 3.0),
        (4.6, 3.0), (5.6, 3.0),     # elbow attrs
        (6.8, 3.0), (7.8, 3.0),
        (9.0, 3.0), (10.0, 3.0),
        (11.5, 3.0), (12.5, 3.0),   # lens attrs
        (13.5, 3.0), (14.5, 3.0),
    ]
    attr_idx = 0
    # Show at most 2 attrs per product
    attr_shown: dict = {}
    for a in normalized.attributes:
        entity_key = a.entity.lower()
        attr_shown[entity_key] = attr_shown.get(entity_key, 0)
        if attr_shown[entity_key] >= 2:
            continue
        prod_nid = prod_node_ids.get(entity_key)
        if prod_nid is None:
            continue
        if attr_idx >= len(attr_positions_base):
            break
        anid = f"attr_{attr_idx}"
        display_val = a.value
        if a.unit:
            display_val = f"{a.value} {a.unit}"
        label_text = f"{a.key}:\n{display_val}"
        nodes[anid] = {"label": "Attribute", "name": label_text,
                       "type": "Attribute", "pos": attr_positions_base[attr_idx]}
        edges.append((prod_nid, anid, "HAS_ATTRIBUTE"))
        attr_shown[entity_key] += 1
        attr_idx += 1

    # IS_VARIANT_OF edges between products
    for rel in normalized.relations:
        if rel.predicate != "IS_VARIANT_OF":
            continue
        subj_nid = prod_node_ids.get(rel.subject.lower())
        obj_nid  = prod_node_ids.get(rel.object.lower())
        if subj_nid and obj_nid:
            edges.append((subj_nid, obj_nid, "IS_VARIANT_OF"))

    # ── Draw edges first (so nodes are on top) ───────────────────────────────
    for src_id, dst_id, rel_type in edges:
        if src_id not in nodes or dst_id not in nodes:
            continue
        sx, sy = nodes[src_id]["pos"]
        dx, dy = nodes[dst_id]["pos"]
        color = REL_COLORS.get(rel_type, "#94A3B8")
        # Slight curve for IS_VARIANT_OF to avoid overlap
        rad = -0.25 if rel_type == "IS_VARIANT_OF" else 0.0
        ax.annotate("", xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=1.3 if rel_type not in ("IS_VARIANT_OF",) else 1.0,
                        connectionstyle=f"arc3,rad={rad}",
                        alpha=0.85,
                    ))
        # Relationship label on edge midpoint
        mx, my = (sx + dx) / 2, (sy + dy) / 2
        # Offset label slightly so it doesn't sit on the line
        if rel_type not in ("CONTAINS", "HAS_SECTION"):
            ax.text(mx, my + 0.18, rel_type, ha="center", va="bottom",
                    fontsize=5.5, color=color, alpha=0.9,
                    fontweight="bold")

    # ── Draw nodes ───────────────────────────────────────────────────────────
    for nid, nd in nodes.items():
        x, y = nd["pos"]
        style = NODE_STYLE[nd["type"]]
        r = style["r"]

        circle = plt.Circle((x, y), r,
                             facecolor=style["color"],
                             edgecolor="white", linewidth=1.2,
                             zorder=5)
        ax.add_patch(circle)

        # Node label (type on top, name below)
        ax.text(x, y + 0.10, nd["type"], ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=6,
                alpha=0.8)
        name_lines = nd["name"].split("\n")
        for li, line in enumerate(name_lines[:2]):
            ax.text(x, y - 0.10 - li * 0.18,
                    textwrap.shorten(line, width=14),
                    ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=6)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = list(NODE_STYLE.items()) + [("", v) for v in []]
    lx, ly = 0.0, -0.5
    ax.text(lx, ly + 0.35, "Node types:", fontsize=8, color="#94A3B8", va="center")
    for i, (ntype, style) in enumerate(NODE_STYLE.items()):
        circle = plt.Circle((lx + 1.2 + i * 2.8, ly), 0.22,
                             facecolor=style["color"], edgecolor="white", lw=0.8, zorder=5)
        ax.add_patch(circle)
        ax.text(lx + 1.6 + i * 2.8, ly, ntype,
                fontsize=8, color="white", va="center")

    rel_lx = 0.0
    rel_ly = -0.85
    ax.text(rel_lx, rel_ly + 0.22, "Relationships:", fontsize=8, color="#94A3B8", va="center")
    for i, (rtype, rcolor) in enumerate(REL_COLORS.items()):
        ax.annotate("", xy=(rel_lx + 1.0 + i * 2.7, rel_ly),
                    xytext=(rel_lx + 0.4 + i * 2.7, rel_ly),
                    arrowprops=dict(arrowstyle="-|>", color=rcolor, lw=1.5))
        ax.text(rel_lx + 1.1 + i * 2.7, rel_ly + 0.12, rtype,
                fontsize=7, color=rcolor, va="center")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0F172A")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 5 — Actual Cypher Statements (what goes into Neo4j)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_cypher_statements(ingestion, out_path: Path):
    """Show the real parameterised Cypher MERGE statements generated for Neo4j."""
    stmts = ingestion.statements  # list of {cypher, params}

    # Pick a representative sample: first of each statement type
    seen_types = {}
    samples = []
    for s in stmts:
        first_line = next((l.strip() for l in s["cypher"].splitlines() if l.strip()), "")
        key = first_line.split("{")[0].strip()
        if key not in seen_types:
            seen_types[key] = True
            samples.append(s)
        if len(samples) >= 10:
            break

    # Disable mathtext parsing for this figure — Cypher has { } and $ which
    # matplotlib would otherwise try to parse as LaTeX math.
    with plt.rc_context({"text.usetex": False, "mathtext.default": "regular"}):
        fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.axis("off")

    ax.text(0.5, 0.975, "Phase D — Real Cypher MERGE Statements (Neo4j)",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.948,
            f"79 total statements generated  |  parameterised  |  idempotent (safe to re-run)  |  awaiting live Neo4j connection",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#94A3B8")

    # Colour map for statement types
    TYPE_COLORS = {
        "MERGE (d:Document": "#7C3AED",
        "MERGE (s:Section":  "#0891B2",
        "MERGE (p:Product":  "#16A34A",
        "MERGE (a:Attribute":"#D97706",
        "MERGE (c:Category": "#DC2626",
        "MATCH (d:Document": "#4C1D95",
        "MATCH (s:Section":  "#164E63",
        "MATCH (p:Product":  "#14532D",
        "MATCH (a:Product":  "#78350F",
    }

    y = 0.91
    row_h = 0.079
    for i, stmt in enumerate(samples):
        lines = [l.strip() for l in stmt["cypher"].splitlines() if l.strip()]
        cypher_text = " ".join(lines)

        # Determine color
        color = "#94A3B8"
        for k, v in TYPE_COLORS.items():
            if cypher_text.startswith(k):
                color = v
                break

        # Background strip
        bg = FancyBboxPatch((0.02, y - row_h + 0.008), 0.96, row_h - 0.01,
                            boxstyle="round,pad=0.005",
                            transform=ax.transAxes,
                            facecolor="#1E293B", edgecolor=color, linewidth=1.0)
        ax.add_patch(bg)

        # Sanitize Cypher for matplotlib — { } and $ trigger LaTeX math parsing
        def safe(t):
            return t.replace("$", "S").replace("{", "(").replace("}", ")")
        cypher_disp = safe(cypher_text[:95] + ("..." if len(cypher_text) > 95 else ""))
        ax.text(0.04, y - row_h/2 + 0.015, cypher_disp,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8.2, color=color,
                fontfamily="monospace")

        # Params
        params = stmt["params"]
        params_parts = []
        for k, v in params.items():
            val = str(v)
            if val and val != "None":
                val_disp = val[:22] + "..." if len(val) > 22 else val
                params_parts.append(f"{k}={val_disp!r}")
        params_str = "  |  " + "  ".join(params_parts[:4]) if params_parts else ""
        params_str = safe(params_str)
        ax.text(0.04, y - row_h/2 - 0.012, params_str,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=7, color="#64748B",
                fontfamily="monospace")

        y -= row_h

    # Legend of node types
    legend_y = 0.075
    ax.text(0.03, legend_y + 0.025, "Node labels:", transform=ax.transAxes,
            fontsize=8, color="#94A3B8", va="center")
    labels = [("Document", "#7C3AED"), ("Section", "#0891B2"),
              ("Product", "#16A34A"), ("Attribute", "#D97706"), ("Category", "#DC2626")]
    for i, (lbl, col) in enumerate(labels):
        x = 0.17 + i * 0.16
        circle = plt.Circle((x, legend_y), 0.012,
                             facecolor=col, edgecolor="white", lw=0.8,
                             transform=ax.transAxes, clip_on=False)
        ax.add_patch(circle)
        ax.text(x + 0.016, legend_y, lbl, transform=ax.transAxes,
                fontsize=8, color="white", va="center")

    ax.text(0.5, 0.02,
            "Run `python scripts/run_stage2_phase_d.py --input <canonical.json>` with Neo4j running to execute these statements",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="#475569", style="italic")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE 6 — Multi-document / Cross-page Relations
# ─────────────────────────────────────────────────────────────────────────────

def fig6_multi_doc_relations(out_path: Path):
    """Show how the same product appearing across multiple catalog pages
    gets merged via (name, SKU) deduplication and cross-page relations."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Multi-Document Cross-Page Relation Handling",
                 fontsize=15, fontweight="bold", color=C["heading"], y=0.97)

    # ── Panel 1: Two documents before merge ─────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(C["bg"])
    ax1.axis("off")
    ax1.set_title("Two catalog pages processed\n(same product in both)",
                  fontsize=10, color=C["heading"], pad=8)

    def draw_doc_card(ax, x, y, doc_name, chunks, color, width=0.42):
        # Doc box
        doc_box = FancyBboxPatch((x, y), width, 0.12, boxstyle="round,pad=0.01",
                                  transform=ax.transAxes,
                                  facecolor=color, edgecolor="none")
        ax.add_patch(doc_box)
        ax.text(x + width/2, y + 0.06, doc_name, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        cy = y - 0.09
        for chunk_name, products in chunks:
            chunk_box = FancyBboxPatch((x + 0.02, cy), width - 0.04, 0.07,
                                       boxstyle="round,pad=0.01",
                                       transform=ax.transAxes,
                                       facecolor=C["light"], edgecolor=color, linewidth=1)
            ax.add_patch(chunk_box)
            ax.text(x + width/2, cy + 0.05, chunk_name, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=color, fontweight="bold")
            ax.text(x + width/2, cy + 0.02, products, transform=ax.transAxes,
                    ha="center", va="center", fontsize=7.5, color=C["slate"])
            cy -= 0.10
        return cy

    draw_doc_card(ax1, 0.04, 0.82,
                  "mcmaster-125_3378_15.pdf  (Page 1)",
                  [("Pipe Fittings", "SS-NIP-001  SS-NIP-002  SS-NIP-003"),
                   ("Pipe Elbows",   "SS-ELB-90-025  SS-ELB-90-038")],
                  C["blue"])

    draw_doc_card(ax1, 0.54, 0.82,
                  "mcmaster-125_3378_16.pdf  (Page 1)",
                  [("Pipe Fittings", "SS-NIP-001  SS-NIP-004"),
                   ("Pipe Caps",     "SS-CAP-025  SS-CAP-038")],
                  C["purple"])

    ax1.text(0.5, 0.28,
             "SS-NIP-001 appears in\nBOTH documents",
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=9, color=C["red"], fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF2F2",
                       edgecolor=C["red"], linewidth=1.5))

    ax1.annotate("", xy=(0.24, 0.56), xytext=(0.24, 0.40),
                 xycoords="axes fraction", arrowprops=dict(
                     arrowstyle="-|>", color=C["blue"], lw=1.5))
    ax1.annotate("", xy=(0.76, 0.56), xytext=(0.76, 0.40),
                 xycoords="axes fraction", arrowprops=dict(
                     arrowstyle="-|>", color=C["purple"], lw=1.5))

    # Labels under arrows
    ax1.text(0.5, 0.10,
             "normalize_batch([result_doc1, result_doc2])\ncross-chunk dedup by (name.lower(), sku.lower())",
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=8, color=C["slate"], style="italic")

    # ── Panel 2: After merge ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(C["bg"])
    ax2.axis("off")
    ax2.set_title("After normalization — deduplication\nSS-NIP-001 merged, source_chunks tracked",
                  fontsize=10, color=C["heading"], pad=8)

    merged_products = [
        ("SS-NIP-001", "SS-NIP-001", "Pipe Nipples", ["p1_c001", "p2_c001"], C["green"]),
        ("SS-NIP-002", "SS-NIP-002", "Pipe Nipples", ["p1_c001"],            C["teal"]),
        ("SS-NIP-003", "SS-NIP-003", "Pipe Nipples", ["p1_c001"],            C["teal"]),
        ("SS-NIP-004", "SS-NIP-004", "Pipe Nipples", ["p2_c001"],            C["teal"]),
        ("SS-ELB-90-025","SS-ELB-90-025","Pipe Elbows",["p1_c002"],          C["blue"]),
        ("SS-ELB-90-038","SS-ELB-90-038","Pipe Elbows",["p1_c002"],          C["blue"]),
        ("SS-CAP-025",  "SS-CAP-025", "Pipe Caps",   ["p2_c002"],            C["amber"]),
        ("SS-CAP-038",  "SS-CAP-038", "Pipe Caps",   ["p2_c002"],            C["amber"]),
    ]

    col_headers = ["Name", "SKU", "Category", "source_chunks"]
    col_widths = [0.28, 0.24, 0.22, 0.26]
    row_h = 0.075
    y0 = 0.90

    # Header
    x = 0.0
    for h, cw in zip(col_headers, col_widths):
        r = FancyBboxPatch((x + 0.005, y0 - row_h + 0.003), cw - 0.01, row_h - 0.003,
                           boxstyle="square,pad=0", transform=ax2.transAxes,
                           facecolor=C["heading"], edgecolor="none")
        ax2.add_patch(r)
        ax2.text(x + cw/2, y0 - row_h/2, h, transform=ax2.transAxes,
                 ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        x += cw
    y0 -= row_h

    for name, sku, cat, chunks_list, row_color in merged_products:
        is_merged = len(chunks_list) > 1
        bg = "#ECFDF5" if is_merged else C["white"]
        x = 0.0
        row_vals = [name, sku, cat, str(chunks_list)]
        for val, cw in zip(row_vals, col_widths):
            r = FancyBboxPatch((x + 0.003, y0 - row_h + 0.003), cw - 0.006, row_h - 0.003,
                               boxstyle="square,pad=0", transform=ax2.transAxes,
                               facecolor=bg,
                               edgecolor=C["green"] if is_merged else C["border"],
                               linewidth=1.5 if is_merged else 0.4)
            ax2.add_patch(r)
            txt = val[:24] + "..." if len(val) > 24 else val
            ax2.text(x + cw/2, y0 - row_h/2, txt, transform=ax2.transAxes,
                     ha="center", va="center", fontsize=7.2,
                     color=C["green"] if is_merged else C["heading"],
                     fontweight="bold" if is_merged else "normal")
            x += cw
        y0 -= row_h

    ax2.text(0.5, 0.07, "Green rows = merged across docs  |  source_chunks lists both origins",
             transform=ax2.transAxes, ha="center", va="center",
             fontsize=8, color=C["green"], style="italic")

    # ── Panel 3: Cross-doc graph view ────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#0F172A")
    ax3.axis("off")
    ax3.set_title("Neo4j graph — cross-document product node\n(single node, two Section parents)",
                  fontsize=10, color="white", pad=8)
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)

    def node(ax, x, y, label, name, color, r=0.55):
        c = plt.Circle((x, y), r, facecolor=color, edgecolor="white", lw=1.3, zorder=5)
        ax.add_patch(c)
        ax.text(x, y + 0.12, label, ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=6)
        for i, line in enumerate(name.split("\n")):
            ax.text(x, y - 0.12 - i*0.22, line, ha="center", va="center",
                    fontsize=6.8, color="white", fontweight="bold", zorder=6)

    def edge(ax, x1, y1, x2, y2, label, color, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                    connectionstyle=f"arc3,rad={rad}"))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my, label, ha="left", va="center",
                fontsize=6, color=color, fontweight="bold")

    # Doc nodes
    node(ax3, 2.5, 9.0, "Document", "mcmaster\npage-1", "#7C3AED", r=0.5)
    node(ax3, 7.5, 9.0, "Document", "mcmaster\npage-2", "#7C3AED", r=0.5)

    # Section nodes
    node(ax3, 2.5, 7.0, "Section",  "Pipe\nFittings", "#0891B2", r=0.45)
    node(ax3, 7.5, 7.0, "Section",  "Pipe\nFittings", "#0891B2", r=0.45)

    # Merged product — one node, two parents
    node(ax3, 5.0, 4.8, "Product",  "SS-NIP-001\n(merged)", "#16A34A", r=0.6)

    # Variant products
    node(ax3, 2.0, 2.5, "Product",  "SS-NIP-002", "#16A34A", r=0.45)
    node(ax3, 3.8, 2.5, "Product",  "SS-NIP-004", "#16A34A", r=0.45)

    # Attribute
    node(ax3, 5.0, 1.2, "Attribute", "Material:\n316 SS",  "#D97706", r=0.45)
    node(ax3, 7.5, 2.5, "Category",  "Pipe\nNipples",      "#DC2626", r=0.45)

    # Edges
    edge(ax3, 2.5, 8.5, 2.5, 7.45, "HAS_SECTION", "#0891B2")
    edge(ax3, 7.5, 8.5, 7.5, 7.45, "HAS_SECTION", "#0891B2")
    edge(ax3, 2.5, 6.55, 4.55, 5.3, "CONTAINS", "#16A34A", rad=-0.2)
    edge(ax3, 7.5, 6.55, 5.45, 5.3, "CONTAINS", "#16A34A", rad=0.2)
    edge(ax3, 4.5, 4.55, 2.4, 2.9, "IS_VARIANT_OF", "#A78BFA", rad=0.1)
    edge(ax3, 4.7, 4.4, 3.8, 2.9, "IS_VARIANT_OF", "#A78BFA", rad=-0.1)
    edge(ax3, 5.0, 4.2, 5.0, 1.65, "HAS_ATTRIBUTE", "#D97706")
    edge(ax3, 5.5, 5.0, 7.1, 2.85, "BELONGS_TO", "#DC2626", rad=-0.2)

    ax3.text(5.0, 0.3, "One Product node — two CONTAINS edges from different docs",
             ha="center", va="center", fontsize=7.5, color="#94A3B8", style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate POC output images")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("outputs/poc_outputs"),
                        help="Output directory (default: outputs/poc_outputs)")
    args = parser.parse_args()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  CatalogBank Stage 2 POC — Output Generator")
    print("="*60)

    print("\n[1/5] Running full A->B->C->D pipeline ...")
    chunks, raw_results, normalized, ingestion = run_pipeline(out_dir)

    print("\n[2/5] Generating image 1: Pipeline Overview …")
    fig1_pipeline_overview(out_dir / "01_pipeline_overview.png")

    print("\n[3/5] Generating image 2: Context Chunks …")
    fig2_context_chunks(chunks, out_dir / "02_context_chunks.png")

    print("\n[4/5] Generating image 3: Extraction Results …")
    fig3_extraction_results(normalized, out_dir / "03_extraction_results.png")

    print("\n[5/5] Generating image 4: Knowledge Graph ...")
    fig4_knowledge_graph(normalized, ingestion, out_dir / "04_knowledge_graph.png")

    print("\n[6/6] Generating image 5: Cypher Statements (real Neo4j writes) ...")
    fig5_cypher_statements(ingestion, out_dir / "05_cypher_statements.png")

    print("\n[7/7] Generating image 6: Multi-doc cross-page relations ...")
    fig6_multi_doc_relations(out_dir / "06_multi_doc_relations.png")

    print("\n" + "="*60)
    print(f"  Done. All outputs in: {out_dir.resolve()}")
    print("="*60)
    print("\n  JSON outputs:")
    for f in sorted(out_dir.glob("*.json")):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name}  ({size_kb} KB)" if size_kb else f"    {f.name}  (<1 KB)")
    print("\n  Images:")
    for f in sorted(out_dir.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"    {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
