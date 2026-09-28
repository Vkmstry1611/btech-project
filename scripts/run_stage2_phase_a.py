"""Stage 2 Phase A — end-to-end test / demo script.

Usage
-----
# Against a real Stage 1 canonical JSON:
    python scripts/run_stage2_phase_a.py --input outputs/canonical/my_doc/my_doc_document.json

# Auto-discover all canonical JSONs under outputs/:
    python scripts/run_stage2_phase_a.py

# Run against the built-in synthetic fixture (no Stage 1 outputs needed):
    python scripts/run_stage2_phase_a.py --fixture

# Write LLM input JSONs to disk:
    python scripts/run_stage2_phase_a.py --fixture --output outputs/stage2/phase_a

What it does
------------
1.  Loads a canonical JSON (real or synthetic).
2.  Runs Context Builder  → list[ContextChunk]
3.  Runs Table Parser     → list[ParsedTable] per chunk
4.  Runs LLM Input Builder → list[dict] (prompt + context)
5.  Prints a human-readable summary and (optionally) writes JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure the src/ tree is importable regardless of how the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catalogbank_ocr.stage2.context_builder import (
    build_context_chunks,
    build_context_chunks_from_file,
    summarise_chunks,
)
from catalogbank_ocr.stage2.table_parser import parse_table_blocks
from catalogbank_ocr.stage2.llm_input_builder import (
    build_llm_input,
    estimate_prompt_chars,
)


# ---------------------------------------------------------------------------
# Synthetic fixture — exercises every block type without needing real outputs
# ---------------------------------------------------------------------------

SYNTHETIC_CANONICAL: dict = {
    "schema_version": "1.0",
    "source": {
        "pdf_path": "/fake/catalog.pdf",
        "page_number": 1,
        "rendered_image_path": "/fake/catalog.png",
        "preprocessed_image_path": "/fake/catalog_processed.png",
        "raw_json_path": "/fake/catalog_res.json",
    },
    "pages": [
        {
            "page": 1,
            "blocks": []  # Phase A reads from hierarchy, not pages[]
        }
    ],
    "hierarchy": {
        "name": "document",
        "semantic_type": "document",
        "block_id": None,
        "page": None,
        "children": [
            {
                "name": "Stainless Steel Pipe Fittings",
                "semantic_type": "heading",
                "block_id": "p1_b001",
                "page": 1,
                "source_type": "paragraph_title",
                "bbox": [81, 13, 702, 59],
                "confidence": 0.97,
                "order": 1,
                "metadata": {"heading_level": 1, "source_ref": "parsing_res_list[0]", "raw": {}},
                "children": [
                    {
                        "name": "Pipe Nipples",
                        "semantic_type": "heading",
                        "block_id": "p1_b002",
                        "page": 1,
                        "source_type": "paragraph_title",
                        "bbox": [81, 65, 400, 95],
                        "confidence": 0.95,
                        "order": 2,
                        "metadata": {"heading_level": 2, "source_ref": "parsing_res_list[1]", "raw": {}},
                        "children": [
                            {
                                "name": "Material: Stainless Steel 316\nThread: 1/4-20\nLength: 2 in\nPressure Rating: 3000 psi",
                                "semantic_type": "specification",
                                "block_id": "p1_b003",
                                "page": 1,
                                "source_type": "vision_footnote",
                                "bbox": [81, 100, 400, 160],
                                "confidence": 0.91,
                                "order": 3,
                                "metadata": {"source_ref": "parsing_res_list[2]", "raw": {"block_content": "Material: Stainless Steel 316\nThread: 1/4-20\nLength: 2 in\nPressure Rating: 3000 psi"}},
                                "children": [],
                            },
                            {
                                "name": "| Part No. | Size | Length | Price |\n|---|---|---|---|\n| SS-NIP-001 | 1/4 in | 2 in | $3.45 |\n| SS-NIP-002 | 3/8 in | 2 in | $4.10 |\n| SS-NIP-003 | 1/2 in | 3 in | $5.80 |",
                                "semantic_type": "table",
                                "block_id": "p1_b004",
                                "page": 1,
                                "source_type": "table",
                                "bbox": [81, 170, 702, 280],
                                "confidence": 0.96,
                                "order": 4,
                                "metadata": {
                                    "source_ref": "parsing_res_list[3]",
                                    "raw": {
                                        "block_content": "| Part No. | Size | Length | Price |\n|---|---|---|---|\n| SS-NIP-001 | 1/4 in | 2 in | $3.45 |\n| SS-NIP-002 | 3/8 in | 2 in | $4.10 |\n| SS-NIP-003 | 1/2 in | 3 in | $5.80 |"
                                    }
                                },
                                "children": [],
                            },
                            {
                                "name": "All fittings conform to ASME B16.11. Available with NPT or BSPT thread.",
                                "semantic_type": "paragraph",
                                "block_id": "p1_b005",
                                "page": 1,
                                "source_type": "text",
                                "bbox": [81, 285, 702, 310],
                                "confidence": 0.88,
                                "order": 5,
                                "metadata": {"source_ref": "parsing_res_list[4]", "raw": {}},
                                "children": [],
                            },
                        ],
                    },
                    {
                        "name": "Pipe Elbows",
                        "semantic_type": "heading",
                        "block_id": "p1_b006",
                        "page": 1,
                        "source_type": "paragraph_title",
                        "bbox": [81, 330, 400, 360],
                        "confidence": 0.93,
                        "order": 6,
                        "metadata": {"heading_level": 2, "source_ref": "parsing_res_list[5]", "raw": {}},
                        "children": [
                            {
                                "name": "SS-ELB-90-025",
                                "semantic_type": "product_card",
                                "block_id": "p1_b007",
                                "page": 1,
                                "source_type": "product_card",
                                "bbox": [81, 370, 350, 440],
                                "confidence": 0.85,
                                "order": 7,
                                "metadata": {"source_ref": "parsing_res_list[6]", "raw": {}},
                                "children": [
                                    {
                                        "name": "Material: 316 SS\nAngle: 90°\nSize: 1/4 in\nPrice: $6.20",
                                        "semantic_type": "specification",
                                        "block_id": "p1_b008",
                                        "page": 1,
                                        "source_type": "vision_footnote",
                                        "bbox": [90, 445, 340, 490],
                                        "confidence": 0.87,
                                        "order": 8,
                                        "metadata": {"source_ref": "parsing_res_list[7]", "raw": {}},
                                        "children": [],
                                    }
                                ],
                            },
                            {
                                "name": "<html><body><table><thead><tr><th>Model</th><th>Angle</th><th>Size</th><th>Material</th><th>Price</th></tr></thead><tbody><tr><td>SS-ELB-90-025</td><td>90°</td><td>1/4 in</td><td>316 SS</td><td>$6.20</td></tr><tr><td>SS-ELB-90-038</td><td>90°</td><td>3/8 in</td><td>316 SS</td><td>$7.50</td></tr><tr><td>SS-ELB-45-025</td><td>45°</td><td>1/4 in</td><td>304 SS</td><td>$5.90</td></tr></tbody></table></body></html>",
                                "semantic_type": "table",
                                "block_id": "p1_b009",
                                "page": 1,
                                "source_type": "table",
                                "bbox": [81, 500, 702, 600],
                                "confidence": 0.94,
                                "order": 9,
                                "metadata": {
                                    "source_ref": "parsing_res_list[8]",
                                    "raw": {
                                        "block_content": "<html><body><table><thead><tr><th>Model</th><th>Angle</th><th>Size</th><th>Material</th><th>Price</th></tr></thead><tbody><tr><td>SS-ELB-90-025</td><td>90°</td><td>1/4 in</td><td>316 SS</td><td>$6.20</td></tr><tr><td>SS-ELB-90-038</td><td>90°</td><td>3/8 in</td><td>316 SS</td><td>$7.50</td></tr><tr><td>SS-ELB-45-025</td><td>45°</td><td>1/4 in</td><td>304 SS</td><td>$5.90</td></tr></tbody></table></body></html>"
                                    }
                                },
                                "children": [],
                            },
                        ],
                    },
                ],
            },
            {
                "name": "Optical Lens Mounts",
                "semantic_type": "heading",
                "block_id": "p1_b010",
                "page": 1,
                "source_type": "paragraph_title",
                "bbox": [81, 620, 500, 660],
                "confidence": 0.96,
                "order": 10,
                "metadata": {"heading_level": 1, "source_ref": "parsing_res_list[9]", "raw": {}},
                "children": [
                    {
                        "name": "Lens Diameter: 25.4 mm\nMount Type: SM1 (1.035-40)\nMaterial: Anodized Aluminum\nCoating: None",
                        "semantic_type": "specification",
                        "block_id": "p1_b011",
                        "page": 1,
                        "source_type": "vision_footnote",
                        "bbox": [81, 665, 500, 720],
                        "confidence": 0.90,
                        "order": 11,
                        "metadata": {"source_ref": "parsing_res_list[10]", "raw": {}},
                        "children": [],
                    },
                    {
                        "name": "Compatible with all Thorlabs SM1-series lens tubes.",
                        "semantic_type": "paragraph",
                        "block_id": "p1_b012",
                        "page": 1,
                        "source_type": "text",
                        "bbox": [81, 725, 600, 750],
                        "confidence": 0.88,
                        "order": 12,
                        "metadata": {"source_ref": "parsing_res_list[11]", "raw": {}},
                        "children": [],
                    },
                ],
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_canonical(
    canonical_doc: dict,
    source_path: str = "(synthetic)",
    output_dir: Path | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Run Phase A on one canonical document.  Returns list of LLM input dicts."""

    print(f"\n{'='*64}")
    print(f"  Source: {source_path}")
    print(f"{'='*64}")

    # Step 1 — Context Builder
    chunks = build_context_chunks(canonical_doc, source_doc_path=source_path, skip_empty=True)
    print(f"\n[1] Context Builder → {len(chunks)} chunk(s) (empty sections skipped)")
    print(summarise_chunks(chunks))

    if not chunks:
        print("  ⚠  No extractable chunks found — check hierarchy structure.")
        return []

    # Step 2 + 3 — Table Parser + LLM Input Builder
    llm_inputs: list[dict] = []
    print(f"\n[2+3] Table Parser + LLM Input Builder")

    for chunk in chunks:
        # Parse tables in this chunk
        parsed_tables = parse_table_blocks(chunk.tables)
        table_summary = ""
        if parsed_tables:
            methods = [f"{t.block_id}→{t.parse_method}({len(t.rows)}r)" for t in parsed_tables]
            table_summary = f"  tables: {', '.join(methods)}"

        llm_input = build_llm_input(chunk, parsed_tables=parsed_tables, include_prompt=True)
        prompt_chars = estimate_prompt_chars(llm_input)
        approx_tokens = prompt_chars // 4

        print(
            f"  • {chunk.chunk_id}  "
            f"specs={len(chunk.specifications)}  "
            f"tables={len(chunk.tables)}  "
            f"products={len(chunk.product_cards)}  "
            f"paras={len(chunk.paragraphs)}  "
            f"~{approx_tokens} tokens"
            + (f"\n    {table_summary}" if table_summary else "")
        )

        if verbose and len(chunks) <= 6:
            _print_chunk_preview(llm_input)

        llm_inputs.append(llm_input)

    # Optionally write outputs
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for llm_input in llm_inputs:
            chunk_id = llm_input["chunk_id"]
            out_path = output_dir / f"{chunk_id}_llm_input.json"
            out_path.write_text(json.dumps(llm_input, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  ✓ Wrote {len(llm_inputs)} LLM input file(s) to: {output_dir}")

    return llm_inputs


def _print_chunk_preview(llm_input: dict) -> None:
    """Print a compact preview of one LLM input (context only, not full prompt)."""
    ctx = llm_input.get("context", {})
    print(f"    ↳ heading: {ctx.get('section_heading', '')!r}")
    for spec in ctx.get("specifications", [])[:2]:
        preview = spec[:120].replace("\n", " | ")
        print(f"      spec: {preview}")
    for pc in ctx.get("product_cards", [])[:2]:
        preview = pc[:80].replace("\n", " | ")
        print(f"      product: {preview}")
    for tbl in ctx.get("tables", [])[:1]:
        if "rows" in tbl:
            print(f"      table({tbl.get('parse_method','?')}): headers={tbl.get('headers',[])}  rows={len(tbl.get('rows',[]))}")
        else:
            raw_preview = str(tbl.get("raw_text", ""))[:80]
            print(f"      table(raw): {raw_preview!r}")
    sup = ctx.get("supporting_text", "")
    if sup:
        print(f"      support: {sup[:100]!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_canonical_jsons(root: Path) -> list[Path]:
    return sorted(root.glob("outputs/canonical/**/*_document.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Phase A demo / test script")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to a canonical JSON file produced by Stage 1.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run against the built-in synthetic fixture instead of real outputs.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Directory to write LLM input JSON files (optional).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-chunk content preview.",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if args.fixture:
        process_canonical(SYNTHETIC_CANONICAL, source_path="(synthetic fixture)", output_dir=args.output, verbose=verbose)
        return

    if args.input:
        if not args.input.exists():
            print(f"ERROR: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with args.input.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        process_canonical(doc, source_path=str(args.input), output_dir=args.output, verbose=verbose)
        return

    # Auto-discover
    repo_root = Path(__file__).resolve().parents[1]
    canonical_files = _find_canonical_jsons(repo_root)

    if not canonical_files:
        print("No canonical JSON files found under outputs/canonical/.")
        print("Run with --fixture to test against the built-in synthetic document:")
        print("  python scripts/run_stage2_phase_a.py --fixture")
        return

    print(f"Found {len(canonical_files)} canonical JSON file(s). Processing all.\n")
    for path in canonical_files:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        out_dir = args.output / path.stem if args.output else None
        process_canonical(doc, source_path=str(path), output_dir=out_dir, verbose=verbose)


if __name__ == "__main__":
    main()
