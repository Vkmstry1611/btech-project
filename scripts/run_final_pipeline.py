"""Final end-to-end pipeline: Stage 1 (multi-page) + Stage 2 (LLM extraction).

Produces one self-contained output JSON per PDF:

    outputs/final/{stem}/{stem}_final.json
    outputs/visualizations/{stem}/{stem}_knowledge_graph.png

Usage
-----
# Process all PDFs in pdfs/ folder (needs Ollama running):
    python scripts/run_final_pipeline.py --input pdfs

# Mock LLM (no GPU needed, for testing):
    python scripts/run_final_pipeline.py --input pdfs --backend mock

# Skip Stage 1 if canonical JSON already exists (re-run Stage 2 only):
    python scripts/run_final_pipeline.py --input pdfs --skip-stage1

# Custom Ollama model:
    python scripts/run_final_pipeline.py --input pdfs --model qwen3:14b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Disable oneDNN and force CPU — PaddlePaddle has no CUDA 13.x build yet.
# Must be set before any paddle import.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPU from paddle

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from catalogbank_ocr.preprocessing.image_preprocessing import ImagePreprocessingConfig
from catalogbank_ocr.research_stage1_pipeline import run_stage1_for_pdf_multipage, Stage1MultiPageResult
from catalogbank_ocr.stage2.context_builder import build_context_chunks
from catalogbank_ocr.stage2.llm_input_builder import build_llm_input
from catalogbank_ocr.stage2.llm_extractor import LLMExtractor
from catalogbank_ocr.stage2.normalization import normalize_batch
from catalogbank_ocr.stage2.validation import validate_batch
from catalogbank_ocr.visualization.research_visualization import visualize_knowledge_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_final_pipeline")


# ---------------------------------------------------------------------------
# Catalog tree builder — converts hierarchy + extractions into the target schema
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Make a short uppercase identifier from text."""
    import re
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()
    words = text.upper().split()[:4]
    return "-".join(words) if words else "UNKNOWN"


def _build_sellable_items(
    product_name: str,
    attributes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build sellable item list from a product and its attributes."""
    price_attrs = [a for a in attributes if a.get("entity", "").lower() == product_name.lower()
                   and any(k in a.get("key", "").lower() for k in ("price", "cost", "amount"))]
    other_attrs = [a for a in attributes if a.get("entity", "").lower() == product_name.lower()
                   and not any(k in a.get("key", "").lower() for k in ("price", "cost", "amount"))]

    # Build properties dict from non-price attributes
    properties = {}
    for a in other_attrs:
        val = a.get("value", "")
        if a.get("unit"):
            val = f"{val} {a['unit']}"
        properties[a.get("key", "").lower().replace(" ", "_")] = val

    # Build list prices
    list_prices = []
    for i, a in enumerate(price_attrs):
        amount_raw = a.get("value_numeric") or a.get("value", "0")
        try:
            amount = float(str(amount_raw).replace(",", ""))
        except ValueError:
            amount = 0.0
        currency = a.get("unit", "USD")
        location = a.get("source_chunk", "").split("__")[0] or "Unknown"
        list_prices.append({
            "type": "List Price",
            "title": f"{product_name} — {location}",
            "identifier": f"LP-{_slugify(product_name)}-{i+1:02d}",
            "amount": amount,
            "currency": currency,
            "location": location,
        })

    item: Dict[str, Any] = {
        "type": "Sellable Item",
        "title": product_name,
        "identifier": f"SI-{_slugify(product_name)}",
    }
    if properties:
        # Use SKU as articleNumber if available
        if "sku" in properties:
            item["articleNumber"] = properties.pop("sku")
        item["properties"] = properties
    if list_prices:
        item["listPrices"] = list_prices

    return [item]


def _build_model(product: Dict[str, Any], attributes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a Model node from a normalized product."""
    name = product.get("name", "Unknown")
    sku = product.get("sku") or product.get("model") or ""
    desc = product.get("description") or (f"Model: {product['model']}" if product.get("model") and product.get("model") != name else None)
    model: Dict[str, Any] = {
        "type": "Model",
        "title": name,
        "identifier": f"MOD-{_slugify(name)}",
    }
    if desc:
        model["description"] = desc
    if sku:
        model["sku"] = sku

    # Pass article number (SKU) into sellable items
    sellable = _build_sellable_items_with_sku(name, sku, attributes)
    if sellable:
        model["sellableItems"] = sellable

    return model


def _build_sellable_items_with_sku(
    product_name: str,
    article_number: str,
    attributes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build sellable items with articleNumber and cleaned prices."""
    price_attrs = [a for a in attributes if a.get("entity", "").lower() == product_name.lower()
                   and any(k in a.get("key", "").lower() for k in ("price", "cost", "amount"))]
    other_attrs = [a for a in attributes if a.get("entity", "").lower() == product_name.lower()
                   and not any(k in a.get("key", "").lower() for k in ("price", "cost", "amount"))]

    properties = {}
    for a in other_attrs:
        val = a.get("value", "")
        if a.get("unit"):
            val = f"{val} {a['unit']}"
        properties[a.get("key", "").lower().replace(" ", "_")] = val

    list_prices = []
    for i, a in enumerate(price_attrs):
        amount_raw = a.get("value_numeric") or a.get("value", "0")
        try:
            amount = float(str(amount_raw).replace(",", ""))
        except ValueError:
            amount = 0.0
        # Normalize currency
        raw_unit = (a.get("unit") or "USD").upper().strip().split()[0]
        currency = raw_unit if raw_unit in {"USD", "EUR", "GBP", "JPY", "CNY", "INR", "AUD"} else "USD"
        list_prices.append({
            "type": "List Price",
            "title": f"{product_name} — {currency}",
            "identifier": f"LP-{_slugify(product_name)}-{i+1:02d}",
            "amount": amount,
            "currency": currency,
        })

    item: Dict[str, Any] = {
        "type": "Sellable Item",
        "title": product_name,
        "identifier": f"SI-{_slugify(product_name)}",
    }
    if article_number:
        item["articleNumber"] = article_number
    if properties:
        item["properties"] = properties
    if list_prices:
        item["listPrices"] = list_prices

    return [item]


def _build_category_from_hierarchy(
    node: Dict[str, Any],
    products: List[Dict[str, Any]],
    attributes: List[Dict[str, Any]],
    depth: int = 0,
) -> Dict[str, Any]:
    """Recursively build catalog category tree from hierarchy node.

    Only real extracted products (from LLM extraction) become Models.
    Paragraphs, body text and other non-product blocks are ignored.
    """
    name = node.get("name", "Unknown")
    children = node.get("children", [])

    category: Dict[str, Any] = {
        "type": "Product Category",
        "title": name,
        "identifier": f"PC-{_slugify(name)}",
    }

    subcategories = []
    models = []

    # Only recurse into heading children — paragraphs/tables don't become categories
    for child in children:
        child_type = child.get("semantic_type", "")
        if child_type == "heading":
            subcategories.append(_build_category_from_hierarchy(child, products, attributes, depth + 1))
        # product_card nodes map to models only if there's a matching extracted product
        elif child_type == "product_card":
            child_name = child.get("name", "")
            # Only add if there's a real extracted product matching this card
            matching = [p for p in products
                        if p.get("name", "").lower() in child_name.lower()
                        or child_name.lower() in p.get("name", "").lower()]
            for p in matching:
                models.append(_build_model(p, attributes))

    # Find products whose source_chunks mention this section's heading
    section_products = [
        p for p in products
        if any(name.lower() in (chunk or "").lower() for chunk in p.get("source_chunks", []))
    ]
    for p in section_products:
        already = any(m.get("title", "").lower() == p.get("name", "").lower() for m in models)
        if not already:
            models.append(_build_model(p, attributes))

    if subcategories:
        category["subcategories"] = subcategories
    if models:
        category["models"] = models

    return category


def build_catalog_json(
    canonical_doc: Dict[str, Any],
    normalized_result: Any,
    backend_used: str,
    pdf_path: str = "",
) -> Dict[str, Any]:
    """Build the final catalog JSON in the target hierarchical schema."""
    import os as _os
    import re as _re
    pdf_name = _os.path.basename(pdf_path).replace(".pdf", "") if pdf_path else "Document"

    products = [p.to_dict() for p in normalized_result.products]
    attributes = [a.to_dict() for a in normalized_result.attributes]

    # Build a lookup: product name (lower) → model dict
    # Skip products that are clearly OCR noise or section headings
    _PRODUCT_NOISE = {"chapters", "sections", "contents", "index", "overview", "introduction"}

    def _is_product_noise(name: str) -> bool:
        n = name.strip().lower()
        return n in _PRODUCT_NOISE or (n.isupper() and len(n.split()) <= 1)

    all_models_by_name: Dict[str, Any] = {}
    for p in products:
        pname = p.get("name", "")
        if _is_product_noise(pname):
            continue
        m = _build_model(p, attributes)
        all_models_by_name[pname.lower()] = m

    hierarchy = canonical_doc.get("hierarchy", {})
    children = hierarchy.get("children", [])

    # Filter noise headings
    _NOISE = {"sections", "chapters", "tables", "contents", "index", "appendix"}

    def _is_noise(name: str) -> bool:
        n = name.strip().lower().rstrip("/").strip()
        return n in _NOISE or (n.isupper() and len(n.split()) <= 2)

    def _find_models_for_section(heading_name: str) -> List[Dict[str, Any]]:
        """Find products whose category or name matches this section heading."""
        found = []
        heading_lower = heading_name.lower()
        for p in products:
            cat = (p.get("category") or "").lower()
            name = (p.get("name") or "").lower()
            # Match if the product's category contains the heading name or vice versa
            if (cat and (cat in heading_lower or heading_lower in cat)) or \
               (name and name in heading_lower):
                m = all_models_by_name.get(name)
                if m:
                    found.append(m)
        return found

    def _build_cat(node: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        name = node.get("name", "Unknown")
        ch   = node.get("children", [])

        cat: Dict[str, Any] = {
            "type": "Product Category",
            "title": name,
            "identifier": f"PC-{_slugify(name)}-{abs(hash(name)) % 10000:04d}",
        }

        subcats = []
        models  = []
        used_model_ids: set = set()

        for child in ch:
            ctype = child.get("semantic_type", "")
            if ctype == "heading":
                if not _is_noise(child.get("name", "")):
                    subcats.append(_build_cat(child, depth + 1))
            elif ctype == "product_card":
                cname = child.get("name", "").lower()
                m = all_models_by_name.get(cname)
                if m and m.get("identifier") not in used_model_ids:
                    used_model_ids.add(m.get("identifier"))
                    models.append(m)

        # Also attach products whose category field matches this heading
        for m in _find_models_for_section(name):
            if m.get("identifier") not in used_model_ids:
                used_model_ids.add(m.get("identifier"))
                models.append(m)

        if subcats:
            cat["subcategories"] = subcats
        if models:
            cat["models"] = models

        return cat

    subcategories = []
    seen_cat_ids: set = set()
    placed_model_ids: set = set()

    for child in children:
        if child.get("semantic_type") in ("heading", "document", "paragraph"):
            name = child.get("name", "")
            if _is_noise(name):
                continue
            cat = _build_cat(child)
            cid = cat.get("identifier", "")
            # Avoid duplicate identifiers at top level
            if cid in seen_cat_ids:
                cat["identifier"] = f"{cid}-{len(seen_cat_ids):02d}"
            seen_cat_ids.add(cat.get("identifier", ""))

            # Track which models got placed
            for m in cat.get("models", []):
                placed_model_ids.add(m.get("identifier", ""))
            for sc in cat.get("subcategories", []):
                for m in sc.get("models", []):
                    placed_model_ids.add(m.get("identifier", ""))

            if cat.get("subcategories") or cat.get("models"):
                subcategories.append(cat)

    # Any remaining unplaced products go under a flat "Products" category
    unplaced = [m for m in all_models_by_name.values()
                if m.get("identifier") not in placed_model_ids]
    if unplaced:
        if subcategories:
            subcategories.append({
                "type": "Product Category",
                "title": "Other Products",
                "identifier": "PC-OTHER-PRODUCTS",
                "models": unplaced,
            })
        else:
            subcategories = [{
                "type": "Product Category",
                "title": "Products",
                "identifier": "PC-PRODUCTS",
                "models": list(all_models_by_name.values()),
            }]

    source_info = canonical_doc.get("source", {})
    total_pages = source_info.get("total_pages", 1)

    catalog: Dict[str, Any] = {
        "type": "Product Category",
        "title": pdf_name,
        "identifier": f"PC-{_slugify(pdf_name)}",
        "description": f"Product catalog extracted from {pdf_name} ({total_pages} page(s))",
    }
    if subcategories:
        catalog["subcategories"] = subcategories

    return catalog


# Keep old build_final_json as alias for backward compat
def build_final_json(
    canonical_doc: Dict[str, Any],
    normalized_result: Any,
    backend_used: str,
) -> Dict[str, Any]:
    return build_catalog_json(
        canonical_doc, normalized_result, backend_used,
        pdf_path=canonical_doc.get("source", {}).get("pdf_path", ""),
    )


# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    output_root: Path,
    extractor: LLMExtractor,
    preprocess_config: ImagePreprocessingConfig,
    skip_stage1: bool,
    ocr_engine: Any = None,
) -> Path:
    """Run the full pipeline for one PDF and write the final JSON.

    Returns the path to the written final JSON file.
    """
    stem = pdf_path.stem
    final_dir = output_root / "final" / stem
    final_dir.mkdir(parents=True, exist_ok=True)
    final_json_path = final_dir / f"{stem}_final.json"

    # ------------------------------------------------------------------ #
    # Stage 1 — multi-page structural pipeline
    # ------------------------------------------------------------------ #
    canonical_json_path = output_root / "canonical" / stem / f"{stem}_document.json"

    if skip_stage1 and canonical_json_path.exists():
        logger.info("[%s] Skipping Stage 1 — using existing canonical JSON", stem)
        with canonical_json_path.open("r", encoding="utf-8") as fh:
            canonical_doc = json.load(fh)
        total_pages = canonical_doc.get("source", {}).get("total_pages", "?")
        logger.info("[%s] Loaded canonical JSON (%s pages)", stem, total_pages)
    else:
        logger.info("[%s] Running Stage 1 (multi-page)…", stem)
        t0 = time.monotonic()
        stage1_result: Stage1MultiPageResult = run_stage1_for_pdf_multipage(
            pdf_path=pdf_path,
            output_root=output_root,
            preprocess_config=preprocess_config,
            ocr_engine=ocr_engine,
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "[%s] Stage 1 done — %d pages in %.1fs → %s",
            stem, stage1_result.total_pages, elapsed, stage1_result.canonical_json_path,
        )
        with stage1_result.canonical_json_path.open("r", encoding="utf-8") as fh:
            canonical_doc = json.load(fh)

    # ------------------------------------------------------------------ #
    # Stage 2 — LLM extraction (phases A → B → C)
    # ------------------------------------------------------------------ #
    logger.info("[%s] Running Stage 2 (backend=%s)…", stem, extractor.backend_name)

    # Phase A — context chunks
    chunks = build_context_chunks(
        canonical_doc,
        source_doc_path=str(canonical_json_path),
        skip_empty=True,
    )
    logger.info("[%s] Phase A: %d chunk(s)", stem, len(chunks))

    if not chunks:
        logger.warning("[%s] No extractable chunks — writing empty extraction.", stem)
        from catalogbank_ocr.stage2.normalization import NormalizedResult
        normalized = NormalizedResult(doc_stem=stem)
    else:
        # Phase B — LLM extraction
        llm_inputs = [build_llm_input(c, include_prompt=True) for c in chunks]
        t0 = time.monotonic()
        extraction_results = extractor.extract_batch(llm_inputs, progress=True)
        elapsed = time.monotonic() - t0
        logger.info("[%s] Phase B done in %.1fs", stem, elapsed)

        # Phase C — validation + normalization
        reports   = validate_batch(extraction_results)
        normalized = normalize_batch(extraction_results)

        errors   = sum(len(r.errors) for r in reports.values())
        warnings = sum(len(r.warnings) for r in reports.values())
        logger.info(
            "[%s] Phase C: %d product(s), %d attr(s), %d relation(s) | "
            "%d error(s), %d warning(s)",
            stem,
            len(normalized.products), len(normalized.attributes), len(normalized.relations),
            errors, warnings,
        )

        # Save per-document normalization for inspection
        norm_path = output_root / "stage2" / stem / "normalized.json"
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        norm_path.write_text(
            json.dumps(normalized.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Knowledge graph visualization
    # ------------------------------------------------------------------ #
    try:
        kg_viz_path = output_root / "visualizations" / stem / f"{stem}_knowledge_graph.png"
        visualize_knowledge_graph(
            normalized_result=normalized,
            output_path=kg_viz_path,
            doc_stem=stem,
        )
        logger.info("[%s] ✓ KG visualization → %s", stem, kg_viz_path)
    except Exception as exc:
        logger.warning("[%s] KG visualization failed (non-fatal): %s", stem, exc)

    # ------------------------------------------------------------------ #
    # Merge → final JSON
    # ------------------------------------------------------------------ #
    final_doc = build_final_json(canonical_doc, normalized, extractor.backend_name)
    final_json_path.write_text(
        json.dumps(final_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[%s] ✓ Final JSON written → %s", stem, final_json_path)
    return final_json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _collect_pdfs(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        pdfs = sorted(input_path.glob("**/*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found under {input_path}")
        return pdfs
    raise ValueError(f"--input must be a PDF file or a directory, got: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: Stage 1 (multi-page OCR) + Stage 2 (LLM extraction) → final JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / output
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="PDF file or directory of PDFs to process.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output root directory (default: outputs/ next to this script).")

    # Stage 1 options
    parser.add_argument("--dpi", type=int, default=200,
                        help="Rendering DPI for PDF pages (default: 200).")
    parser.add_argument("--deskew", action="store_true", help="Enable deskew preprocessing.")
    parser.add_argument("--denoise", action="store_true", help="Enable denoising preprocessing.")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip Stage 1 if canonical JSON already exists.")

    # Stage 2 / LLM options
    parser.add_argument("--backend", choices=["ollama", "transformers", "mock"], default=None,
                        help="LLM backend (default: auto-detect Ollama → transformers → mock).")
    parser.add_argument("--model", default=None,
                        help="LLM model name (e.g. qwen3:8b for Ollama).")
    parser.add_argument("--ollama-host", default=None,
                        help="Ollama host URL if not localhost.")

    args = parser.parse_args()

    # Resolve output root
    output_root = args.output or (_REPO_ROOT / "outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect PDFs
    try:
        pdfs = _collect_pdfs(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(pdfs)} PDF(s) to process.")
    print(f"Output root : {output_root}")
    print(f"LLM backend : {args.backend or 'auto'}")
    print()

    # Build preprocessing config
    preprocess_config = ImagePreprocessingConfig(
        dpi=args.dpi,
        first_page_only=False,  # always multi-page in this script
        deskew=args.deskew,
        denoise=args.denoise,
    )

    # Build LLM extractor (shared across all PDFs)
    print(f"Initialising LLM extractor (backend={args.backend or 'auto'})…")
    extractor = LLMExtractor(
        backend=args.backend,
        model=args.model,
        ollama_host=args.ollama_host,
    )
    print(f"  → {extractor.backend_name}\n")

    # Initialize OCR engine once (expensive)
    ocr_engine = None
    if not args.skip_stage1:
        from catalogbank_ocr.ocr.ppstructure import initialize_ppstructurev3
        print("Initialising OCR engine (PPStructureV3)…")
        ocr_engine = initialize_ppstructurev3()
        print("  → ready\n")

    # Process each PDF
    final_paths: List[Path] = []
    failed: List[str] = []

    for i, pdf_path in enumerate(pdfs, start=1):
        print(f"{'='*64}")
        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
        print(f"{'='*64}")
        try:
            final_path = process_pdf(
                pdf_path=pdf_path,
                output_root=output_root,
                extractor=extractor,
                preprocess_config=preprocess_config,
                skip_stage1=args.skip_stage1,
                ocr_engine=ocr_engine,
            )
            final_paths.append(final_path)
        except Exception as exc:
            logger.exception("Failed to process %s: %s", pdf_path.name, exc)
            failed.append(pdf_path.name)

    # Summary
    print(f"\n{'='*64}")
    print(f"Done. {len(final_paths)} succeeded, {len(failed)} failed.")
    if final_paths:
        print("\nFinal JSON files:")
        for p in final_paths:
            print(f"  {p}")
    if failed:
        print("\nFailed:")
        for name in failed:
            print(f"  {name}")


if __name__ == "__main__":
    main()
