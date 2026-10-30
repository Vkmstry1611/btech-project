"""LLM- and OCR-free pipeline: reproduce the final catalog from saved artifacts.

This is the *deployment* pipeline. It does **not** call any LLM and does **not**
run PaddleOCR. Instead it reuses the Stage-2 extractions that were previously
captured (``outputs/stage2/<stem>/extractions.json``) and re-runs the
deterministic part of the pipeline:

    Phase A  — rebuild context chunks from the saved canonical JSON
    Phase C  — evidence validation + normalization
    Export   — build the final catalog tree (``build_catalog_json``)

Outputs are written under ``outputs/deploy/`` so the saved research artifacts
under ``outputs/`` stay untouched and remain the source of truth::

    outputs/deploy/final/<stem>/<stem>_final.json
    outputs/deploy/stage2/<stem>/{normalized,validation,review_issues}.json
    outputs/deploy/metrics/<stem>/pipeline.json
    outputs/deploy/run_summary.json

Usage
-----
    # process every saved canonical document
    python scripts/run_mock_pipeline.py

    # restrict to specific stems
    python scripts/run_mock_pipeline.py --stem industrial_fittings_5_pages

    # custom input/output roots
    python scripts/run_mock_pipeline.py --input outputs --output outputs/deploy
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — make ``src`` importable without an installed package.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Imports (all pure-Python Stage-2 modules; no paddle/torch/LLM).
# ---------------------------------------------------------------------------
from catalogbank_ocr.stage2.context_builder import build_context_chunks
from catalogbank_ocr.stage2.llm_input_builder import build_complete_inputs
from catalogbank_ocr.stage2.llm_extractor import ExtractionResult
from catalogbank_ocr.stage2.evidence_validation import validate_evidence
from catalogbank_ocr.stage2.validation import validate_batch
from catalogbank_ocr.stage2.normalization import normalize_batch
from catalogbank_ocr.stage2.catalog_export import build_catalog_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-24s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mock_pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_stems(canonical_root: Path) -> List[str]:
    """Return the list of document stems that have a saved canonical JSON."""
    stems: List[str] = []
    if not canonical_root.exists():
        return stems
    for child in sorted(canonical_root.iterdir()):
        if not child.is_dir():
            continue
        doc = child / f"{child.name}_document.json"
        if doc.exists():
            stems.append(child.name)
    return stems


def _count_catalog(node: Dict[str, Any]) -> Dict[str, int]:
    """Walk the catalog tree and count categories / models / items / prices."""
    counts = {"categories": 0, "models": 0, "items": 0, "prices": 0}

    def visit(n: Dict[str, Any]) -> None:
        t = n.get("type")
        if t == "Product Category":
            counts["categories"] += 1
        elif t == "Model":
            counts["models"] += 1
        elif t == "Sellable Item":
            counts["items"] += 1
        elif t == "List Price":
            counts["prices"] += 1
        for key in ("subcategories", "models", "sellableItems", "listPrices"):
            for child in n.get(key, []) or []:
                visit(child)

    visit(node)
    return counts


# ---------------------------------------------------------------------------
# Core per-document processing
# ---------------------------------------------------------------------------
def process_stem(
    stem: str,
    input_root: Path,
    output_root: Path,
    backend_label: str,
) -> Dict[str, Any]:
    """Reproduce the final catalog for one document from saved artifacts."""
    t0 = time.monotonic()
    canonical_path = input_root / "canonical" / stem / f"{stem}_document.json"
    extractions_path = input_root / "stage2" / stem / "extractions.json"
    saved_final_path = input_root / "final" / stem / f"{stem}_final.json"

    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical JSON not found: {canonical_path}")
    if not extractions_path.exists():
        raise FileNotFoundError(
            f"Saved extractions not found: {extractions_path}. "
            "Run the real pipeline once (or with --backend ollama) to capture them."
        )

    canonical_doc = _read_json(canonical_path)
    pdf_path = canonical_doc.get("source", {}).get("pdf_path", "")

    # Phase A — rebuild context chunks (deterministic, pure Python). Used as a
    # structural sanity check; the saved extractions already carry the table-row
    # expansions, so we do NOT require a 1:1 match with the rebuilt chunks.
    chunks = build_context_chunks(canonical_doc, source_doc_path=str(canonical_path), skip_empty=True)
    llm_inputs = build_complete_inputs(chunks)
    logger.info("[%s] Phase A: %d chunk(s)", stem, len(llm_inputs))

    # Reuse saved Stage-2 extractions (no LLM call). These are the post-table-
    # reconciliation results, so they may outnumber the context chunks.
    saved = _read_json(extractions_path)
    extraction_results = [ExtractionResult.from_dict(item) for item in saved]
    logger.info("[%s] Reusing %d saved extraction(s) (backend=%s)", stem, len(extraction_results), backend_label)

    # The saved review issues were produced by evidence validation during the
    # real run; reuse them so the export handles ambiguous prices identically.
    saved_issues_path = input_root / "stage2" / stem / "review_issues.json"
    review_issues: List[Any] = _read_json(saved_issues_path) if saved_issues_path.exists() else []

    # Recompute evidence issues only when the rebuilt chunks line up exactly;
    # otherwise the saved issues are the authoritative record.
    rebuilt_ids = [req.get("chunk_id") for req in llm_inputs]
    saved_ids = [r.chunk_id for r in extraction_results]
    if rebuilt_ids == saved_ids and len(saved_ids) > 0:
        try:
            review_issues = validate_evidence(extraction_results, llm_inputs)
        except Exception as exc:  # noqa: BLE001 - keep saved issues on failure
            logger.debug("[%s] evidence recompute failed (%s); using saved issues", stem, exc)

    # Phase C — validation + normalization, then export (deterministic, no
    # LLM/OCR/PDF). If anything goes wrong we fall back to the previously
    # exported final catalog so the deployment always has a valid artifact.
    normalized_dict: Optional[Dict[str, Any]] = None
    validation_dict: Optional[Dict[str, Any]] = None
    try:
        reports = validate_batch(extraction_results)
        normalized = normalize_batch(extraction_results)
        logger.info(
            "[%s] Phase C: %d product(s), %d attr(s), %d relation(s)",
            stem,
            len(normalized.products),
            len(normalized.attributes),
            len(normalized.relations),
        )
        normalized_dict = normalized.to_dict()
        validation_dict = {k: v.to_dict() for k, v in reports.items()}

        # Export — build the final catalog tree (deterministic, hash-based ids).
        final_doc = build_catalog_json(
            canonical_doc,
            normalized,
            backend_used=backend_label,
            pdf_path=pdf_path,
            review_issues=review_issues,
        )
        reproduced = True
        logger.info("[%s] Export: reproduced catalog from saved extractions", stem)
    except Exception as exc:  # noqa: BLE001 - fall back to saved final
        logger.warning("[%s] reproduction failed (%s); using saved final catalog", stem, exc)
        if not saved_final_path.exists():
            raise FileNotFoundError(
                f"Cannot reproduce and no saved final catalog at {saved_final_path}"
            ) from exc
        final_doc = _read_json(saved_final_path)
        reproduced = False

    # ---- write deployment outputs ----
    final_path = _save_json(output_root / "final" / stem / f"{stem}_final.json", final_doc)
    _save_json(output_root / "stage2" / stem / "normalized.json", normalized_dict or {})
    _save_json(output_root / "stage2" / stem / "validation.json", validation_dict or {})
    _save_json(output_root / "stage2" / stem / "review_issues.json", review_issues)

    counts = _count_catalog(final_doc)
    elapsed = time.monotonic() - t0
    metrics = {
        "stem": stem,
        "backend": backend_label,
        "reproduced_from_extractions": reproduced,
        "chunks": len(llm_inputs),
        "elapsed_s": round(elapsed, 3),
        **counts,
    }
    _save_json(output_root / "metrics" / stem / "pipeline.json", metrics)
    logger.info("[%s] ✓ final -> %s  (%s)", stem, final_path, counts)
    return {"stem": stem, "reproduced": reproduced, "final_path": str(final_path), **counts}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Reproduce the final catalog from saved artifacts (no LLM, no OCR).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, default=_REPO_ROOT / "outputs",
                        help="Root of saved research artifacts (default: outputs/).")
    parser.add_argument("--output", "-o", type=Path, default=_REPO_ROOT / "outputs" / "deploy",
                        help="Deployment output root (default: outputs/deploy/).")
    parser.add_argument("--stem", action="append", default=None,
                        help="Restrict to specific document stem(s). Repeatable.")
    parser.add_argument("--backend-label", default="mock-reuse",
                        help="Label recorded as the backend in metrics (default: mock-reuse).")
    args = parser.parse_args()

    canonical_root = args.input / "canonical"
    all_stems = _discover_stems(canonical_root)
    stems = args.stem if args.stem else all_stems
    if not stems:
        print(f"No canonical documents found under {canonical_root}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Mock deployment pipeline")
    print(f"  input   : {args.input}")
    print(f"  output  : {args.output}")
    print(f"  stems   : {len(stems)} -> {', '.join(stems)}")
    print(f"  backend : {args.backend_label} (no LLM, no OCR)")
    print("=" * 64)

    results: List[Dict[str, Any]] = []
    failed: List[str] = []
    for stem in stems:
        try:
            results.append(process_stem(stem, args.input, args.output, args.backend_label))
        except Exception as exc:
            logger.exception("Failed to process %s: %s", stem, exc)
            failed.append(stem)

    summary = {
        "pipeline": "catalogbank-ocr mock deployment",
        "backend": args.backend_label,
        "total": len(stems),
        "succeeded": len(results),
        "failed": len(failed),
        "failed_stems": failed,
        "documents": results,
    }
    _save_json(args.output / "run_summary.json", summary)

    print("=" * 64)
    print(f"Done. {len(results)} succeeded, {len(failed)} failed.")
    for r in results:
        print(
            f"  {r['stem']:<40} "
            f"items={r['items']:>4} models={r['models']:>3} prices={r['prices']:>4} "
            f"({'reproduced' if r['reproduced'] else 'saved-fallback'})"
        )
    if failed:
        print("Failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
