"""Stage 2 Phase C — Validation & Normalization.

Chains Phase A (context builder) → Phase B (LLM extraction) → Phase C
(validation + normalization) and prints a clean summary of the final output.

Usage
-----
# Dry-run with synthetic fixture and mock LLM (no GPU needed):
    python scripts/run_stage2_phase_c.py --fixture --backend mock

# Against a real canonical JSON with Ollama:
    python scripts/run_stage2_phase_c.py --input outputs/canonical/my_doc/my_doc_document.json

# Write normalized output to disk:
    python scripts/run_stage2_phase_c.py --fixture --backend mock --output outputs/stage2/phase_c

# Suppress Phase A/B progress noise, show only Phase C results:
    python scripts/run_stage2_phase_c.py --fixture --backend mock --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catalogbank_ocr.stage2.context_builder import build_context_chunks, summarise_chunks
from catalogbank_ocr.stage2.llm_input_builder import build_llm_input
from catalogbank_ocr.stage2.llm_extractor import LLMExtractor, ExtractionResult
from catalogbank_ocr.stage2.validation import validate_batch, ValidationReport, Severity
from catalogbank_ocr.stage2.normalization import normalize_batch, NormalizedResult

sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from run_stage2_phase_a import SYNTHETIC_CANONICAL


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_validation(reports: Dict[str, ValidationReport], verbose: bool) -> None:
    total_errors = sum(len(r.errors) for r in reports.values())
    total_warnings = sum(len(r.warnings) for r in reports.values())
    total_info = sum(
        len([i for i in r.issues if i.severity == Severity.INFO])
        for r in reports.values()
    )
    print(f"\n[Phase C — Validation]")
    print(f"  {total_errors} error(s)  {total_warnings} warning(s)  {total_info} info(s)")

    if verbose:
        for chunk_id, report in reports.items():
            if report.issues:
                print(f"  ┌ {chunk_id}")
                report.print_report(indent=4)
        if not any(r.issues for r in reports.values()):
            print("  ✓ All chunks passed validation with no issues")


def _print_normalized(normalized: NormalizedResult) -> None:
    print(f"\n[Phase C — Normalized Output]  doc={normalized.doc_stem!r}")
    print(f"  products   : {len(normalized.products)}")
    print(f"  attributes : {len(normalized.attributes)}")
    print(f"  relations  : {len(normalized.relations)}")

    if normalized.products:
        print("\n  Products:")
        for p in normalized.products:
            sku_str = f"  sku={p.sku!r}" if p.sku else ""
            model_str = f"  model={p.model!r}" if p.model else ""
            cat_str = f"  category={p.category!r}" if p.category else ""
            chunks_str = f"  chunks={p.source_chunks}"
            print(f"    • {p.name!r}{model_str}{sku_str}{cat_str}{chunks_str}")

    if normalized.attributes:
        print("\n  Attributes:")
        for a in normalized.attributes:
            unit_str = f" [{a.unit}]" if a.unit else ""
            num_str = f" (numeric={a.value_numeric})" if a.value_numeric is not None else ""
            print(f"    • {a.entity!r}  {a.key}: {a.value}{unit_str}{num_str}")

    if normalized.relations:
        print("\n  Relations:")
        for r in normalized.relations:
            print(f"    • {r.subject!r}  --[{r.predicate}]-->  {r.object!r}")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_canonical(
    canonical_doc: dict,
    extractor: LLMExtractor,
    source_path: str = "(unknown)",
    output_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[NormalizedResult, Dict[str, ValidationReport]]:

    print(f"\n{'='*64}")
    print(f"  Source : {source_path}")
    print(f"  Backend: {extractor.backend_name}")
    print(f"{'='*64}")

    # ---- Phase A: Context Builder ----------------------------------------
    chunks = build_context_chunks(canonical_doc, source_doc_path=source_path, skip_empty=True)
    if verbose:
        print(f"\n[Phase A] {len(chunks)} chunk(s)")
        print(summarise_chunks(chunks))
    else:
        print(f"[Phase A] {len(chunks)} chunk(s) extracted")

    if not chunks:
        empty = NormalizedResult(doc_stem=Path(source_path).stem)
        return empty, {}

    # ---- Phase B: LLM Extraction -----------------------------------------
    llm_inputs = [build_llm_input(c, include_prompt=True) for c in chunks]
    print(f"\n[Phase B] Extracting with {extractor.backend_name} …")
    extraction_results: List[ExtractionResult] = extractor.extract_batch(
        llm_inputs, progress=True
    )

    # ---- Phase C: Validation ---------------------------------------------
    reports = validate_batch(extraction_results)
    _print_validation(reports, verbose=verbose)

    # ---- Phase C: Normalization ------------------------------------------
    normalized = normalize_batch(extraction_results)
    _print_normalized(normalized)

    # ---- Write outputs ---------------------------------------------------
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalized result
        norm_path = output_dir / "normalized.json"
        norm_path.write_text(
            json.dumps(normalized.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Validation report
        val_report = {
            chunk_id: report.to_dict()
            for chunk_id, report in reports.items()
        }
        val_path = output_dir / "validation_report.json"
        val_path.write_text(
            json.dumps(val_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Raw extractions for debugging
        raw_path = output_dir / "raw_extractions.json"
        raw_path.write_text(
            json.dumps(
                [r.to_dict() for r in extraction_results],
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"\n  ✓ Wrote normalized.json, validation_report.json, raw_extractions.json → {output_dir}")

    return normalized, reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_canonical_jsons(root: Path) -> List[Path]:
    return sorted(root.glob("outputs/canonical/**/*_document.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 Phase C — Validation & Normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="Path to a Stage 1 canonical JSON file.")
    parser.add_argument("--fixture", action="store_true",
                        help="Use built-in synthetic fixture.")
    parser.add_argument("--backend", choices=["ollama", "transformers", "mock"], default=None)
    parser.add_argument("--model", default=None,
                        help="Model name override.")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Directory to write output files.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress Phase A/B detail; show Phase C results only.")
    args = parser.parse_args()

    print(f"Initialising extractor (backend={args.backend or 'auto'}) …")
    extractor = LLMExtractor(
        backend=args.backend,
        model=args.model,
        ollama_host=args.ollama_host,
    )
    print(f"  → {extractor.backend_name}")

    verbose = not args.quiet

    if args.fixture:
        out = args.output / "fixture" if args.output else None
        process_canonical(SYNTHETIC_CANONICAL, extractor,
                          source_path="(synthetic fixture)",
                          output_dir=out, verbose=verbose)
        return

    if args.input:
        if not args.input.exists():
            print(f"ERROR: {args.input} not found", file=sys.stderr)
            sys.exit(1)
        with args.input.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        out = args.output / args.input.stem if args.output else None
        process_canonical(doc, extractor, source_path=str(args.input),
                          output_dir=out, verbose=verbose)
        return

    # Auto-discover
    canonical_files = _find_canonical_jsons(_REPO_ROOT)
    if not canonical_files:
        print("No canonical JSONs found. Run with --fixture to test:")
        print("  python scripts/run_stage2_phase_c.py --fixture --backend mock")
        return

    for path in canonical_files:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        out = args.output / path.stem if args.output else None
        process_canonical(doc, extractor, source_path=str(path),
                          output_dir=out, verbose=verbose)


if __name__ == "__main__":
    main()
