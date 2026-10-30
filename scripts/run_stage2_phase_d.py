"""Stage 2 Phase D — Neo4j Knowledge Graph ingestion.

Chains Phase A → B → C → D and writes the normalized extraction into Neo4j.

Usage
-----
# Dry-run with synthetic fixture (no Neo4j, no GPU needed):
    python scripts/run_stage2_phase_d.py --fixture --backend mock --dry-run

# Live against a canonical JSON (Neo4j must be running):
    python scripts/run_stage2_phase_d.py --input outputs/canonical/my_doc/my_doc_document.json

# Dry-run — inspect the Cypher that would be executed:
    python scripts/run_stage2_phase_d.py --fixture --backend mock --dry-run --output outputs/stage2/phase_d

# Custom Neo4j connection:
    python scripts/run_stage2_phase_d.py --fixture --backend mock \\
        --neo4j-uri bolt://localhost:7687 \\
        --neo4j-user neo4j \\
        --neo4j-password catalogbank

Neo4j setup (Docker)
--------------------
    docker run --name neo4j \\
        -p 7687:7687 -p 7474:7474 \\
        -e NEO4J_AUTH=neo4j/catalogbank \\
        neo4j:5-community

    pip install neo4j>=5.0.0
    # Then run without --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catalogbank_ocr.stage2.context_builder import build_context_chunks, summarise_chunks
from catalogbank_ocr.stage2.llm_input_builder import build_llm_input
from catalogbank_ocr.stage2.llm_extractor import LLMExtractor, ExtractionResult
from catalogbank_ocr.stage2.validation import validate_batch, ValidationReport, Severity
from catalogbank_ocr.stage2.normalization import normalize_batch, NormalizedResult
from catalogbank_ocr.stage2.graph_ingest import GraphIngestor, IngestionResult

sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from run_stage2_phase_a import SYNTHETIC_CANONICAL


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_ingestion(result: IngestionResult, verbose: bool) -> None:
    mode = "[DRY-RUN]" if result.dry_run else "[LIVE]"
    status = "✗" if result.error else "✓"
    print(f"\n[Phase D — Graph Ingestion] {mode}")
    print(f"  {status} doc={result.doc_stem!r}")

    if result.error:
        print(f"  Error: {result.error}")
        return

    print(f"  Statements : {result.statements_executed}")
    print(f"  Nodes ~    : {result.nodes_merged}")
    print(f"  Relations ~: {result.rels_merged}")

    if result.dry_run and verbose and result.statements:
        print(f"\n  Cypher preview (first 8 of {len(result.statements)} statements):")
        for i, stmt in enumerate(result.statements[:8]):
            # Show just the first non-empty line of the Cypher + params
            first_line = next(
                (l.strip() for l in stmt["cypher"].splitlines() if l.strip()), ""
            )
            params_preview = {
                k: (v[:40] + "…" if isinstance(v, str) and len(v) > 40 else v)
                for k, v in stmt["params"].items()
            }
            print(f"  [{i+1:02d}] {first_line}")
            print(f"       params: {params_preview}")
        if len(result.statements) > 8:
            print(f"       … and {len(result.statements) - 8} more")


def _print_phase_c_summary(
    reports: Dict[str, ValidationReport],
    normalized: NormalizedResult,
) -> None:
    errors   = sum(len(r.errors)   for r in reports.values())
    warnings = sum(len(r.warnings) for r in reports.values())
    info     = sum(
        len([i for i in r.issues if i.severity == Severity.INFO])
        for r in reports.values()
    )
    print(f"\n[Phase C] validation: {errors} error(s)  {warnings} warning(s)  {info} info(s)")
    if errors:
        for chunk_id, report in reports.items():
            if report.errors:
                print(f"  ✗ {chunk_id}:")
                report.print_report(indent=6)
    print(
        f"[Phase C] normalized: "
        f"{len(normalized.products)} product(s)  "
        f"{len(normalized.attributes)} attribute(s)  "
        f"{len(normalized.relations)} relation(s)"
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_canonical(
    canonical_doc: dict,
    extractor: LLMExtractor,
    ingestor: GraphIngestor,
    source_path: str = "(unknown)",
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> IngestionResult:
    """Run the full A → B → C → D pipeline for one document."""

    print(f"\n{'='*64}")
    print(f"  Source  : {source_path}")
    print(f"  Backend : {extractor.backend_name}")
    print(f"  Neo4j   : {'dry-run' if ingestor.dry_run else ingestor.uri}")
    print(f"{'='*64}")

    # ---- Phase A ---------------------------------------------------------
    chunks = build_context_chunks(canonical_doc, source_doc_path=source_path, skip_empty=True)
    if verbose:
        print(f"\n[Phase A] {len(chunks)} chunk(s)")
        print(summarise_chunks(chunks))
    else:
        print(f"[Phase A] {len(chunks)} chunk(s)")

    if not chunks:
        print("  ⚠  No extractable chunks — nothing to ingest.")
        doc_stem = Path(source_path).stem
        return IngestionResult(doc_stem=doc_stem, dry_run=ingestor.dry_run,
                               error="No extractable chunks")

    # ---- Phase B ---------------------------------------------------------
    llm_inputs = [build_llm_input(c, include_prompt=True) for c in chunks]
    print(f"\n[Phase B] Extracting ({extractor.backend_name}) …")
    extraction_results: List[ExtractionResult] = extractor.extract_batch(
        llm_inputs, progress=True
    )

    # ---- Phase C ---------------------------------------------------------
    reports   = validate_batch(extraction_results)
    normalized = normalize_batch(extraction_results)
    _print_phase_c_summary(reports, normalized)

    # ---- Phase D ---------------------------------------------------------
    ingestion = ingestor.ingest(normalized)
    _print_ingestion(ingestion, verbose=verbose)

    # ---- Write outputs ---------------------------------------------------
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "normalized.json").write_text(
            json.dumps(normalized.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "ingestion_result.json").write_text(
            json.dumps(ingestion.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        val_data = {cid: r.to_dict() for cid, r in reports.items()}
        (output_dir / "validation_report.json").write_text(
            json.dumps(val_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  ✓ Wrote normalized.json, ingestion_result.json, validation_report.json → {output_dir}")

    return ingestion


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_canonical_jsons(root: Path) -> List[Path]:
    return sorted(root.glob("outputs/canonical/**/*_document.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 Phase D — Neo4j Knowledge Graph ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="Path to a Stage 1 canonical JSON.")
    parser.add_argument("--fixture", action="store_true",
                        help="Use the built-in synthetic fixture.")

    # LLM
    parser.add_argument("--backend", choices=["ollama", "transformers", "mock"], default=None)
    parser.add_argument("--model",        default=None)
    parser.add_argument("--ollama-host",  default=None)

    # Neo4j
    parser.add_argument("--neo4j-uri",      default="bolt://localhost:7687",
                        help="Neo4j bolt URI (default: bolt://localhost:7687)")
    parser.add_argument("--neo4j-user",     default="neo4j")
    parser.add_argument("--neo4j-password", default="catalogbank")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Build Cypher statements but do NOT connect to Neo4j.")

    # Output / display
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Directory to write output files.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress Phase A/B detail.")

    args = parser.parse_args()
    verbose = not args.quiet

    # Build LLM extractor
    print(f"Initialising extractor (backend={args.backend or 'auto'}) …")
    extractor = LLMExtractor(
        backend=args.backend,
        model=args.model,
        ollama_host=args.ollama_host,
    )
    print(f"  → {extractor.backend_name}")

    # Build graph ingestor
    dry_run = args.dry_run
    if not dry_run:
        print(f"Connecting to Neo4j at {args.neo4j_uri} …")
    ingestor = GraphIngestor(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database,
        dry_run=dry_run,
    )
    if not dry_run:
        print("  → connected. Ensuring constraints …")
        ingestor.ensure_constraints()
        print("  → constraints OK")

    try:
        if args.fixture:
            out = args.output / "fixture" if args.output else None
            process_canonical(
                SYNTHETIC_CANONICAL, extractor, ingestor,
                source_path="(synthetic fixture)",
                output_dir=out, verbose=verbose,
            )
            return

        if args.input:
            if not args.input.exists():
                print(f"ERROR: {args.input} not found", file=sys.stderr)
                sys.exit(1)
            with args.input.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            out = args.output / args.input.stem if args.output else None
            process_canonical(
                doc, extractor, ingestor,
                source_path=str(args.input),
                output_dir=out, verbose=verbose,
            )
            return

        # Auto-discover
        canonical_files = _find_canonical_jsons(_REPO_ROOT)
        if not canonical_files:
            print("No canonical JSONs found. Run with --fixture to test:")
            print("  python scripts/run_stage2_phase_d.py --fixture --backend mock --dry-run")
            return

        print(f"Found {len(canonical_files)} canonical JSON file(s). Processing all.")
        for path in canonical_files:
            with path.open("r", encoding="utf-8") as fh:
                doc = json.load(fh)
            out = args.output / path.stem if args.output else None
            process_canonical(
                doc, extractor, ingestor,
                source_path=str(path),
                output_dir=out, verbose=verbose,
            )

    finally:
        ingestor.close()


if __name__ == "__main__":
    main()
