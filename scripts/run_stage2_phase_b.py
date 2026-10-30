"""Stage 2 Phase B — LLM entity and relation extraction.

Chains Phase A (context builder + LLM input builder) directly into Phase B
(LLM extraction) and prints / saves structured results.

Usage
-----
# Dry-run with the synthetic fixture (no LLM needed):
    python scripts/run_stage2_phase_b.py --fixture --backend mock

# Real extraction against a canonical JSON using Ollama:
    python scripts/run_stage2_phase_b.py --input outputs/canonical/my_doc/my_doc_document.json

# Explicit backend / model:
    python scripts/run_stage2_phase_b.py --fixture --backend ollama --model qwen3:8b
    python scripts/run_stage2_phase_b.py --fixture --backend transformers --model Qwen/Qwen3-8B-Instruct

# Write extraction results to disk:
    python scripts/run_stage2_phase_b.py --fixture --backend mock --output outputs/stage2/phase_b

# Suppress per-chunk content previews:
    python scripts/run_stage2_phase_b.py --fixture --backend mock --quiet

Setup (Ollama — recommended)
-----
    # 1. Install Ollama from https://ollama.com and start the service
    # 2. Pull the model
    ollama pull qwen3:8b
    # 3. Install Python client
    pip install ollama
    # 4. Run
    python scripts/run_stage2_phase_b.py --input <canonical_json>

Setup (transformers — GPU fallback)
-----
    pip install transformers>=4.40 accelerate bitsandbytes
    python scripts/run_stage2_phase_b.py --input <canonical_json> --backend transformers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from catalogbank_ocr.stage2.context_builder import build_context_chunks, summarise_chunks
from catalogbank_ocr.stage2.llm_input_builder import build_llm_input, estimate_prompt_chars
from catalogbank_ocr.stage2.llm_extractor import LLMExtractor, ExtractionResult

# Re-use the synthetic fixture defined in Phase A script
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from run_stage2_phase_a import SYNTHETIC_CANONICAL


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _print_result(result: ExtractionResult, verbose: bool = True) -> None:
    """Print one ExtractionResult in a human-readable format."""
    status = "✓" if not result.parse_error else "✗"
    section = " › ".join(result.section_path) if result.section_path else "(root)"
    print(f"\n  {status} [{result.chunk_id}]  section: {section}")
    print(f"    backend: {result.backend_used}  latency: {result.latency_s:.2f}s")

    if result.parse_error:
        print(f"    ⚠  parse error: {result.parse_error}")
        if verbose and result.raw_response:
            preview = result.raw_response[:300].replace("\n", " ")
            print(f"    raw response (truncated): {preview!r}")
        return

    if result.is_empty:
        print("    (no entities extracted)")
        return

    if result.products:
        print(f"    products ({len(result.products)}):")
        for p in result.products:
            parts = [f"name={p.name!r}"]
            if p.model:
                parts.append(f"model={p.model!r}")
            if p.sku:
                parts.append(f"sku={p.sku!r}")
            if p.category:
                parts.append(f"category={p.category!r}")
            print(f"      • {', '.join(parts)}")

    if result.attributes:
        print(f"    attributes ({len(result.attributes)}):")
        for a in result.attributes:
            unit_str = f" [{a.unit}]" if a.unit else ""
            print(f"      • {a.entity!r}  {a.key}: {a.value}{unit_str}")

    if result.relations:
        print(f"    relations ({len(result.relations)}):")
        for r in result.relations:
            print(f"      • {r.subject!r}  --[{r.predicate}]-->  {r.object!r}")


def _print_summary(results: list[ExtractionResult]) -> None:
    """Print aggregate stats across all extraction results."""
    total_products = sum(len(r.products) for r in results)
    total_attrs = sum(len(r.attributes) for r in results)
    total_relations = sum(len(r.relations) for r in results)
    failed = [r for r in results if r.parse_error]
    total_latency = sum(r.latency_s for r in results)

    print(f"\n{'='*64}")
    print(f"  Extraction summary")
    print(f"{'='*64}")
    print(f"  Chunks processed : {len(results)}")
    print(f"  Products found   : {total_products}")
    print(f"  Attributes found : {total_attrs}")
    print(f"  Relations found  : {total_relations}")
    print(f"  Parse errors     : {len(failed)}")
    print(f"  Total latency    : {total_latency:.1f}s")
    if failed:
        print(f"\n  Failed chunks:")
        for r in failed:
            print(f"    • {r.chunk_id}: {r.parse_error}")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_canonical(
    canonical_doc: dict,
    extractor: LLMExtractor,
    source_path: str = "(unknown)",
    output_dir: Path | None = None,
    verbose: bool = True,
) -> list[ExtractionResult]:
    """Run Phase A + Phase B on one canonical document."""

    print(f"\n{'='*64}")
    print(f"  Source : {source_path}")
    print(f"  Backend: {extractor.backend_name}")
    print(f"{'='*64}")

    # --- Phase A ---
    chunks = build_context_chunks(canonical_doc, source_doc_path=source_path, skip_empty=True)
    print(f"\n[Phase A] Context Builder → {len(chunks)} chunk(s)")
    print(summarise_chunks(chunks))

    if not chunks:
        print("  ⚠  No extractable chunks — nothing to send to LLM.")
        return []

    # Build LLM inputs
    llm_inputs = [build_llm_input(chunk, include_prompt=True) for chunk in chunks]
    total_chars = sum(estimate_prompt_chars(inp) for inp in llm_inputs)
    print(f"  Total prompt size: ~{total_chars // 4} tokens across {len(llm_inputs)} prompt(s)")

    # --- Phase B ---
    print(f"\n[Phase B] LLM Extraction ({extractor.backend_name})")
    results = extractor.extract_batch(llm_inputs, progress=True)

    # Print per-chunk detail
    if verbose:
        for result in results:
            _print_result(result, verbose=verbose)

    _print_summary(results)

    # --- Write outputs ---
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # One JSON per chunk
        for result in results:
            out_path = output_dir / f"{result.chunk_id}_extraction.json"
            out_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Combined results file
        combined = {
            "source": source_path,
            "backend": extractor.backend_name,
            "chunks": len(results),
            "results": [r.to_dict() for r in results],
        }
        combined_path = output_dir / "_combined_extractions.json"
        combined_path.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  ✓ Wrote {len(results)} extraction file(s) + combined JSON to: {output_dir}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_canonical_jsons(root: Path) -> list[Path]:
    return sorted(root.glob("outputs/canonical/**/*_document.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 Phase B — LLM entity/relation extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="Path to a canonical JSON file from Stage 1.")
    parser.add_argument("--fixture", action="store_true",
                        help="Use the built-in synthetic fixture (no Stage 1 outputs needed).")
    parser.add_argument("--backend", choices=["ollama", "transformers", "mock"], default=None,
                        help="LLM backend to use (default: auto-detect).")
    parser.add_argument("--model", default=None,
                        help="Model name/tag override (e.g. qwen3:8b, qwen3:14b).")
    parser.add_argument("--ollama-host", default=None,
                        help="Ollama server URL if not running on localhost:11434.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Directory to write extraction JSON files.")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress per-chunk entity detail.")
    args = parser.parse_args()

    # Build the extractor once — shared across all documents
    print(f"Initialising LLM extractor (backend={args.backend or 'auto'}, model={args.model or 'default'})…")
    extractor = LLMExtractor(
        backend=args.backend,
        model=args.model,
        ollama_host=args.ollama_host,
    )
    print(f"  → using: {extractor.backend_name}")

    verbose = not args.quiet

    if args.fixture:
        out_dir = args.output / "fixture" if args.output else None
        process_canonical(SYNTHETIC_CANONICAL, extractor,
                          source_path="(synthetic fixture)", output_dir=out_dir, verbose=verbose)
        return

    if args.input:
        if not args.input.exists():
            print(f"ERROR: file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        with args.input.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        out_dir = args.output / args.input.stem if args.output else None
        process_canonical(doc, extractor, source_path=str(args.input),
                          output_dir=out_dir, verbose=verbose)
        return

    # Auto-discover
    repo_root = Path(__file__).resolve().parents[1]
    canonical_files = _find_canonical_jsons(repo_root)

    if not canonical_files:
        print("No canonical JSON files found under outputs/canonical/.")
        print("Run with --fixture to test without Stage 1 outputs:")
        print("  python scripts/run_stage2_phase_b.py --fixture --backend mock")
        return

    print(f"Found {len(canonical_files)} canonical JSON file(s). Processing all.\n")
    all_results: list[ExtractionResult] = []
    for path in canonical_files:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
        out_dir = args.output / path.stem if args.output else None
        results = process_canonical(doc, extractor, source_path=str(path),
                                    output_dir=out_dir, verbose=verbose)
        all_results.extend(results)

    if len(canonical_files) > 1:
        print(f"\n{'='*64}")
        print(f"  Grand total across {len(canonical_files)} document(s)")
        _print_summary(all_results)


if __name__ == "__main__":
    main()
