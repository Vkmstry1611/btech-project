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

# Retain existing oneDNN compatibility flags; device selection is configurable.
# Must be set before any paddle import.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")


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

from catalogbank_ocr.stage2.catalog_export import build_catalog_json


# Keep old build_final_json as alias for backward compat
def build_final_json(
    canonical_doc: Dict[str, Any],
    normalized_result: Any,
    backend_used: str,
    review_issues=None,
) -> Dict[str, Any]:
    return build_catalog_json(
        canonical_doc, normalized_result, backend_used,
        pdf_path=canonical_doc.get("source", {}).get("pdf_path", ""),
        review_issues=review_issues,
    )


# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------

def _peak_memory():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel.GetCurrentProcess()
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if fn(process, ctypes.byref(counters), counters.cb): return counters.PeakWorkingSetSize
    return None


def process_pdf(pdf_path, output_root, extractor, preprocess_config, skip_stage1,
                ocr_engine=None, debug_artifacts=False, max_pages=None, streaming=True, page_batch_size=1,
                reuse_extractions=False):
    """Publish a final file only for a successful run; retain previous outputs."""
    final_dir = output_root / 'final' / pdf_path.stem
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f'{pdf_path.stem}_final.json'
    status_path = final_dir / 'status.json'
    if final_path.exists():
        final_path.rename(final_dir / f'{pdf_path.stem}_previous_{time.time_ns()}.json')
    status = {'state': 'running', 'input': str(pdf_path), 'partial_page_limit': max_pages}
    status_path.write_text(json.dumps(status), encoding='utf-8')
    try:
        result = _process_pdf(pdf_path, output_root, extractor, preprocess_config, skip_stage1,
                              ocr_engine, debug_artifacts, max_pages, streaming, page_batch_size,
                              reuse_extractions)
    except Exception as exc:
        if final_path.exists():
            final_path.rename(final_dir / f'{pdf_path.stem}_failed_{time.time_ns()}.json')
        status.update(state='failed', error=str(exc))
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        raise
    issues_path = output_root / 'stage2' / pdf_path.stem / 'review_issues.json'
    issues = json.loads(issues_path.read_text(encoding='utf-8')) if issues_path.exists() else []
    review_count = sum(v.get('severity') == 'warning' for v in issues)
    status.update(state='complete_with_review' if review_count else 'complete',
                  final_path=str(result), review_warning_count=review_count)
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    return result


def _process_pdf(
    pdf_path: Path,
    output_root: Path,
    extractor: LLMExtractor,
    preprocess_config: ImagePreprocessingConfig,
    skip_stage1: bool,
    ocr_engine: Any = None,
    debug_artifacts: bool = False,
    max_pages: int | None = None,
    streaming: bool = True,
    page_batch_size: int = 1,
    reuse_extractions: bool = False,
) -> Path:
    """Run the full pipeline for one PDF and write the final JSON.

    Returns the path to the written final JSON file.
    """
    start_time = time.monotonic()
    timings = {}
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
        expected_hash = canonical_doc.get('source', {}).get('sha256')
        if expected_hash:
            import hashlib
            digest = hashlib.sha256()
            with pdf_path.open('rb') as source:
                for data in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(data)
            if digest.hexdigest() != expected_hash:
                raise ValueError('Saved OCR belongs to different PDF content; rerun without --skip-stage1')
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
            debug_artifacts=debug_artifacts, max_pages=max_pages, streaming=streaming, page_batch_size=page_batch_size,
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

    timings["stage1_s"] = time.monotonic() - start_time
    phase2_start = time.monotonic()
    evidence_issues = []
    if max_pages is not None and skip_stage1:
        raise ValueError("--max-pages cannot be combined with --skip-stage1; use a separate benchmark output directory")
    # Phase A — context chunks
    chunks = build_context_chunks(
        canonical_doc,
        source_doc_path=str(canonical_json_path),
        skip_empty=True,
    )
    logger.info("[%s] Phase A: %d chunk(s)", stem, len(chunks))
    canonical_json_path.write_text(json.dumps(canonical_doc, indent=2, ensure_ascii=False), encoding='utf-8')

    if not chunks:
        logger.warning("[%s] No extractable chunks — writing empty extraction.", stem)
        from catalogbank_ocr.stage2.normalization import NormalizedResult
        normalized = NormalizedResult(doc_stem=stem)
    else:
        # Phase B — LLM extraction
        from catalogbank_ocr.stage2.llm_input_builder import build_complete_inputs
        llm_inputs = build_complete_inputs(chunks)
        evidence_path = output_root / "stage2" / stem / "evidence.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        if reuse_extractions:
            from catalogbank_ocr.stage2.llm_extractor import ExtractionResult
            saved = evidence_path.parent / 'extractions.json'
            if not saved.exists(): raise ValueError('No saved extractions to reuse: ' + str(saved))
            extraction_results = [ExtractionResult.from_dict(item) for item in
                                  json.loads(saved.read_text(encoding='utf-8'))]
            extraction_results = extraction_results[:len(llm_inputs)]
            if [r.chunk_id for r in extraction_results] != [r['chunk_id'] for r in llm_inputs]:
                raise ValueError('Saved extractions do not match current evidence chunks')
            heading_notes, retry_history = [], []
        else:
            evidence_path.write_text(json.dumps(llm_inputs, indent=2, ensure_ascii=False), encoding="utf-8")
            from catalogbank_ocr.stage2.table_products import extract_with_tables, reconcile_table_products
            extraction_results = extract_with_tables(extractor, llm_inputs, progress=True)
            extraction_results = reconcile_table_products(extraction_results, llm_inputs)
            from catalogbank_ocr.stage2.evidence_validation import repair_extractions, remove_heading_only_products
            heading_notes = remove_heading_only_products(extraction_results, llm_inputs)
            extraction_results, retry_history = repair_extractions(extractor, extraction_results, llm_inputs)
            extraction_results = reconcile_table_products(extraction_results, llm_inputs)
            heading_notes.extend(remove_heading_only_products(extraction_results, llm_inputs))
        from catalogbank_ocr.stage2.native_text import supplement_missing_skus, recover_table_fragment_errors, recover_existing_alpha_prices
        native_added, native_skus = supplement_missing_skus(pdf_path, extraction_results, llm_inputs)
        if native_added:
            logger.info('[%s] Recovered %d exact SKU/price pairs from PDF text', stem, len(native_added))
        heading_notes.extend(recover_existing_alpha_prices(pdf_path, extraction_results, llm_inputs))
        from catalogbank_ocr.stage2.evidence_validation import remove_heading_only_products
        heading_notes.extend(remove_heading_only_products(extraction_results, llm_inputs))
        heading_notes.extend(recover_table_fragment_errors(extraction_results, llm_inputs, native_skus))
        (evidence_path.parent / 'retry_history.json').write_text(json.dumps(retry_history, indent=2, ensure_ascii=False), encoding='utf-8')
        elapsed = time.monotonic() - t0
        logger.info("[%s] Phase B done in %.1fs", stem, elapsed)

        # Phase C — validation + normalization
        from catalogbank_ocr.stage2.evidence_validation import validate_evidence
        evidence_issues = heading_notes + validate_evidence(extraction_results, llm_inputs)
        issues_path = output_root / 'stage2' / stem / 'review_issues.json'
        issues_path.write_text(json.dumps(evidence_issues, indent=2), encoding='utf-8')
        review_path = output_root / 'stage2' / stem / 'extractions.json'
        review_path.write_text(json.dumps([r.to_dict() for r in extraction_results], indent=2, ensure_ascii=False), encoding='utf-8')
        parse_errors = [r.parse_error for r in extraction_results if r.parse_error]
        if any(v['severity'] == 'error' for v in evidence_issues):
            raise ValueError('Unsupported or ambiguous product evidence; see review_issues.json')
        if parse_errors:
            raise ValueError('Extraction failed; review extractions.json: ' + '; '.join(parse_errors))
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

        report_path = output_root / 'stage2' / stem / 'validation.json'
        report_path.write_text(json.dumps({k: v.to_dict() for k, v in reports.items()}, indent=2), encoding='utf-8')
        if errors:
            raise ValueError('Validation errors; final export blocked. See validation.json')
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
    if debug_artifacts:
        kg_viz_path = output_root / "visualizations" / stem / f"{stem}_knowledge_graph.png"
        try:
            visualize_knowledge_graph(
                normalized_result=normalized,
                output_path=kg_viz_path,
                doc_stem=stem,
            )
            logger.info("[%s] ✓ KG visualization → %s", stem, kg_viz_path)
        except Exception:
            logger.exception('Optional graph visualization failed')


    # ------------------------------------------------------------------ #
    # Merge → final JSON
    # ------------------------------------------------------------------ #
    timings["stage2_s"] = time.monotonic() - phase2_start
    final_doc = build_final_json(canonical_doc, normalized, extractor.backend_name, evidence_issues)
    issues_path = output_root / 'stage2' / stem / 'review_issues.json'
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path.write_text(json.dumps(evidence_issues, indent=2, ensure_ascii=False), encoding='utf-8')
    final_json_path.write_text(
        json.dumps(final_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[%s] ✓ Final JSON written → %s", stem, final_json_path)
    timings['total_s'] = time.monotonic() - start_time
    timings['pages'] = len(canonical_doc.get('pages', []))
    timings['backend'] = extractor.backend_name
    timings['reused_ocr'] = skip_stage1
    timings['partial_page_limit'] = max_pages or canonical_doc.get('source', {}).get('page_limit')
    timings['peak_process_memory_bytes'] = _peak_memory()
    metric_path = output_root / 'metrics' / stem / 'pipeline.json'
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(timings, indent=2), encoding='utf-8')
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
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
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
    parser.add_argument('--reuse-extractions', action='store_true',
                        help='Revalidate saved Stage 2 extractions without rerunning the LLM; requires --skip-stage1.')

    # Stage 2 / LLM options
    parser.add_argument("--backend", choices=["ollama", "transformers", "mock"], default="ollama",
                        help="LLM backend (default: Ollama; mock must be explicitly selected).")
    parser.add_argument("--model", default=None,
                        help="LLM model name (e.g. qwen3:8b for Ollama).")
    parser.add_argument("--ollama-host", default=None,
                        help="Ollama host URL if not localhost.")

    parser.add_argument('--interactive', action='store_true', help='Accept additional PDF paths without unloading models.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu:0'], default='auto')
    parser.add_argument('--recognition-batch-size', type=int, default=None,
                        help='Override recognition batching; default is 1 on GPU or the model default on CPU.')
    parser.add_argument('--page-batch-size', type=int, default=1, help='Pages submitted together to OCR (1 to 16).')
    parser.add_argument('--debug-artifacts', action='store_true')
    parser.add_argument('--max-pages', type=int)
    parser.add_argument('--sequential-preprocessing', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.page_batch_size <= 16:
        parser.error('--page-batch-size must be between 1 and 16')
    if args.max_pages is not None and args.max_pages < 1:
        parser.error('--max-pages must be positive')
    if args.skip_stage1 and args.max_pages is not None:
        parser.error('--skip-stage1 and --max-pages cannot be combined')
    if args.reuse_extractions and not args.skip_stage1:
        parser.error('--reuse-extractions requires --skip-stage1')
    if args.recognition_batch_size is not None and args.recognition_batch_size < 1:
        parser.error('--recognition-batch-size must be positive')

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

    startup_started = time.monotonic()
    # Initialize OCR engine once (expensive)
    ocr_engine = None
    if not args.skip_stage1:
        from catalogbank_ocr.ocr.ppstructure import initialize_ppstructurev3
        print("Initialising OCR engine (PPStructureV3)…")
        ocr_engine = initialize_ppstructurev3(device=args.device, recognition_batch_size=args.recognition_batch_size)
        print("  → ready\n")

    # Third-party model initialization may change the root logging level.
    logging.getLogger('catalogbank_ocr').setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    startup_seconds = time.monotonic() - startup_started
    (output_root / "startup_metrics.json").write_text(json.dumps({"ocr_startup_s": startup_seconds, "device_requested": args.device, "recognition_batch_size": args.recognition_batch_size}, indent=2), encoding="utf-8")
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
                debug_artifacts=args.debug_artifacts, max_pages=args.max_pages,
                streaming=not args.sequential_preprocessing, page_batch_size=args.page_batch_size,
                reuse_extractions=args.reuse_extractions,
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

    if args.interactive:
        while True:
            try:
                next_path = input('Next PDF/folder path (blank to quit): ').strip().strip('"')
            except EOFError:
                break
            if not next_path: break
            try:
                if ocr_engine is None:
                    from catalogbank_ocr.ocr.ppstructure import initialize_ppstructurev3
                    ocr_engine = initialize_ppstructurev3(device=args.device,
                                                         recognition_batch_size=args.recognition_batch_size)
                for pdf in _collect_pdfs(Path(next_path)):
                    print(process_pdf(pdf, output_root, extractor, preprocess_config, False,
                                      ocr_engine, args.debug_artifacts, args.max_pages,
                                      not args.sequential_preprocessing, args.page_batch_size))
            except Exception:
                logger.exception('Processing failed; models remain available for the next file')
    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
