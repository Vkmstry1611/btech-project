# HierarchIQ TODO

Updated: 22 October 2026

Agreed scope: improve multi-page catalog output, match `sampleoutput.json`, distinguish headings/models/SKUs, handle non-product pages, and reduce processing time for a newly uploaded PDF without reducing extraction quality. Core implementation is in place. Checked items have regression coverage; remaining performance and catalog-level acceptance work is listed explicitly below.

## 1. Establish correctness and timing baselines

- [ ] Select representative catalogs with product pages, covers, indexes, section dividers, small part numbers and cross-page tables.
- [ ] Annotate expected categories, models, SKUs, descriptions, prices and parent-child relationships for comparison.
- [ ] Record startup, rendering, preprocessing, OCR, LLM extraction and export times separately, together with hardware, model versions, page count and peak memory.
- [ ] Measure a representative page subset to estimate a 60-page run, then validate with a full 60-page catalog. Report cold-start and warm-service times separately; include LLM time and table-heavy pages in the estimate.

## 2. Repair extraction and final output

- [x] Retain descriptions through extraction, typed records, normalization and final export.
- [x] Fix orphan-attribute validation, including when the extracted product set is empty.
- [x] Prevent duplicate model placement; use explicit category membership rather than substring matching.
- [x] Replace process-dependent identifiers with stable identifiers and preserve conflicting attribute values with their sources.
- [x] Map a clean intermediate product representation deterministically to `sampleoutput.json`: Product Category → Model → Sellable Item → List Price.
- [x] Preserve field names, types and nesting required by the sample; validate identity, references and required fields before final export.
- [x] Block invalid final output on fatal validation errors and report ambiguous records for review.

## 3. Separate headings, models and SKUs

- [x] Classify headings using layout, numbering and surrounding content; require product evidence before creating a product from a heading.
- [x] Distinguish a product model/design from a sellable SKU/article number; allow multiple sellable variants under one model.
- [x] Preserve exact codes from tables and nearby text. Do not invent missing identifiers or interpret generic titles as product codes.
- [x] Keep extraction-chunk and page/block evidence provenance; test ambiguous titles, repeated codes and model/variant relationships. Field-specific OCR bounding boxes are not claimed.

## 4. Handle page roles and multi-page context

- [x] Classify pages as cover/title, index/table of contents, section divider, product, mixed or uncertain.
- [x] Store cover metadata and index section/page references without creating products from those entries.
- [x] Carry section-divider context into following product pages until the section changes.
- [x] Use block-level classification on mixed pages; retain uncertain content for further checking.
- [x] Preserve every page and its classification in intermediate artifacts; skip product extraction only for confidently non-product content.
- [ ] Carry active headings and unfinished tables across page boundaries, with page-aware geometry checks.
- [x] Combine evidence for the same product across pages before constructing the final business hierarchy; keep model/SKU distinctions intact.
- [x] Avoid losing long-section evidence through prompt truncation; use bounded chunks with explicit continuation and reconciliation.

## 5. Speed up new-PDF processing without reducing quality

- [x] Inspect available GPU/VRAM and compatible PaddlePaddle setup; replace unconditional CPU forcing with configurable device selection and a CPU fallback.
- [x] Keep the OCR engine and LLM service initialized across uploads to avoid repeated model loading.
- [x] Evaluate supported inference batching while retaining page order, provenance and the same recognition models/settings. GPU recognition defaults to batch 1 on this 8 GB laptop; larger batches remain optional.
- [ ] Overlap rendering/preprocessing with OCR using bounded queues; begin extraction only when a section has sufficient context, and reconcile later continuations.
- [x] Reduce redundant image reads and intermediate writes; make expensive debug visualizations optional while retaining required evidence.
- [ ] Compare cold runs on new PDFs before and after optimization. Retain current resolution and recognition quality; check exact SKU accuracy, table values, hierarchy and output completeness.

The primary speed requirement concerns previously unseen PDFs. Cache reuse and `--skip-stage1` are conveniences for repeat runs, not the solution to this requirement. Do not adopt lower DPI, weaker models or disabled recognition features as speed shortcuts without an explicit scope change and quality evaluation.

## 6. Regression and completion checks

- [x] Restore missing OCR fixtures and add regression cases for description retention, orphan attributes and duplicate model placement.
- [x] Test covers, indexes, section dividers, mixed pages, cross-page tables and repeated products across pages.
- [ ] Validate generated output against the sample structure and annotated expected results.
- [ ] Report measured 60-page processing time, stage breakdown, memory use and before/after quality results.
- [x] Update run instructions for device selection, persistent execution, diagnostics and final output paths.

Completion means validated output improvements and measured new-PDF speed gains without an observed quality regression on the agreed evaluation set. No runtime guarantee is established yet.

## Verification status

- Regression suite: 83 passing tests, no skips (including regenerated real OCR fixtures from both vendors).
- Exact metric/imperial table SKUs are reconciled independently of LLM omissions. Clear literal rows bypass LLM decoding; ambiguous price cells remain verbatim in review artifacts. Mixed text and uncertain structure use the semantic extraction path.
- Rendering/preprocessing is overlapped with OCR using one preparation worker. Pixel parity against the original renderer is tested at the same DPI.
- Separate `.venv-gpu` installation is complete and CUDA inference is verified on the RTX 5060. The system Python installation remains unchanged.
- Four-page saved-OCR acceptance exported 92 products and 466 attributes in 471.9 seconds, with ambiguous price cells flagged for review. This validates extraction and export, not new-PDF OCR timing or annotated catalog-wide accuracy.
- On one matched page at 200 DPI and recognition batch 1, CPU OCR took 81.8 seconds and GPU OCR took 17.8 seconds. All 25 extracted block types/texts matched exactly. This is a page-level observation, not a 60-page runtime guarantee.
- A fresh 60-page Newark catalog completed rendering/preprocessing/OCR in 1,417.7 seconds (23 min 38 sec), excluding model startup. Its semantic pass was interrupted to prioritize the requested two-domain evaluation; the completed OCR remains available. No uninterrupted end-to-end 60-page timing is claimed.
- Two new five-page subsets were prepared from optical hardware and industrial fittings PDFs and copied into `pdfs/`. Full-resolution GPU OCR completed in 70.1 and 92.6 seconds respectively with a warm engine. Optical semantic extraction completed in 111.7 seconds with 42 exact source-supported SKUs and prices; two displaced SKU prices were recovered from a printed USD row in the PDF text layer. The dense industrial semantic pass took 493.2 seconds, followed by source-text reconciliation: all 750 exact code/price pairs are present, 278 of them recovered from the text layer, with three malformed OCR fragments flagged for review. Currency remains unknown on 708 prices lacking an explicit local symbol. These are component timings from separate runs, not one uninterrupted end-to-end measurement.
- Initial diagnostic on one new Thorlabs page: OCR 81.8 s; full pipeline 110.7 s, excluding model startup. This predates final refinements and is not a final speed claim or a measured 60-page result.
- Starting LLM extraction before the entire document is reconstructed remains deferred: cross-page correctness takes precedence. The implemented overlap is page rendering/preprocessing with OCR.
- Broad vendor-separated ground truth and a complete 60-page end-to-end benchmark remain required before claiming catalog-wide accuracy or a runtime guarantee.
