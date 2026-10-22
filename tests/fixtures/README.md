# Captured OCR fixtures

These are real PP-StructureV3 results, reduced to the fields needed by the tests:
`page_index`, `parsing_res_list` and `layout_det_res`.

- `V21_1_Optomechanics_45_res.json`: Thorlabs CatalogBank sample,
  `Thorlabs/OptoMechanics_v21/_pdfs/V21_1_Optomechanics_45.pdf`.
- `mcmaster-125_3378_15_res.json`: McMaster-Carr CatalogBank sample,
  `McMasterCarr/_pdfs/mcmaster-125_3378_15.pdf`.

Captured with pretrained PaddleOCR 3.0.1 / PaddleX 3.0.1 / PaddlePaddle CPU 3.0.0,
at 200 DPI, with the existing default OCR features and preprocessing. The files
contain model outputs, not manually verified ground truth or accuracy labels.
Original PDFs are supplied by the project's CatalogBank dataset checkout.
