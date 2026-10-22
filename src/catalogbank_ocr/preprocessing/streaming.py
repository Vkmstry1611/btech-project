"""Bounded page preparation overlapping OCR, with one owner of the PDF handle."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from PIL import Image
from catalogbank_ocr.preprocessing.image_preprocessing import apply_optional_preprocessing, ImagePreprocessingResult


def prepared_pages(pdf_path, rendered_dir, processed_dir, config, max_pages=None):
    import fitz
    if max_pages is not None and max_pages < 1: raise ValueError('max_pages must be positive')
    rendered_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    def render(index):
        start = perf_counter()
        # Each call owns its document: no fitz document is shared between threads.
        with fitz.open(pdf_path) as doc:
            pix = doc[index].get_pixmap(matrix=fitz.Matrix(config.dpi / 72, config.dpi / 72))
            path = rendered_dir / f'{Path(pdf_path).stem}_p{index+1:03d}.png'
            pix.save(str(path))
            image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        render_s = perf_counter() - start
        start = perf_counter()
        processed = processed_dir / path.name
        apply_optional_preprocessing(image, config).save(processed)
        return ImagePreprocessingResult(source_pdf=pdf_path, rendered_image=path, processed_image=processed,
            steps=['render_page', 'configured_preprocessing', 'autocontrast'],
            metadata={'dpi': config.dpi, 'render_s': render_s, 'preprocess_s': perf_counter()-start})
    with fitz.open(pdf_path) as doc: count = min(len(doc), max_pages or len(doc))
    if not count: raise ValueError('PDF has no pages')
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(render, 0)
        for index in range(count):
            result = pending.result()
            if index+1 < count: pending = pool.submit(render, index+1)
            yield result
