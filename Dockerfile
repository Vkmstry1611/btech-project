# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# CatalogBank OCR — deployment image (slim, no OCR / no LLM).
#
# Reproduces the final catalogs from saved research artifacts inside the image
# (scripts/run_mock_pipeline.py — no PaddleOCR, no LLM) and serves them through
# a FastAPI API. Build:
#
#   docker build -t catalogbank-ocr-api .
#   docker run -p 8000:8000 catalogbank-ocr-api
#   # open http://localhost:8000/docs
# -----------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- 1. Install only the slim runtime dependencies ---------------------------
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# --- 2. Install the package without its heavy core deps ---------------------
# The full pyproject pulls paddlepaddle/torch/paddleocr; the deployment image
# only needs the pure-Python stage2 + api modules, so install with --no-deps.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

# --- 3. Copy the pipeline scripts + saved research artifacts (inputs) -------
# Only the inputs the mock pipeline needs are copied (keeps the image small).
COPY scripts/run_mock_pipeline.py ./scripts/run_mock_pipeline.py
COPY scripts/evaluate_catalog.py  ./scripts/evaluate_catalog.py
COPY outputs/canonical ./outputs/canonical
COPY outputs/stage2    ./outputs/stage2
COPY outputs/final     ./outputs/final

# --- 4. Reproduce the deployment artifacts inside the image -----------------
# Deterministic, no network, no GPU. Produces outputs/deploy/{final,stage2,
# metrics,run_summary.json} + outputs/deploy/evaluation.json.
RUN python scripts/run_mock_pipeline.py \
        --input outputs --output outputs/deploy --backend-label mock-reuse \
 && python scripts/evaluate_catalog.py \
        --final outputs/deploy/final --out outputs/deploy/evaluation.json

EXPOSE 8000
ENV HOST=0.0.0.0 PORT=8000 CATALOG_DIR=/app/outputs/deploy/final

# Liveness probe against the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health').read(); sys.exit(0)" || exit 1

CMD ["uvicorn", "catalogbank_ocr.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
