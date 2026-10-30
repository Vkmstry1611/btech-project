"""FastAPI application that serves the extracted catalog data.

The service reads the JSON artifacts produced by the DVC deployment pipeline
(``dvc repro``) — it does **not** load any OCR or LLM models at runtime, so it
runs in a slim container with only FastAPI + Uvicorn.

Catalog root resolution order:
    1. ``CATALOG_DIR`` environment variable
    2. ``outputs/deploy/final``  (DVC-managed deployment output)
    3. ``outputs/final``          (saved research artifacts, fallback)

Run locally:
    uvicorn catalogbank_ocr.api.app:app --reload

Endpoints:
    GET /                                 service info
    GET /health                           liveness + document count
    GET /catalogs                         list of available catalogs + counts
    GET /catalogs/{stem}                  full catalog tree for one document
    GET /catalogs/{stem}/search?q=...     sellable items matching the query
    GET /metrics                          pipeline run summary + evaluation
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    # src/catalogbank_ocr/api/app.py -> repo root is 3 parents up.
    return Path(__file__).resolve().parents[3]


def _resolve_catalog_root() -> Path:
    env = os.environ.get("CATALOG_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(_repo_root() / "outputs" / "deploy" / "final")
    candidates.append(_repo_root() / "outputs" / "final")
    for c in candidates:
        if c.is_dir() and any(c.glob("**/*_final.json")):
            return c
    # default to the first candidate even if empty (endpoints will 404)
    return candidates[0]


def _resolve_aux(name: str) -> Optional[Path]:
    """Find an auxiliary JSON (run_summary / evaluation) under outputs/deploy."""
    for base in (_repo_root() / "outputs" / "deploy", _repo_root() / "outputs"):
        p = base / name
        if p.exists():
            return p
    return None


def _count_catalog(node: Dict[str, Any]) -> Dict[str, int]:
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


def _iter_sellable_items(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the catalog tree into a list of sellable items (with model + category)."""
    items: List[Dict[str, Any]] = []

    def walk(n: Dict[str, Any], category_path: List[str]) -> None:
        cat_title = n.get("title")
        path = category_path + ([cat_title] if n.get("type") == "Product Category" else [])
        for model in n.get("models", []) or []:
            model_title = model.get("title")
            for item in model.get("sellableItems", []) or []:
                items.append({
                    "articleNumber": item.get("articleNumber"),
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "model": model_title,
                    "category": " > ".join(path) if path else None,
                    "properties": item.get("properties", {}),
                    "listPrices": item.get("listPrices", []),
                    "identifier": item.get("identifier"),
                })
        for sub in n.get("subcategories", []) or []:
            walk(sub, path)

    walk(node, [])
    return items


class CatalogStore:
    """In-memory store of all catalogs + auxiliary metrics, loaded at startup."""

    def __init__(self, catalog_root: Path) -> None:
        self.catalog_root = catalog_root
        self.catalogs: Dict[str, Dict[str, Any]] = {}
        self.summaries: List[Dict[str, Any]] = []
        self.run_summary: Optional[Dict[str, Any]] = None
        self.evaluation: Optional[Dict[str, Any]] = None
        self._load()

    def _load(self) -> None:
        for path in sorted(self.catalog_root.glob("**/*_final.json")):
            stem = path.parent.name  # directory holds the {stem}_final.json
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            self.catalogs[stem] = doc
            counts = _count_catalog(doc)
            self.summaries.append({
                "stem": stem,
                "title": doc.get("title"),
                "identifier": doc.get("identifier"),
                **counts,
            })

        summary_path = _resolve_aux("run_summary.json")
        if summary_path:
            try:
                self.run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                self.run_summary = None
        eval_path = _resolve_aux("evaluation.json")
        if eval_path:
            try:
                self.evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            except Exception:
                self.evaluation = None

    def get(self, stem: str) -> Dict[str, Any]:
        if stem not in self.catalogs:
            raise HTTPException(status_code=404, detail=f"Catalog '{stem}' not found")
        return self.catalogs[stem]

    def search(self, stem: str, query: str) -> List[Dict[str, Any]]:
        catalog = self.get(stem)
        items = _iter_sellable_items(catalog)
        q = query.strip().lower()
        if not q:
            return items
        hits = []
        for it in items:
            haystack = " ".join(
                str(v) for v in (
                    it.get("articleNumber"), it.get("title"),
                    it.get("description"), it.get("model"), it.get("category"),
                ) if v
            ).lower()
            if q in haystack:
                hits.append(it)
        return hits


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
store = CatalogStore(_resolve_catalog_root())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # store is loaded at import time; expose count for readiness checks
    app.state.document_count = len(store.catalogs)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="CatalogBank OCR API",
        description="Serves extracted product catalogs from saved pipeline artifacts.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", summary="Service info")
    def root() -> Dict[str, Any]:
        return {
            "service": "CatalogBank OCR API",
            "version": "0.1.0",
            "catalog_root": str(store.catalog_root),
            "documents": len(store.catalogs),
            "stems": [s["stem"] for s in store.summaries],
            "endpoints": ["/health", "/catalogs", "/catalogs/{stem}",
                          "/catalogs/{stem}/search?q=", "/metrics"],
        }

    @app.get("/health", summary="Liveness check")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "documents": len(store.catalogs)}

    @app.get("/catalogs", summary="List available catalogs")
    def list_catalogs() -> Dict[str, Any]:
        return {"count": len(store.summaries), "catalogs": store.summaries}

    @app.get("/catalogs/{stem}", summary="Full catalog tree for one document")
    def get_catalog(stem: str) -> Dict[str, Any]:
        return store.get(stem)

    @app.get("/catalogs/{stem}/search", summary="Search sellable items")
    def search_catalog(
        stem: str,
        q: str = Query("", description="Case-insensitive substring over article number, title, description, model, category"),
    ) -> Dict[str, Any]:
        hits = store.search(stem, q)
        return {"stem": stem, "query": q, "count": len(hits), "items": hits}

    @app.get("/metrics", summary="Pipeline run summary and evaluation")
    def metrics() -> Dict[str, Any]:
        return {
            "run_summary": store.run_summary,
            "evaluation": store.evaluation,
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "catalogbank_ocr.api.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
