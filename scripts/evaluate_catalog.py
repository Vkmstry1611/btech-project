"""Evaluate reproduced catalogs: aggregate stats over the deployment outputs.

Reads every ``*_final.json`` under the deployment ``final/`` directory and
writes a single ``evaluation.json`` with per-document and aggregate metrics.

Usage
-----
    python scripts/evaluate_catalog.py --final outputs/deploy/final --out outputs/deploy/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _walk(node: Dict[str, Any], acc: Dict[str, Any]) -> None:
    """Accumulate counts and price samples while walking the catalog tree."""
    t = node.get("type")
    if t == "Product Category":
        acc["categories"] += 1
    elif t == "Model":
        acc["models"] += 1
    elif t == "Sellable Item":
        acc["items"] += 1
        if node.get("articleNumber"):
            acc["article_numbers"] += 1
    elif t == "List Price":
        acc["prices"] += 1
        amount = node.get("amount")
        if isinstance(amount, (int, float)):
            acc["price_min"] = min(acc["price_min"], amount) if acc["price_min"] is not None else amount
            acc["price_max"] = max(acc["price_max"], amount) if acc["price_max"] is not None else amount
            currency = node.get("currency")
            if currency:
                acc["currencies"].add(currency)
    for key in ("subcategories", "models", "sellableItems", "listPrices"):
        for child in node.get(key, []) or []:
            _walk(child, acc)


def _evaluate_document(path: Path) -> Dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    acc: Dict[str, Any] = {
        "stem": path.stem.replace("_final", ""),
        "title": doc.get("title"),
        "identifier": doc.get("identifier"),
        "categories": 0,
        "models": 0,
        "items": 0,
        "article_numbers": 0,
        "prices": 0,
        "price_min": None,
        "price_max": None,
        "currencies": set(),
    }
    _walk(doc, acc)
    acc["currencies"] = sorted(acc["currencies"])
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation stats over reproduced catalogs.")
    parser.add_argument("--final", type=Path, default=Path("outputs/deploy/final"),
                        help="Directory of *_final.json catalogs (default: outputs/deploy/final).")
    parser.add_argument("--out", type=Path, default=Path("outputs/deploy/evaluation.json"),
                        help="Output evaluation JSON path.")
    args = parser.parse_args()

    if not args.final.exists():
        print(f"No deployment catalogs found at {args.final}", file=sys.stderr)
        sys.exit(1)

    docs: List[Dict[str, Any]] = []
    for path in sorted(args.final.glob("**/*_final.json")):
        docs.append(_evaluate_document(path))

    aggregate = {
        "documents": len(docs),
        "total_categories": sum(d["categories"] for d in docs),
        "total_models": sum(d["models"] for d in docs),
        "total_items": sum(d["items"] for d in docs),
        "total_prices": sum(d["prices"] for d in docs),
        "currencies": sorted({c for d in docs for c in d["currencies"]}),
    }

    evaluation = {"documents": docs, "aggregate": aggregate}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Evaluation written -> {args.out}")
    print(f"  documents={aggregate['documents']}  "
          f"categories={aggregate['total_categories']}  "
          f"models={aggregate['total_models']}  "
          f"items={aggregate['total_items']}  "
          f"prices={aggregate['total_prices']}  "
          f"currencies={aggregate['currencies']}")


if __name__ == "__main__":
    main()
