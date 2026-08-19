"""Inspect PP-StructureV3 JSON outputs without assuming a fixed schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass()
class JsonInspection:
    """A lightweight description of a JSON document."""

    path: Path
    root_type: str
    top_level_keys: list[str]
    preview: str


def load_json(path: Path) -> Any:
    """Load a JSON document from disk."""

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inspect_json_document(path: Path, preview_chars: int = 2000) -> JsonInspection:
    """Inspect a JSON document safely and return summary information."""

    data = load_json(path)
    root_type = type(data).__name__
    top_level_keys = list(data.keys()) if isinstance(data, dict) else []
    preview = json.dumps(data, indent=2, ensure_ascii=False)
    return JsonInspection(
        path=path,
        root_type=root_type,
        top_level_keys=top_level_keys,
        preview=preview[:preview_chars],
    )


def find_json_files(output_dir: Path) -> list[Path]:
    """Find JSON files under an output directory."""

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    return sorted(output_dir.rglob("*.json"))
