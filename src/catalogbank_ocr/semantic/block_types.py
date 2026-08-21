"""Semantic classes and block-typing heuristics."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, Optional


class SemanticType(str, Enum):
    """Canonical semantic classes for document blocks."""

    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    PRODUCT_CARD = "product_card"
    SPECIFICATION = "specification"
    PARAGRAPH = "paragraph"


HEADING_LABELS = {
    "title",
    "heading",
    "header",
    "subtitle",
    "section_title",
    "section",
    "chapter",
}

TABLE_LABELS = {"table", "table_body", "table_caption"}
FIGURE_LABELS = {"figure", "image", "chart", "diagram", "graph", "plot"}
PRODUCT_LABELS = {"product", "product_card", "item", "sku", "catalog_item"}
SPEC_LABELS = {"specification", "spec", "attributes", "attribute", "key_value", "kv"}

MODEL_TOKEN_RE = re.compile(r"\b[A-Z0-9][A-Z0-9\-_/]{2,}\b")
PRICE_RE = re.compile(r"[\$€£]\s?\d")
DIMENSION_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:in|mm|cm|m|ft|lb|oz|kg|g|w|v|amp|amps)\b", re.I)
KEY_VALUE_RE = re.compile(r"^[^:\n]{2,40}:\s*.+$", re.M)


def normalize_label(value: Optional[Any]) -> str:
    """Normalize a raw PP-Structure label into a lowercase identifier."""

    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_text(value: Optional[Any]) -> str:
    """Normalize textual content from a block-like object."""

    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def markdown_heading_level(text: str) -> Optional[int]:
    """Infer Markdown heading level from text when present."""

    stripped = text.lstrip()
    if not stripped.startswith("#"):
        return None
    level = 0
    for char in stripped:
        if char == "#":
            level += 1
        else:
            break
    return level if level > 0 else None


def is_specification_text(text: str) -> bool:
    """Heuristic for compact key-value or specification-style text."""

    if not text:
        return False
    kv_count = len(KEY_VALUE_RE.findall(text))
    if kv_count >= 2:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 3 and sum(1 for line in lines if ":" in line) >= 2:
        return True
    return False


def is_product_card_text(text: str) -> bool:
    """Heuristic for product card-like content."""

    if not text:
        return False
    score = 0
    if PRICE_RE.search(text):
        score += 1
    if DIMENSION_RE.search(text):
        score += 1
    if MODEL_TOKEN_RE.search(text):
        score += 1
    if text.count("\n") >= 2:
        score += 1
    if is_specification_text(text):
        score += 1
    return score >= 3


def infer_semantic_type(source_type: str, text: str, raw: Optional[Dict[str, Any]] = None) -> SemanticType:
    """Infer one of the canonical semantic types from raw PP-Structure evidence."""

    source_type_norm = normalize_label(source_type)
    text_norm = normalize_text(text)
    raw = raw or {}

    if source_type_norm in HEADING_LABELS:
        return SemanticType.HEADING
    if source_type_norm in TABLE_LABELS:
        return SemanticType.TABLE
    if source_type_norm in FIGURE_LABELS:
        return SemanticType.FIGURE
    if source_type_norm in PRODUCT_LABELS:
        return SemanticType.PRODUCT_CARD
    if source_type_norm in SPEC_LABELS:
        return SemanticType.SPECIFICATION

    heading_level = markdown_heading_level(text_norm)
    if heading_level is not None:
        return SemanticType.HEADING

    if isinstance(raw.get("markdown"), dict) and raw["markdown"].get("text"):
        markdown_text = normalize_text(raw["markdown"].get("text"))
        if markdown_heading_level(markdown_text) is not None:
            return SemanticType.HEADING
        if is_specification_text(markdown_text):
            return SemanticType.SPECIFICATION

    if is_product_card_text(text_norm):
        return SemanticType.PRODUCT_CARD
    if is_specification_text(text_norm):
        return SemanticType.SPECIFICATION

    if source_type_norm in {"text", "paragraph", "body", "sentence", "caption"}:
        return SemanticType.PARAGRAPH
    if not text_norm and source_type_norm in {"image", "figure", "chart", "diagram"}:
        return SemanticType.FIGURE
    return SemanticType.PARAGRAPH


def infer_heading_level(source_type: str, text: str, raw: Optional[Dict[str, Any]] = None) -> int:
    """Infer a heading level from text and raw layout evidence."""

    raw = raw or {}
    text_norm = normalize_text(text)
    explicit = raw.get("heading_level") or raw.get("level") or raw.get("headingLevel")
    if isinstance(explicit, int) and explicit > 0:
        return explicit

    md_level = markdown_heading_level(text_norm)
    if md_level is not None:
        return md_level

    label = normalize_label(source_type)
    if label in {"title", "header", "chapter"}:
        return 1
    if label in {"heading", "subtitle", "section", "section_title"}:
        return 2

    if text_norm and len(text_norm.split()) <= 6 and len(text_norm) <= 60:
        return 2
    return 3
