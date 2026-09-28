"""Stage 2 — Step 5b: Normalization.

Takes ExtractionResult objects (from Phase B) and produces a
NormalizedResult with cleaned, deduplicated, and canonicalized data
ready for Neo4j ingestion.

What this module does
---------------------
1. String cleaning       — strip whitespace, collapse internal whitespace,
                           title-case product names, remove leading articles.
2. Unit canonicalization — parse numeric values + unit strings into a
                           standard form: {value: float, unit: str (SI/common)}.
                           e.g. "2 in" → {value: 2.0, unit: "in"}
                                "25.4mm" → {value: 25.4, unit: "mm"}
                                "3,000 psi" → {value: 3000.0, unit: "psi"}
3. SKU normalization     — uppercase, strip spaces/hyphens noise.
4. Cross-chunk dedup     — identical (name, sku) products merged; duplicate
                           attributes for the same (entity, key) kept as the
                           first seen value.
5. Confidence flagging   — blocks below a threshold get a low_confidence flag
                           in metadata (for downstream filtering).

Design: pure Python + stdlib only (no pint, no spacy).  The unit table is a
hand-rolled lookup — sufficient for industrial catalogs (length, weight,
pressure, temperature, electrical).
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from catalogbank_ocr.stage2.llm_extractor import (
    ExtractionResult,
    ExtractedAttribute,
    ExtractedProduct,
    ExtractedRelation,
)


# ---------------------------------------------------------------------------
# Unit canonicalization tables
# ---------------------------------------------------------------------------

# Maps raw unit strings (lowercase, stripped) → canonical unit label.
# Keys are the most common OCR / LLM variants we see in industrial catalogs.
_UNIT_ALIASES: Dict[str, str] = {
    # Length
    "in": "in", "inch": "in", "inches": "in", '"': "in", "\"": "in",
    "ft": "ft", "foot": "ft", "feet": "ft", "'": "ft",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    "m": "m", "meter": "m", "meters": "m", "metre": "m",
    # Weight / mass
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    # Pressure
    "psi": "psi", "bar": "bar", "kpa": "kPa", "mpa": "MPa",
    "pa": "Pa", "pascal": "Pa", "pascals": "Pa",
    # Temperature
    "°c": "°C", "c": "°C", "celsius": "°C", "deg c": "°C",
    "°f": "°F", "f": "°F", "fahrenheit": "°F", "deg f": "°F",
    # Electrical
    "v": "V", "volt": "V", "volts": "V",
    "a": "A", "amp": "A", "amps": "A", "ampere": "A",
    "w": "W", "watt": "W", "watts": "W",
    "kw": "kW", "kilowatt": "kW",
    "mw": "MW", "megawatt": "MW",
    "hz": "Hz", "hertz": "Hz",
    "khz": "kHz", "mhz": "MHz",
    "ohm": "Ω", "ohms": "Ω",
    # Angle
    "°": "°", "deg": "°", "degree": "°", "degrees": "°",
    # Thread / pitch
    "tpi": "TPI", "pitch": "pitch",
    # Dimensionless / count
    "%": "%", "percent": "%",
}

# Regex: optional leading/trailing whitespace, a numeric part, optional unit
_VALUE_UNIT_RE = re.compile(
    r"^\s*"
    r"(?P<value>[+-]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<unit>[^\d\s].*?)?"
    r"\s*$",
    re.IGNORECASE,
)

# Regex for numbers with units embedded (e.g. "25.4mm", "3/8 in")
_EMBEDDED_RE = re.compile(
    r"(?P<num>(?:\d+/\d+|\d+(?:\.\d+)?))\s*(?P<unit>[a-zA-Z\"\'°]+)",
    re.IGNORECASE,
)

# Fraction pattern: "3/8"
_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")


def _parse_numeric(text: str) -> Optional[float]:
    """Parse a string into a float, handling commas and fractions."""
    text = text.replace(",", "").replace(" ", "").strip()
    frac = _FRACTION_RE.match(text)
    if frac:
        num, denom = int(frac.group(1)), int(frac.group(2))
        return num / denom if denom != 0 else None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class NormalizedValue:
    """A numeric value with a canonical unit label."""
    value: float
    unit: str
    raw: str

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "raw": self.raw}


def parse_value_unit(text: str) -> Optional[NormalizedValue]:
    """Try to extract a (value, unit) pair from a raw attribute value string.

    Returns None if the string does not look like a measurement.

    Strategy (order matters to avoid partial matches):
      1. Fraction+unit pattern  — "3/8 in", "1/4-20" (fractions must go first)
      2. Full-string anchored   — "3,000 psi", "2 in", "25.4 mm"
      3. Embedded search        — "25.4mm", "120V" (run-together, no space)
    """
    if not text:
        return None

    stripped = text.strip()

    # 1. Explicit fraction pattern: "3/8 in", "1/4 in"
    frac_unit_match = re.match(
        r"^(\d+/\d+)\s*([a-zA-Z\"\'°]+)\s*$", stripped, re.IGNORECASE
    )
    if frac_unit_match:
        num = _parse_numeric(frac_unit_match.group(1))
        unit_raw = frac_unit_match.group(2).strip().lower().rstrip(".")
        if num is not None:
            canonical = _UNIT_ALIASES.get(unit_raw, unit_raw)
            return NormalizedValue(value=num, unit=canonical, raw=stripped)

    # 2. Full-string anchored match (comma-thousands, space-separated)
    match = _VALUE_UNIT_RE.match(stripped)
    if match:
        num_str = match.group("value")
        unit_raw = (match.group("unit") or "").strip().lower().rstrip(".")
        num = _parse_numeric(num_str)
        if num is not None and unit_raw:
            canonical = _UNIT_ALIASES.get(unit_raw, unit_raw)
            return NormalizedValue(value=num, unit=canonical, raw=stripped)

    # 3. Embedded pattern: "25.4mm", "120V" (no space between value and unit)
    match = _EMBEDDED_RE.search(stripped)
    if match:
        num_str = match.group("num")
        unit_raw = match.group("unit").strip().lower().rstrip(".")
        num = _parse_numeric(num_str)
        if num is not None:
            canonical = _UNIT_ALIASES.get(unit_raw, unit_raw)
            return NormalizedValue(value=num, unit=canonical, raw=stripped)

    return None


# ---------------------------------------------------------------------------
# String cleaning
# ---------------------------------------------------------------------------

_LEADING_ARTICLES_RE = re.compile(
    r"^(the|a|an)\s+", re.IGNORECASE
)
_WHITESPACE_RE = re.compile(r"\s+")

_UNICODE_REPLACEMENTS = {
    "\u2019": "'",   # right single quotation mark
    "\u2018": "'",   # left single quotation mark
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u00ae": "",    # registered trademark
    "\u2122": "",    # trademark
    "\u00b0": "°",   # degree sign (keep — used in units)
}


def clean_string(text: str) -> str:
    """Normalize a raw string: unicode → ASCII-safe, collapse whitespace."""
    if not text:
        return ""
    # Apply known replacements
    for src, dst in _UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_product_name(name: str) -> str:
    """Clean a product name: strip leading articles, trim whitespace.

    Does NOT force title-case because product names in industrial catalogs
    often use ALL-CAPS model numbers which should be preserved.
    """
    name = clean_string(name)
    # Remove leading "The / A / An"
    name = _LEADING_ARTICLES_RE.sub("", name).strip()
    return name


def normalize_sku(sku: str) -> str:
    """Canonical SKU: uppercase, strip outer whitespace, normalize separators."""
    if not sku:
        return ""
    sku = clean_string(sku).upper()
    # Collapse multiple spaces/hyphens but keep single hyphens (part of SKU)
    sku = re.sub(r"[\s]+", "-", sku)
    sku = re.sub(r"-{2,}", "-", sku).strip("-")
    return sku


def normalize_attribute_key(key: str) -> str:
    """Canonical attribute key: title-case, strip whitespace."""
    return clean_string(key).title()


def normalize_attribute_value(value: str, unit: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Normalize an attribute value and its unit.

    Returns (normalized_value_str, canonical_unit_or_none).
    If the value is a measurement, we attempt unit canonicalization.
    """
    value = clean_string(value)

    # If a unit was already supplied by the LLM, canonicalize it
    if unit:
        unit_clean = clean_string(unit).strip().lower().rstrip(".")
        canonical_unit = _UNIT_ALIASES.get(unit_clean, unit_clean) or None
    else:
        canonical_unit = None

    # Try to parse a numeric value+unit even if unit not explicitly provided
    parsed = parse_value_unit(value)
    if parsed is not None:
        # Prefer explicitly provided unit over parsed unit
        final_unit = canonical_unit or parsed.unit
        # Return just the numeric part as the value string
        return str(parsed.value), final_unit

    return value, canonical_unit


# ---------------------------------------------------------------------------
# Normalized output types
# ---------------------------------------------------------------------------

@dataclass
class NormalizedProduct:
    name: str
    model: Optional[str]
    sku: Optional[str]          # normalized SKU
    sku_raw: Optional[str]      # original as-extracted
    category: Optional[str]
    source_chunks: List[str] = field(default_factory=list)  # chunk_ids it appeared in

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "sku": self.sku,
            "sku_raw": self.sku_raw,
            "category": self.category,
            "source_chunks": self.source_chunks,
        }


@dataclass
class NormalizedAttribute:
    entity: str
    key: str
    value: str
    unit: Optional[str]
    value_numeric: Optional[float]   # populated if value is a measurement
    source_chunk: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "key": self.key,
            "value": self.value,
            "unit": self.unit,
            "value_numeric": self.value_numeric,
            "source_chunk": self.source_chunk,
        }


@dataclass
class NormalizedRelation:
    subject: str
    predicate: str
    object: str
    source_chunk: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source_chunk": self.source_chunk,
        }


@dataclass
class NormalizedResult:
    """Cleaned, deduplicated extraction output for one document (all chunks)."""

    doc_stem: str
    products: List[NormalizedProduct] = field(default_factory=list)
    attributes: List[NormalizedAttribute] = field(default_factory=list)
    relations: List[NormalizedRelation] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_stem": self.doc_stem,
            "product_count": len(self.products),
            "attribute_count": len(self.attributes),
            "relation_count": len(self.relations),
            "source_chunks": self.source_chunk_ids,
            "products": [p.to_dict() for p in self.products],
            "attributes": [a.to_dict() for a in self.attributes],
            "relations": [r.to_dict() for r in self.relations],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_result(result: ExtractionResult) -> NormalizedResult:
    """Normalize a single ExtractionResult.

    Useful when processing one chunk at a time.  For cross-chunk dedup, use
    normalize_batch() instead.
    """
    return normalize_batch([result])


def normalize_batch(results: List[ExtractionResult]) -> NormalizedResult:
    """Normalize and deduplicate a list of ExtractionResults for one document.

    Cross-chunk deduplication:
      - Products: keyed by (normalized_name, normalized_sku).  Same product
        appearing in multiple chunks is merged into one entry; source_chunks
        lists all chunk_ids it appeared in.
      - Attributes: keyed by (entity_norm, key_norm).  First occurrence wins;
        duplicates are dropped.
      - Relations: keyed by (subject_norm, predicate, object_norm).  Exact
        duplicates are dropped.
    """
    if not results:
        return NormalizedResult(doc_stem="")

    doc_stem = results[0].doc_stem

    # Dedup accumulators
    # product key → NormalizedProduct
    products_map: Dict[Tuple[str, str], NormalizedProduct] = {}
    # (entity_norm, key_norm) → NormalizedAttribute
    attrs_map: Dict[Tuple[str, str], NormalizedAttribute] = {}
    # (subject_norm, predicate, object_norm) → NormalizedRelation
    rels_map: Dict[Tuple[str, str, str], NormalizedRelation] = {}
    all_chunk_ids: List[str] = []

    for res in results:
        all_chunk_ids.append(res.chunk_id)

        # ---- Products -------------------------------------------------------
        for p in res.products:
            name_norm = normalize_product_name(p.name)
            sku_norm = normalize_sku(p.sku) if p.sku else ""
            model_norm = clean_string(p.model) if p.model else None
            cat_norm = clean_string(p.category) if p.category else None

            key = (name_norm.lower(), sku_norm.lower())
            if key in products_map:
                # Merge: add this chunk to source_chunks
                products_map[key].source_chunks.append(res.chunk_id)
                # Fill in missing fields from this occurrence
                if not products_map[key].model and model_norm:
                    products_map[key].model = model_norm
                if not products_map[key].category and cat_norm:
                    products_map[key].category = cat_norm
            else:
                products_map[key] = NormalizedProduct(
                    name=name_norm,
                    model=model_norm,
                    sku=sku_norm or None,
                    sku_raw=p.sku,
                    category=cat_norm,
                    source_chunks=[res.chunk_id],
                )

        # ---- Attributes -----------------------------------------------------
        for a in res.attributes:
            entity_norm = normalize_product_name(a.entity)
            key_norm = normalize_attribute_key(a.key)
            value_norm, unit_canon = normalize_attribute_value(a.value, a.unit)

            # Try to extract numeric value
            parsed = parse_value_unit(a.value)
            numeric = parsed.value if parsed else None

            dedup_key = (entity_norm.lower(), key_norm.lower())
            if dedup_key not in attrs_map:
                attrs_map[dedup_key] = NormalizedAttribute(
                    entity=entity_norm,
                    key=key_norm,
                    value=value_norm,
                    unit=unit_canon,
                    value_numeric=numeric,
                    source_chunk=res.chunk_id,
                )
            # else: first-seen wins; duplicate dropped silently

        # ---- Relations ------------------------------------------------------
        for r in res.relations:
            subj_norm = normalize_product_name(r.subject)
            pred_norm = r.predicate.upper().strip()
            obj_norm = clean_string(r.object)

            dedup_key = (subj_norm.lower(), pred_norm, obj_norm.lower())
            if dedup_key not in rels_map:
                rels_map[dedup_key] = NormalizedRelation(
                    subject=subj_norm,
                    predicate=pred_norm,
                    object=obj_norm,
                    source_chunk=res.chunk_id,
                )

    return NormalizedResult(
        doc_stem=doc_stem,
        products=list(products_map.values()),
        attributes=list(attrs_map.values()),
        relations=list(rels_map.values()),
        source_chunk_ids=all_chunk_ids,
    )
