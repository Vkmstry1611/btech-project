"""Stage 2 — Step 5b: Normalization.

Takes ExtractionResult objects (from Phase B) and produces a
NormalizedResult with cleaned, deduplicated, and canonicalized data
ready for Neo4j ingestion.

What this module does
---------------------
Preserves manufacturer identifiers and descriptions, merges exact SKU identities,
canonicalizes unambiguous measurements, and retains conflicting attribute values
with their chunk provenance. It does not convert physical units or guess values.

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
        if num is not None and unit_raw in _UNIT_ALIASES:
            canonical = _UNIT_ALIASES[unit_raw]
            return NormalizedValue(value=num, unit=canonical, raw=stripped)

    # 2. Full-string anchored match (comma-thousands, space-separated)
    match = _VALUE_UNIT_RE.match(stripped)
    if match:
        num_str = match.group("value")
        unit_raw = (match.group("unit") or "").strip().lower().rstrip(".")
        num = _parse_numeric(num_str)
        if num is not None and unit_raw in _UNIT_ALIASES:
            canonical = _UNIT_ALIASES.get(unit_raw, unit_raw)
            return NormalizedValue(value=num, unit=canonical, raw=stripped)

    # 3. Embedded pattern: "25.4mm", "120V" (no space between value and unit)
    match = _EMBEDDED_RE.fullmatch(stripped)
    if match:
        num_str = match.group("num")
        unit_raw = match.group("unit").strip().lower().rstrip(".")
        num = _parse_numeric(num_str)
        if num is not None and unit_raw in _UNIT_ALIASES:
            canonical = _UNIT_ALIASES[unit_raw]
            return NormalizedValue(value=num, unit=canonical, raw=stripped)

    return None


# ---------------------------------------------------------------------------
# String cleaning
# ---------------------------------------------------------------------------

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
    """Clean a product name while preserving its wording and case.

    Does NOT force title-case because product names in industrial catalogs
    often use ALL-CAPS model numbers which should be preserved.
    """
    name = clean_string(name)
    # Manufacturer names and codes may legitimately start with A or The.
    return name


def normalize_sku(sku: str) -> str:
    """Preserve the exact code, trimming only surrounding whitespace."""
    if not sku:
        return ""
    return sku.strip()  # Preserve manufacturer codes exactly; never invent separators.


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
        if canonical_unit and canonical_unit != parsed.unit:
            raise ValueError(f'Conflicting attribute units: {value!r} versus {unit!r}')
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
    description: Optional[str] = None
    variant_group: Optional[str] = None
    source_evidence: List[Dict[str, Any]] = field(default_factory=list)
    category_candidates: List[str] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)  # chunk_ids it appeared in

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "sku": self.sku,
            "sku_raw": self.sku_raw,
            "description": self.description,
            "variant_group": self.variant_group,
            "source_evidence": self.source_evidence,
            "category_candidates": self.category_candidates,
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
    location: Optional[str] = None
    product_sku: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "key": self.key,
            "value": self.value,
            "unit": self.unit,
            "value_numeric": self.value_numeric,
            "source_chunk": self.source_chunk,
            "location": self.location,
            "product_sku": self.product_sku,
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
      - Attributes: exact values are coalesced within their source chunk;
        conflicting values and distinct SKU owners are retained.
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

    aliases = {}
    sku_aliases = {}
    canonical_names = {}
    group_parent, sku_group = {}, {}
    def group_root(group):
        group_parent.setdefault(group, group)
        while group_parent[group] != group:
            group_parent[group] = group_parent[group_parent[group]]
            group = group_parent[group]
        return group
    for result in results:
        for product in result.products:
            if product.sku and product.variant_group:
                sku = product.sku.strip()
                root = group_root(product.variant_group)
                if sku in sku_group:
                    prior = group_root(sku_group[sku])
                    group_parent[max(root, prior)] = min(root, prior)
                sku_group[sku] = product.variant_group
            identity = ('sku', product.sku.strip()) if product.sku else ('name', product.name, product.model)
            canonical = canonical_names.setdefault(identity, normalize_product_name(product.name))
            for alias in (product.name, product.sku):
                if alias: aliases.setdefault(alias, set()).add(canonical)
                if alias and product.sku:
                    sku_aliases.setdefault(alias, set()).add(product.sku.strip())
    for res in results:
        all_chunk_ids.append(res.chunk_id)

        # ---- Products -------------------------------------------------------
        for p in res.products:
            name_norm = normalize_product_name(p.name)
            sku_norm = normalize_sku(p.sku) if p.sku else ""
            model_norm = p.model.strip() if p.model else None
            cat_norm = clean_string(p.category) if p.category else None

            key = ("sku", sku_norm) if sku_norm else ('name', name_norm, model_norm or '')
            if key in products_map:
                # Merge: add this chunk to source_chunks
                if res.chunk_id not in products_map[key].source_chunks:
                    products_map[key].source_chunks.append(res.chunk_id)
                for evidence in p.source_evidence:
                    if evidence not in products_map[key].source_evidence:
                        products_map[key].source_evidence.append(evidence)
                if p.description:
                    previous = products_map[key].description or ''
                    if not previous or previous in p.description:
                        products_map[key].description = p.description
                    elif p.description not in previous:
                        products_map[key].description = previous + '\n\n' + p.description
                # Fill in missing fields from this occurrence
                if not products_map[key].model and model_norm:
                    products_map[key].model = model_norm
                if not products_map[key].category and cat_norm:
                    products_map[key].category = cat_norm
                if cat_norm and cat_norm not in products_map[key].category_candidates:
                    products_map[key].category_candidates.append(cat_norm)
            else:
                products_map[key] = NormalizedProduct(
                    name=name_norm,
                    model=model_norm,
                    sku=sku_norm or None,
                    sku_raw=p.sku,
                    description=p.description,
                    variant_group=p.variant_group,
                    source_evidence=list(p.source_evidence),
                    category=cat_norm,
                    category_candidates=[cat_norm] if cat_norm else [],
                    source_chunks=[res.chunk_id],
                )

        # ---- Attributes -----------------------------------------------------
        for a in res.attributes:
            matches = aliases.get(a.entity, set())
            entity_norm = next(iter(matches)) if len(matches) == 1 else a.entity.strip()
            sku_matches = sku_aliases.get(a.entity, set())
            product_sku = next(iter(sku_matches)) if len(sku_matches) == 1 else None
            key_norm = normalize_attribute_key(a.key)
            if key_norm.lower() in {'sku', 'model', 'part number', 'article number', 'description', 'thread size'}:
                value_norm, unit_canon = a.value.strip(), a.unit
            else:
                value_norm, unit_canon = normalize_attribute_value(a.value, a.unit)

            # Try to extract numeric value
            parsed = parse_value_unit(a.value)
            numeric = parsed.value if parsed else None

            dedup_key = (entity_norm, product_sku, key_norm, value_norm, unit_canon, a.location, res.chunk_id)
            if dedup_key not in attrs_map:
                attrs_map[dedup_key] = NormalizedAttribute(
                    entity=entity_norm,
                    key=key_norm,
                    value=value_norm,
                    unit=unit_canon,
                    value_numeric=numeric,
                    location=a.location,
                    product_sku=product_sku,
                    source_chunk=res.chunk_id,
                )
            # Exact repeats are coalesced; conflicting values retain their source chunk.

        # ---- Relations ------------------------------------------------------
        for r in res.relations:
            subj_norm = normalize_product_name(r.subject)
            pred_norm = r.predicate.upper().strip()
            obj_norm = clean_string(r.object)

            dedup_key = (subj_norm, pred_norm, obj_norm)
            if dedup_key not in rels_map:
                rels_map[dedup_key] = NormalizedRelation(
                    subject=subj_norm,
                    predicate=pred_norm,
                    object=obj_norm,
                    source_chunk=res.chunk_id,
                )

    # A repeated SKU links explicit variant rows across page/chunk boundaries.
    # Never merge rows merely because their headings or descriptions match.
    for product in products_map.values():
        if product.sku in sku_group:
            product.variant_group = group_root(sku_group[product.sku])

    return NormalizedResult(
        doc_stem=doc_stem,
        products=list(products_map.values()),
        attributes=list(attrs_map.values()),
        relations=list(rels_map.values()),
        source_chunk_ids=all_chunk_ids,
    )
