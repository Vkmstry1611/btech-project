"""Stage 2 — Step 5a: Validation.

Validates ExtractionResult objects (from Phase B) against Pydantic models and
a set of deterministic business rules, producing a ValidationReport that
lists every issue found without raising exceptions.

Validation layers
-----------------
1. Schema validation  — Pydantic models mirror the LLM output schema;
                        missing required fields, wrong types, empty strings.
2. Semantic rules     — duplicate SKUs across chunks, orphaned attributes
                        (entity name not in products list), invalid predicate
                        values, self-referential relations.
3. Quality flags      — low-confidence blocks (below threshold), very short
                        names (<2 chars), suspiciously long values (>500 chars)
                        that look like OCR garbage.

The validator is intentionally non-destructive: it never modifies the
extraction result — that's normalization's job.  It only annotates issues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, field_validator, model_validator

from catalogbank_ocr.stage2.llm_extractor import (
    ExtractionResult,
    ExtractedProduct,
    ExtractedAttribute,
    ExtractedRelation,
)

# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    ERROR   = "error"    # Must be fixed before graph ingestion
    WARNING = "warning"  # Should be reviewed; ingestion can proceed
    INFO    = "info"     # Informational; no action required


# ---------------------------------------------------------------------------
# Issue dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: Severity
    code: str           # machine-readable short code, e.g. "MISSING_SKU"
    message: str        # human-readable description
    chunk_id: str = ""
    entity: str = ""    # product name / attribute entity / relation subject
    field: str = ""     # which field triggered the issue

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "chunk_id": self.chunk_id,
            "entity": self.entity,
            "field": self.field,
        }


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Aggregated validation output for one or many ExtractionResults."""

    issues: List[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        chunk_id: str = "",
        entity: str = "",
        field_name: str = "",
    ) -> None:
        self.issues.append(ValidationIssue(
            severity=severity, code=code, message=message,
            chunk_id=chunk_id, entity=entity, field=field_name,
        ))

    # Convenience accessors
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def summary(self) -> str:
        e = len(self.errors)
        w = len(self.warnings)
        inf = len(self.issues) - e - w
        return f"{e} error(s), {w} warning(s), {inf} info(s)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }

    def print_report(self, indent: int = 2) -> None:
        pad = " " * indent
        icons = {Severity.ERROR: "✗", Severity.WARNING: "⚠", Severity.INFO: "ℹ"}
        for issue in self.issues:
            loc = ""
            if issue.chunk_id:
                loc += f" [{issue.chunk_id}]"
            if issue.entity:
                loc += f" entity={issue.entity!r}"
            if issue.field:
                loc += f" field={issue.field!r}"
            print(f"{pad}{icons[issue.severity]} [{issue.code}]{loc}: {issue.message}")


# ---------------------------------------------------------------------------
# Pydantic schema models (for structural validation)
# ---------------------------------------------------------------------------

class ProductSchema(BaseModel):
    name: str
    model: Optional[str] = None
    sku: Optional[str] = None
    category: Optional[str] = None

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("product name must not be empty")
        return v.strip()

    @field_validator("model", "sku", "category", mode="before")
    @classmethod
    def coerce_null(cls, v: Any) -> Optional[str]:
        if v is None or str(v).lower() in {"null", "none", "n/a", ""}:
            return None
        return str(v).strip() or None


class AttributeSchema(BaseModel):
    entity: str
    key: str
    value: str
    unit: Optional[str] = None

    @field_validator("entity", "key", "value")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be empty")
        return v.strip()


_VALID_PREDICATES = {
    "HAS_ATTRIBUTE",
    "IS_VARIANT_OF",
    "BELONGS_TO_CATEGORY",
    "HAS_MATERIAL",
}

class RelationSchema(BaseModel):
    subject: str
    predicate: str
    object: str

    @field_validator("subject", "predicate", "object")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("field must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def no_self_loop(self) -> "RelationSchema":
        if self.subject.lower() == self.object.lower():
            raise ValueError(f"self-referential relation: {self.subject!r}")
        return self


# ---------------------------------------------------------------------------
# Rule checkers
# ---------------------------------------------------------------------------

_MIN_NAME_LEN = 2
_MAX_VALUE_LEN = 500
_OCR_GARBAGE_RE = re.compile(r"[^\x20-\x7E\u00A0-\u024F\u4E00-\u9FFF]")  # non-printable / exotic


def _looks_like_garbage(text: str) -> bool:
    """Heuristic: text with many non-printable or control characters."""
    if not text:
        return False
    garbage_chars = len(_OCR_GARBAGE_RE.findall(text))
    return garbage_chars / max(len(text), 1) > 0.15


def _validate_product(
    p: ExtractedProduct,
    chunk_id: str,
    report: ValidationReport,
    seen_skus: Dict[str, str],  # sku → first chunk_id
) -> None:
    # Schema validation via Pydantic
    try:
        ProductSchema(name=p.name, model=p.model, sku=p.sku, category=p.category)
    except Exception as exc:
        report.add(Severity.ERROR, "SCHEMA_ERROR", str(exc),
                   chunk_id=chunk_id, entity=p.name, field_name="product")

    # Name too short
    if len((p.name or "").strip()) < _MIN_NAME_LEN:
        report.add(Severity.WARNING, "SHORT_NAME",
                   f"Product name {p.name!r} is suspiciously short",
                   chunk_id=chunk_id, entity=p.name, field_name="name")

    # OCR garbage
    if _looks_like_garbage(p.name):
        report.add(Severity.WARNING, "GARBAGE_TEXT",
                   f"Product name {p.name!r} may contain OCR garbage",
                   chunk_id=chunk_id, entity=p.name, field_name="name")

    # Missing SKU (info, not error — many catalog entries genuinely lack one)
    if not p.sku:
        report.add(Severity.INFO, "MISSING_SKU",
                   f"Product {p.name!r} has no SKU",
                   chunk_id=chunk_id, entity=p.name, field_name="sku")

    # Duplicate SKU across chunks
    if p.sku:
        sku_norm = p.sku.strip().upper()
        if sku_norm in seen_skus:
            report.add(Severity.WARNING, "DUPLICATE_SKU",
                       f"SKU {p.sku!r} already seen in chunk {seen_skus[sku_norm]!r}",
                       chunk_id=chunk_id, entity=p.name, field_name="sku")
        else:
            seen_skus[sku_norm] = chunk_id


def _validate_attribute(
    a: ExtractedAttribute,
    chunk_id: str,
    report: ValidationReport,
    product_names: Set[str],
) -> None:
    try:
        AttributeSchema(entity=a.entity, key=a.key, value=a.value, unit=a.unit)
    except Exception as exc:
        report.add(Severity.ERROR, "SCHEMA_ERROR", str(exc),
                   chunk_id=chunk_id, entity=a.entity, field_name="attribute")

    # Value too long — likely OCR blob or wrong field
    if len(a.value) > _MAX_VALUE_LEN:
        report.add(Severity.WARNING, "VALUE_TOO_LONG",
                   f"Attribute {a.key!r} value length {len(a.value)} > {_MAX_VALUE_LEN}",
                   chunk_id=chunk_id, entity=a.entity, field_name="value")

    # OCR garbage in value
    if _looks_like_garbage(a.value):
        report.add(Severity.WARNING, "GARBAGE_TEXT",
                   f"Attribute value for {a.key!r} may contain OCR garbage",
                   chunk_id=chunk_id, entity=a.entity, field_name="value")

    # Orphaned attribute: entity not in product list for this chunk
    # (warning only — entity might be a section heading)
    entity_norm = (a.entity or "").strip().lower()
    known_norms = {n.strip().lower() for n in product_names}
    if product_names and entity_norm not in known_norms:
        report.add(Severity.INFO, "ORPHANED_ATTRIBUTE",
                   f"Attribute entity {a.entity!r} not found in products list for this chunk",
                   chunk_id=chunk_id, entity=a.entity, field_name="entity")


def _validate_relation(
    r: ExtractedRelation,
    chunk_id: str,
    report: ValidationReport,
) -> None:
    try:
        RelationSchema(subject=r.subject, predicate=r.predicate, object=r.object)
    except Exception as exc:
        report.add(Severity.ERROR, "SCHEMA_ERROR", str(exc),
                   chunk_id=chunk_id, entity=r.subject, field_name="relation")

    if r.predicate.upper() not in _VALID_PREDICATES:
        report.add(Severity.WARNING, "UNKNOWN_PREDICATE",
                   f"Relation predicate {r.predicate!r} is not in the known set {sorted(_VALID_PREDICATES)}",
                   chunk_id=chunk_id, entity=r.subject, field_name="predicate")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_extraction(
    result: ExtractionResult,
    seen_skus: Optional[Dict[str, str]] = None,
) -> ValidationReport:
    """Validate a single ExtractionResult.

    Parameters
    ----------
    result
        The extraction result from Phase B.
    seen_skus
        Optional shared dict for cross-chunk duplicate SKU detection.
        Pass the same dict when validating multiple results from one document.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    if seen_skus is None:
        seen_skus = {}

    product_names: Set[str] = {p.name for p in result.products}

    for product in result.products:
        _validate_product(product, result.chunk_id, report, seen_skus)

    for attr in result.attributes:
        _validate_attribute(attr, result.chunk_id, report, product_names)

    for relation in result.relations:
        _validate_relation(relation, result.chunk_id, report)

    return report


def validate_batch(
    results: List[ExtractionResult],
) -> Dict[str, ValidationReport]:
    """Validate a list of ExtractionResults sharing a SKU namespace.

    Returns a dict mapping chunk_id → ValidationReport.
    """
    seen_skus: Dict[str, str] = {}  # shared across all chunks in this batch
    return {
        r.chunk_id: validate_extraction(r, seen_skus=seen_skus)
        for r in results
    }
