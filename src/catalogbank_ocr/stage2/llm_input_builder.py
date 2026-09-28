"""Stage 2 — Step 3: LLM Input Builder.

Assembles a ContextChunk (+ its ParsedTables) into a compact, token-budgeted
dict that can be serialised to JSON and dropped directly into an LLM prompt.

Design goals:
  - Keep it small: only the fields the LLM actually needs, nothing else.
  - Token budget: configurable max characters per text field and per table.
  - Self-contained: includes a ready-to-use prompt string (instruction +
    context JSON + output schema) so Phase B can just call the LLM.
  - Stable schema: the LLM output JSON schema is embedded as a comment so
    the extraction prompt and the validator (Phase C) stay in sync.

Output dict schema
------------------
{
  "chunk_id":     str,
  "doc_stem":     str,
  "section_path": list[str],
  "context": {
    "section_heading": str,
    "specifications":  list[str],          # one string per spec block
    "product_cards":   list[str],          # first line of each product card
    "tables":          list[{              # parsed table rows (trimmed)
      "block_id": str,
      "headers":  list[str],
      "rows":     list[dict[str, str]],    # up to MAX_TABLE_ROWS rows
    }],
    "supporting_text": str,               # joined paragraph text, truncated
  },
  "extraction_schema": { ... },           # what the LLM must output
  "prompt": str,                          # full ready-to-send prompt string
}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from catalogbank_ocr.stage2.context_builder import ContextChunk
from catalogbank_ocr.stage2.table_parser import ParsedTable, parse_table_block


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default character budgets — intentionally conservative for 8B models.
DEFAULT_MAX_SPEC_CHARS: int = 400    # per specification block
DEFAULT_MAX_PARA_CHARS: int = 600    # total for all paragraph text
DEFAULT_MAX_TABLE_ROWS: int = 20     # per table
DEFAULT_MAX_TABLE_CELL_CHARS: int = 80   # per cell value
DEFAULT_MAX_PRODUCT_CARD_CHARS: int = 300  # per product card


# ---------------------------------------------------------------------------
# Extraction schema — shared between builder and future validator (Phase C)
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA: Dict[str, Any] = {
    "products": [
        {
            "name": "string — product name or model name",
            "model": "string or null — model number / part number",
            "sku": "string or null — SKU / catalog number",
            "category": "string or null — product category inferred from section heading",
        }
    ],
    "attributes": [
        {
            "entity": "string — product name or model this attribute belongs to",
            "key": "string — attribute name (e.g. 'Material', 'Thread Size', 'Length')",
            "value": "string — attribute value",
            "unit": "string or null — unit of measure if applicable (e.g. 'mm', 'in', 'kg')",
        }
    ],
    "relations": [
        {
            "subject": "string — source entity name",
            "predicate": "string — one of: HAS_ATTRIBUTE, IS_VARIANT_OF, BELONGS_TO_CATEGORY, HAS_MATERIAL",
            "object": "string — target entity or value",
        }
    ],
}

# The instruction block injected before the context JSON in every prompt.
_PROMPT_INSTRUCTION = """\
You are an industrial product catalog parser. Extract structured product data \
from the catalog section below and return ONLY a valid JSON object matching the \
schema provided. Do not add commentary, markdown fences, or any text outside the JSON.

EXTRACTION SCHEMA:
{schema}

CATALOG SECTION:
{context_json}

Return a single JSON object with keys: "products", "attributes", "relations".\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    """Truncate a string to max_chars, appending '…' if cut."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _trim_table(parsed: ParsedTable, max_rows: int, max_cell_chars: int) -> Dict[str, Any]:
    """Return a trimmed table dict safe for JSON embedding."""
    trimmed_rows = []
    for row in parsed.rows[:max_rows]:
        trimmed_rows.append(
            {k: _truncate(str(v), max_cell_chars) for k, v in row.items()}
        )
    result: Dict[str, Any] = {
        "block_id": parsed.block_id,
        "parse_method": parsed.parse_method,
        "headers": parsed.headers,
        "rows": trimmed_rows,
    }
    if len(parsed.rows) > max_rows:
        result["rows_truncated"] = len(parsed.rows) - max_rows
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_llm_input(
    chunk: ContextChunk,
    parsed_tables: Optional[List[ParsedTable]] = None,
    max_spec_chars: int = DEFAULT_MAX_SPEC_CHARS,
    max_para_chars: int = DEFAULT_MAX_PARA_CHARS,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    max_table_cell_chars: int = DEFAULT_MAX_TABLE_CELL_CHARS,
    max_product_card_chars: int = DEFAULT_MAX_PRODUCT_CARD_CHARS,
    include_prompt: bool = True,
) -> Dict[str, Any]:
    """Build a complete LLM input dict from a ContextChunk.

    Parameters
    ----------
    chunk
        The ContextChunk produced by context_builder.
    parsed_tables
        Pre-parsed tables for this chunk.  If None, table blocks in the chunk
        are parsed inline via table_parser.parse_table_block().
    max_spec_chars / max_para_chars / max_table_rows / max_table_cell_chars
        Token-budget controls — truncate long content to stay within limits.
    include_prompt
        If True, add a ``"prompt"`` key with a complete ready-to-send string.
        Set to False if you only need the structured context dict.

    Returns
    -------
    dict
        Self-contained LLM input.  Serialise with ``json.dumps(..., indent=2)``.
    """

    # ---- Resolve parsed tables -------------------------------------------
    if parsed_tables is None:
        parsed_tables = [parse_table_block(b) for b in chunk.tables]

    # Build a lookup by block_id for alignment
    table_by_id: Dict[str, ParsedTable] = {t.block_id: t for t in parsed_tables}

    # ---- Specifications ---------------------------------------------------
    spec_texts: List[str] = []
    for blk in chunk.specifications:
        spec_texts.append(_truncate(blk.text, max_spec_chars))

    # ---- Product cards ----------------------------------------------------
    product_card_texts: List[str] = []
    for blk in chunk.product_cards:
        product_card_texts.append(_truncate(blk.text, max_product_card_chars))

    # ---- Tables -----------------------------------------------------------
    table_dicts: List[Dict[str, Any]] = []
    for blk in chunk.tables:
        pt = table_by_id.get(blk.block_id)
        if pt is None:
            pt = parse_table_block(blk)
        if not pt.is_empty:
            table_dicts.append(_trim_table(pt, max_table_rows, max_table_cell_chars))
        else:
            # Table parse failed — include raw text so LLM still sees it
            table_dicts.append({
                "block_id": blk.block_id,
                "parse_method": "raw",
                "raw_text": _truncate(blk.text, max_spec_chars),
            })

    # ---- Supporting paragraphs -------------------------------------------
    para_parts: List[str] = []
    budget = max_para_chars
    for blk in chunk.paragraphs:
        if budget <= 0:
            break
        part = _truncate(blk.text, budget)
        if part:
            para_parts.append(part)
            budget -= len(part)
    supporting_text = " ".join(para_parts).strip()

    # ---- Assemble context dict -------------------------------------------
    context: Dict[str, Any] = {
        "section_heading": chunk.section_name,
        "section_path": chunk.section_path,
    }
    if spec_texts:
        context["specifications"] = spec_texts
    if product_card_texts:
        context["product_cards"] = product_card_texts
    if table_dicts:
        context["tables"] = table_dicts
    if supporting_text:
        context["supporting_text"] = supporting_text

    # ---- Top-level output dict -------------------------------------------
    output: Dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "doc_stem": chunk.doc_stem,
        "section_path": chunk.section_path,
        "context": context,
        "extraction_schema": EXTRACTION_SCHEMA,
    }

    if include_prompt:
        output["prompt"] = _PROMPT_INSTRUCTION.format(
            schema=json.dumps(EXTRACTION_SCHEMA, indent=2),
            context_json=json.dumps(context, indent=2),
        )

    return output


def build_llm_inputs_for_chunks(
    chunks: List[ContextChunk],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Build LLM input dicts for a list of ContextChunks.

    All keyword arguments are forwarded to ``build_llm_input()``.
    """
    return [build_llm_input(chunk, **kwargs) for chunk in chunks]


def estimate_prompt_chars(llm_input: Dict[str, Any]) -> int:
    """Return the character length of the prompt string (proxy for token count).

    Rough rule of thumb: ~4 chars per token for English technical text.
    """
    prompt = llm_input.get("prompt", "")
    return len(prompt)
