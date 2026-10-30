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
import re
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
            "name": "string — product name or model name (e.g. 'MB4', 'PBG11102')",
            "model": "string or null — model number / part number",
            "sku": "string or null — SKU / catalog number / article number",
            "category": "string or null — product category inferred from section heading",
            "description": "string or null — brief product description",
        }
    ],
    "attributes": [
        {
            "entity": "string — exact product name or SKU this attribute belongs to",
            "location": "string or null — explicit market/country for a price; never infer",
            "key": "string — attribute name (e.g. 'Material', 'Thread Size', 'Length', 'Price')",
            "value": "string — attribute value",
            "unit": "string or null — unit of measure if applicable (e.g. 'mm', 'in', 'kg', 'USD', 'EUR')",
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
You are extracting product data from an industrial catalog page. \
Return ONLY a valid JSON object. No markdown, no explanation, no text outside the JSON.

CRITICAL RULES:
1. "products" must contain ONLY actual product items with model numbers or SKUs \
   (e.g. "MB4", "PBG11102", "MBC12"). \
   Do NOT use section headings or category names as product names.
2. If a table has rows like "MB4 | $49.40 | 4\\"x6\\"", each row is ONE product.
3. Use the exact model/SKU code as the "name" (e.g. "MB4", not "Optical Breadboard").
4. Put the full descriptive name in "description".
5. Put the SKU/part number in "sku".
6. For prices: entity=the product SKU, key="Price", value=numeric amount, unit=currency.
7. For dimensions: entity=the product SKU, key="Dimensions", value=the size string.

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


def build_complete_inputs(chunks, max_chars=6000):
    """Page/row-aware bounded requests; no silent field or table truncation.

    Oversized individual rows fail explicitly instead of publishing incomplete data.
    Full canonical evidence remains available alongside each request's block IDs.
    """
    inputs = []
    rules = '''Extract only products supported by the evidence. Headings and index entries
are context, never products by themselves. Preserve exact SKU/article codes including
case, punctuation and spaces. A model is a design/series; a SKU is a sellable variant.
Do not copy a SKU into model unless evidence explicitly identifies it as a model.
If a section only describes a product family and gives no specific sellable item,
return empty arrays. Do not create a sellable item from that section title.
Use null for unknown values. Keep the descriptive product name and description.
Attributes must reference the exact product name or SKU. Do not invent currency.
Relations must describe explicit links between distinct products or categories.
Do not repeat attributes as HAS_ATTRIBUTE relations; the graph builder creates
those links from the attributes array. Do not invent compatibility or variants.
Extract EVERY distinct SKU from item/article/part-number columns, including separate
metric and imperial columns in the same row. Preserve prices and their explicit
currencies; do not assume a currency for an ambiguous symbol or shift misaligned cells.
Return ONLY JSON with products, attributes and relations matching this schema: '''
    for chunk in chunks:
        records = []
        previous_table = None
        for bucket in ('specifications', 'product_cards', 'paragraphs', 'tables'):
            for block in getattr(chunk, bucket):
                base = {'block_id': block.block_id, 'page': block.page, 'kind': bucket}
                if bucket == 'tables':
                    table = parse_table_block(block)
                    if table.parse_method != 'raw' and table.rows:
                        if previous_table and not table.headers and block.page == previous_table['page'] + 1:
                            base['possible_continuation_of'] = previous_table['block_id']
                            base['previous_headers'] = previous_table['headers']
                        records.extend(dict(base, headers=table.headers, row=row) for row in table.rows)
                        previous_table = {'page': block.page, 'block_id': block.block_id, 'headers': table.headers}
                        continue
                text = block.text
                # Dense OCR paragraphs can contain dozens of concatenated catalog
                # rows. A character-budget-only request may require the model to
                # emit too many products and time out, so keep whole lines in
                # smaller requests. Newlines are row boundaries, not SKU bounds.
                dense_codes = len(re.findall(r'\d{4,5}[A-Z]\d{2,4}', text)) >= 20
                if dense_codes and '\n' in text:
                    offset, pending, start = 0, '', 0
                    for line in text.splitlines(keepends=True):
                        if pending and len(pending) + len(line) > 700:
                            records.append(dict(base, text=pending, offset=start, dense_text=True))
                            pending, start = '', offset
                        pending += line
                        offset += len(line)
                    if pending: records.append(dict(base, text=pending, offset=start, dense_text=True))
                else:
                    # Overlap ordinary text windows to preserve entities at a
                    # character-window boundary.
                    for start in range(0, max(1, len(text)), 2800):
                        records.append(dict(base, text=text[start:start+3000], offset=start))
        expanded = []
        for record in records:
            if len(json.dumps(record, ensure_ascii=False)) <= max_chars:
                expanded.append(record)
                continue
            if 'row' not in record:
                raise ValueError(f'Oversized evidence record: {record["block_id"]}')
            base = {key:value for key,value in record.items() if key != 'row'}
            fragments = []
            for column, value in record['row'].items():
                if not isinstance(value, str): value = str(value)
                starts = range(0, max(1, len(value)), 2800) if len(value) > 3000 else [0]
                for start in starts:
                    piece = dict(base, row={column:value[start:start+3000]},
                                 cell_offsets={column:start}, cell_lengths={column:len(value)})
                    if len(json.dumps(piece, ensure_ascii=False)) > max_chars:
                        raise ValueError(f'Oversized table cell: {record["block_id"]}/{column}')
                    fragments.append(piece)
            # Pack adjacent columns together. A one-cell overlap preserves a
            # literal code in one cell and its price in the following cell.
            pending = None
            for piece in fragments:
                if pending is None:
                    pending = piece
                    continue
                candidate = dict(pending, row={**pending['row'], **piece['row']},
                                 cell_offsets={**pending['cell_offsets'], **piece['cell_offsets']},
                                 cell_lengths={**pending['cell_lengths'], **piece['cell_lengths']})
                if (next(iter(piece['cell_offsets'].values())) == 0 and not (set(piece['row']) & set(pending['row']))
                    and len(json.dumps(candidate, ensure_ascii=False)) <= max_chars):
                    pending = candidate
                else:
                    expanded.append(pending)
                    previous = list(pending['row'].items())[-1]
                    overlap = {previous[0]:previous[1]} if len(previous[1]) < 500 else {}
                    pending = dict(piece, row={**overlap, **piece['row']},
                                   cell_offsets={**{key:pending['cell_offsets'][key] for key in overlap}, **piece['cell_offsets']},
                                   cell_lengths={**{key:pending['cell_lengths'][key] for key in overlap}, **piece['cell_lengths']})
                    if len(json.dumps(pending, ensure_ascii=False)) > max_chars:
                        pending = piece
            if pending is not None: expanded.append(pending)
        groups, group, size = [], [], 0
        for record in expanded:
            length = len(json.dumps(record, ensure_ascii=False))
            if length > max_chars: raise ValueError(f'Oversized evidence row: {record["block_id"]}')
            if group and (size + length > max_chars or len(group) >= 8 or
                          record.get('dense_text') or any(v.get('dense_text') for v in group)):
                groups.append(group); group, size = [], 0
            group.append(record); size += length
        if group: groups.append(group)
        for i, evidence in enumerate(groups):
            context = {'section_path': chunk.section_path, 'continuation': i+1, 'parts': len(groups), 'evidence': evidence}
            inputs.append({'chunk_id': f'{chunk.chunk_id}__part{i+1}', 'doc_stem': chunk.doc_stem,
                           'section_path': chunk.section_path, 'context': context,
                           'prompt': rules + json.dumps(EXTRACTION_SCHEMA) + '\nEVIDENCE:\n' + json.dumps(context, ensure_ascii=False)})
    return inputs


def estimate_prompt_chars(llm_input: Dict[str, Any]) -> int:
    """Return the character length of the prompt string (proxy for token count).

    Rough rule of thumb: ~4 chars per token for English technical text.
    """
    prompt = llm_input.get("prompt", "")
    return len(prompt)
