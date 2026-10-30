"""Stage 2 — Step 2: Table Parser.

Converts a table block's raw text content into a structured list of row dicts,
making it easy for the LLM Input Builder to embed compact table data in prompts.

PP-StructureV3 produces tables in two possible formats:
  1. HTML  — ``<html><body><table>...</table></body></html>`` (SLANeXt output)
  2. Markdown pipe tables  — ``| col1 | col2 |\\n|---|---|\\n| val | val |``

Strategy:
  - Try HTML parse first (BeautifulSoup); if that yields rows, return them.
  - Fall back to the markdown pipe-table parser.
  - If both fail, return a single-row dict with the raw text under key "raw_text".

The parser is intentionally lenient: malformed HTML or ragged markdown tables
are handled gracefully rather than raising exceptions.

Dependencies:
  - beautifulsoup4 (optional but recommended; falls back to regex if missing)
  - No pandas needed — we return plain list[dict[str, str]]
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from catalogbank_ocr.stage2.context_builder import RawBlock

# ---------------------------------------------------------------------------
# Optional import — BeautifulSoup
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

class ParsedTable:
    """Result of parsing one table block.

    Attributes
    ----------
    block_id    Source block ID for traceability.
    headers     Column header strings (may be empty if no <th> / header row).
    rows        List of row dicts mapping header → cell value.
                If no headers were detected, keys are ``"col_0"``, ``"col_1"``, …
    parse_method
                ``"html"``, ``"markdown"``, or ``"raw"`` — indicates which
                parser succeeded.
    raw_text    The original text from the block (always preserved).
    """

    __slots__ = ("block_id", "headers", "rows", "parse_method", "raw_text")

    def __init__(
        self,
        block_id: str,
        headers: List[str],
        rows: List[Dict[str, str]],
        parse_method: str,
        raw_text: str,
    ) -> None:
        self.block_id = block_id
        self.headers = headers
        self.rows = rows
        self.parse_method = parse_method
        self.raw_text = raw_text

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "parse_method": self.parse_method,
            "headers": self.headers,
            "row_count": len(self.rows),
            "rows": self.rows,
        }

    def __repr__(self) -> str:
        return (
            f"ParsedTable(block_id={self.block_id!r}, method={self.parse_method!r}, "
            f"headers={self.headers}, rows={len(self.rows)})"
        )


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def _clean_cell(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_html_table(text: str) -> Optional[tuple[List[str], List[Dict[str, str]]]]:
    """Try to extract headers + rows from an HTML string.

    Returns (headers, rows) on success, None if no table found or BS4 missing.
    """
    if not _BS4_AVAILABLE:
        return None

    # Quick check — if there's no <table> tag, skip expensive parse
    if "<table" not in text.lower():
        return None

    try:
        soup = BeautifulSoup(text, "html.parser")
    except Exception:
        return None

    table = soup.find("table")
    if table is None:
        return None

    grid, header_rows = [], []
    spans = {}
    for row_idx, tr in enumerate(table.find_all('tr')):
        cells = tr.find_all(['td', 'th'], recursive=False)
        if not cells: continue
        row = {}
        for col, (value, remaining) in list(spans.items()):
            row[col] = value
            if remaining == 1: del spans[col]
            else: spans[col] = (value, remaining-1)
        col = 0
        for cell in cells:
            while col in row: col += 1
            try:
                width = max(1, min(100, int(cell.get('colspan', 1))))
                height = max(1, min(1000, int(cell.get('rowspan', 1))))
            except ValueError:
                width = height = 1
            value = _clean_cell(cell.get_text(' '))
            for offset in range(width):
                row[col+offset] = value
                if height > 1: spans[col+offset] = (value, height-1)
            col += width
        grid.append(row)
        if all(c.name == 'th' for c in cells) and len(header_rows) == len(grid)-1:
            header_rows.append(row)
    if not grid: return None
    header_indices = set(range(len(header_rows)))
    if not header_rows:
        # OCR may emit a printed header at the bottom of its HTML grid. Use
        # explicit labels wherever they occur, retaining every other data row.
        header_words = r'^(?:metric\s+|imperial\s+)?(?:item\s*#|sku|part\s*(?:number|no\.?|#)|article\s*(?:number|no\.?|#)|(?:mfg\.?|mfr\.?)\s*part\s*(?:number|no\.?|#)|model|description|details|price(?:\s+each)?|dimensions|weight\*?|material|length|width|height|size)$'
        from catalogbank_ocr.stage2.identifiers import SKU_HEADER
        def labelled(value):
            return bool(re.match(header_words, value.strip(), re.I) or SKU_HEADER.fullmatch(value.strip()) or
                        re.fullmatch(r'(?:\d+(?:-\d+)?\+?\s+)?(?:list\s+)?price(?:\s+each)?(?:\s+\d+(?:-\d+)?\+?)?', value.strip(), re.I) or
                        value.strip().upper() in {'$', '€', '£', '¥', '￥', 'RMB', 'CNY', 'USD', 'EUR', 'GBP', 'JPY', 'INR'})
        candidates = [i for i,row in enumerate(grid)
                      if sum(labelled(v) for v in row.values()) >= 2
                      and (i == 0 or any(SKU_HEADER.fullmatch(v.strip()) for v in row.values()))]
        if candidates:
            strongest = max(sum(labelled(v) for v in grid[i].values()) for i in candidates)
            candidates = [i for i in candidates if sum(labelled(v) for v in grid[i].values()) == strongest]
        if candidates and all(grid[i] == grid[candidates[0]] for i in candidates):
            index = candidates[0]
            header_indices.update(candidates)
            # Retain an adjacent sparse label row, e.g. a separate "Price Each"
            # label beneath the quantity-band header "1-4+".
            for adjacent in (index-1, index+1):
                if 0 <= adjacent < len(grid):
                    values = [v for v in grid[adjacent].values() if v]
                    if values and any(labelled(v) for v in values) and all(labelled(v) or re.fullmatch(r'\d+(?:-\d+)?\+?', v) for v in values):
                        header_indices.add(adjacent)
            header_rows = [grid[i] for i in sorted(header_indices)]
    width = max(max(r, default=-1) for r in grid)+1
    headers = []
    if header_rows:
        for col in range(width):
            name = ' / '.join(dict.fromkeys(r.get(col, '') for r in header_rows if r.get(col))) or f'col_{col}'
            if name in headers: name += f' [{col+1}]'
            headers.append(name)
    keys = headers or [f'col_{i}' for i in range(width)]
    rows = [{k: row.get(i, '') for i, k in enumerate(keys)} for index,row in enumerate(grid)
            if index not in header_indices and any(row.values())]
    # Rowspans can repeat the identical evidence dozens of times. Keep one copy
    # per table; the complete original HTML remains in raw_text for inspection.
    unique_rows = {}
    for row in rows:
        unique_rows.setdefault(tuple(row.items()), row)
    rows = list(unique_rows.values())
    return (headers, rows) if rows else None


# ---------------------------------------------------------------------------
# Markdown pipe-table parser
# ---------------------------------------------------------------------------

_SEPARATOR_RE = re.compile(r"^\s*\|?[\s\-:]+(\|[\s\-:]+)*\|?\s*$")


def _split_pipe_row(line: str) -> List[str]:
    """Split a markdown pipe-table row into cells, stripping outer pipes."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _parse_markdown_table(text: str) -> Optional[tuple[List[str], List[Dict[str, str]]]]:
    """Try to extract headers + rows from a markdown pipe table.

    Returns (headers, rows) on success, None if no valid table found.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    table_lines: List[str] = []

    # Find a contiguous block of pipe-table lines
    in_table = False
    for line in lines:
        if "|" in line:
            in_table = True
            table_lines.append(line)
        else:
            if in_table:
                break  # first non-pipe line after table starts → done

    if len(table_lines) < 2:
        return None

    headers: List[str] = []
    rows: List[Dict[str, str]] = []
    header_parsed = False

    for i, line in enumerate(table_lines):
        # Skip separator rows like |---|---|
        if _SEPARATOR_RE.match(line):
            continue

        cells = _split_pipe_row(line)

        if not header_parsed:
            headers = cells
            header_parsed = True
            continue

        # Data row
        if headers:
            row: Dict[str, str] = {}
            for j, h in enumerate(headers):
                row[h] = cells[j] if j < len(cells) else ""
        else:
            row = {f"col_{j}": v for j, v in enumerate(cells)}

        if any(v for v in row.values()):
            rows.append(row)

    if not rows:
        return None

    return headers, rows


# ---------------------------------------------------------------------------
# Raw text fallback
# ---------------------------------------------------------------------------

def _parse_raw_fallback(block_id: str, text: str) -> ParsedTable:
    """Last resort: wrap raw text in a single-row dict."""
    return ParsedTable(
        block_id=block_id,
        headers=["raw_text"],
        rows=[{"raw_text": text}] if text else [],
        parse_method="raw",
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_table_block(block: RawBlock) -> ParsedTable:
    """Parse a single table RawBlock into a ParsedTable.

    Tries HTML first, then markdown, then raw fallback.

    The text is sourced preferentially from:
      1. ``block.raw["block_content"]`` — the verbatim PP-StructureV3 entry,
         which often contains richer HTML than the cleaned ``block.text``
      2. ``block.text`` — cleaned/normalized text from Stage 1
    """
    raw_content: str = ""
    if isinstance(block.raw, dict):
        raw_content = (block.raw.get("block_content") or "").strip()

    # Primary: try block_content (may have full HTML table)
    if raw_content:
        result = _parse_html_table(raw_content)
        if result:
            headers, rows = result
            return ParsedTable(
                block_id=block.block_id,
                headers=headers,
                rows=rows,
                parse_method="html",
                raw_text=raw_content,
            )

        result = _parse_markdown_table(raw_content)
        if result:
            headers, rows = result
            return ParsedTable(
                block_id=block.block_id,
                headers=headers,
                rows=rows,
                parse_method="markdown",
                raw_text=raw_content,
            )

    # Secondary: try block.text
    if block.text and block.text != raw_content:
        result = _parse_html_table(block.text)
        if result:
            headers, rows = result
            return ParsedTable(
                block_id=block.block_id,
                headers=headers,
                rows=rows,
                parse_method="html",
                raw_text=block.text,
            )

        result = _parse_markdown_table(block.text)
        if result:
            headers, rows = result
            return ParsedTable(
                block_id=block.block_id,
                headers=headers,
                rows=rows,
                parse_method="markdown",
                raw_text=block.text,
            )

    # Fallback
    return _parse_raw_fallback(block.block_id, raw_content or block.text)


def parse_table_blocks(blocks: List[RawBlock]) -> List[ParsedTable]:
    """Parse a list of table RawBlocks.  Non-table blocks are silently skipped."""
    results: List[ParsedTable] = []
    for block in blocks:
        if block.block_type != "table":
            continue
        results.append(parse_table_block(block))
    return results
