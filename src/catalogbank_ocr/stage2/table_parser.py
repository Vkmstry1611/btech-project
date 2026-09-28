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

    headers: List[str] = []
    rows: List[Dict[str, str]] = []

    # --- Extract headers from <thead> <th> or first <tr> that contains <th>
    thead = table.find("thead")
    if thead:
        th_tags = thead.find_all("th")
        if th_tags:
            headers = [_clean_cell(th.get_text()) for th in th_tags]
    
    if not headers:
        # Try first row for <th> elements
        first_row = table.find("tr")
        if first_row:
            th_tags = first_row.find_all("th")
            if th_tags:
                headers = [_clean_cell(th.get_text()) for th in th_tags]

    # --- Extract data rows from <tbody> or all <tr>
    tbody = table.find("tbody")
    target = tbody if tbody else table

    for tr in target.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        cell_texts = [_clean_cell(c.get_text()) for c in cells]

        # Skip rows that are pure header rows (all cells are <th> and already captured)
        if all(c.name == "th" for c in cells) and headers:
            continue

        if headers:
            # Align cells to headers; pad with "" if row is shorter
            row: Dict[str, str] = {}
            for i, h in enumerate(headers):
                row[h] = cell_texts[i] if i < len(cell_texts) else ""
        else:
            # No headers detected — use positional keys
            row = {f"col_{i}": v for i, v in enumerate(cell_texts)}

        # Skip completely empty rows
        if any(v for v in row.values()):
            rows.append(row)

    if not rows:
        return None

    return headers, rows


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
