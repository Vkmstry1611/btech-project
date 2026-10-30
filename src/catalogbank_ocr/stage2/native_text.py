"""Source-backed SKU/price pairs from a digital PDF's embedded text layer.

This supplements OCR when dense printed tables fuse adjacent numbers. It never
guesses a price from a neighboring item or treats an unpaired reference as a SKU.
"""

from __future__ import annotations

import re
import json
import hashlib
from pathlib import Path


NATIVE_SKU = re.compile(r'(?<![A-Z0-9])\d{4,5}[A-Z]\d{2,5}(?![A-Z0-9])')
NATIVE_PRICE = re.compile(r'(?<![\d.])(?P<currency>[$€£])?\s*(?P<amount>\d{1,5}(?:,\d{3})*\.\d{2})(?![\d.])')
ALPHA_SKU_PRICE_ROW = re.compile(
    r'(?m)(?P<codes>(?:^[A-Z]{2,6}\d{2,6}(?:/[A-Z0-9]{1,4})?[ \t]*\n){1,3})'
    r'^\$[ \t]*\n(?P<amount>\d{1,3}(?:,\d{3})*\.\d{2})')


def paired_native_codes(pdf_path: Path):
    """Yield exact codes with an adjacent dotted-leader price from PDF text.

    The code and price must occur before the next code and within 120 source
    characters. This intentionally ignores unpriced mentions and scanned PDFs.
    """
    import pymupdf

    pairs = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, 1):
            pairs.extend(paired_codes_in_text(page.get_text(), page_number))
    return pairs


def paired_codes_in_text(text: str, page_number: int):
    pairs = []
    matches = list(NATIVE_SKU.finditer(text))
    for index, match in enumerate(matches):
        end = min(match.end() + 120,
                  matches[index + 1].start() if index + 1 < len(matches) else len(text))
        tail = text[match.end():end]
        price = NATIVE_PRICE.search(tail)
        if not price or '.' not in tail[:price.start()]:
            continue
        pairs.append({'page': page_number, 'sku': match.group(),
                      'price': price.group('amount'),
                      'currency': {'$':'USD', '€':'EUR', '£':'GBP'}.get(price.group('currency'))})
    return pairs


def supplement_missing_skus(pdf_path, results, inputs):
    """Add exact source-text SKU/price pairs missing after OCR extraction."""
    from catalogbank_ocr.stage2.llm_extractor import ExtractedAttribute, ExtractedProduct, ExtractionResult

    pairs = paired_native_codes(pdf_path)
    existing = {product.sku for result in results for product in result.products if product.sku}
    evidence = []
    for request in inputs:
        records = request['context'].get('evidence', [])
        pages = {record['page'] for record in records}
        content = json.dumps(records, ensure_ascii=False)
        for page in pages:
            evidence.append((page, request.get('section_path', []), content))
    grouped = {}
    for pair in pairs:
        code = pair['sku']
        if code in existing:
            continue
        page = pair['page']
        matches = [path for source_page, path, content in evidence if source_page == page and code in content]
        fallback = next((path for source_page, path, _ in evidence if source_page == page and path), [])
        path = matches[0] if matches else fallback
        key = (page, tuple(path))
        grouped.setdefault(key, []).append(pair)
        existing.add(code)
    added = []
    for (page, path), items in grouped.items():
        chunk_id = f'{pdf_path.stem}__native_p{page}__g{len(added)+1}'
        result = ExtractionResult(chunk_id, pdf_path.stem, list(path))
        result.backend_used = 'deterministic/native-text'
        records = []
        family_group = hashlib.sha256(json.dumps([page, path]).encode()).hexdigest()[:16]
        for item in items:
            code = item['sku']
            result.products.append(ExtractedProduct(name=path[-1] if path else code, sku=code,
                category=' > '.join(path) or None, variant_group=family_group))
            result.attributes.append(ExtractedAttribute(code, 'Price', item['price'], item['currency']))
            records.append({'page': page, 'block_id': f'native_p{page}_{code}',
                            'kind': 'native_text_price',
                            'row': {'SKU': code, 'Price': item['price']}})
        inputs.append({'chunk_id': chunk_id, 'doc_stem': pdf_path.stem,
                       'section_path': list(path), 'context': {'evidence': records}})
        results.append(result)
        added.extend(items)
    return added, {pair['sku'] for pair in pairs}


def recover_table_fragment_errors(results, inputs, native_skus):
    """Mark malformed responses recovered when source prices cover their SKUs."""
    by_id = {request['chunk_id']: request for request in inputs}
    notes = []
    for result in results:
        try:
            response = json.loads(result.raw_response) if result.raw_response else None
        except (ValueError, TypeError):
            response = None
        missing_arrays = isinstance(response, dict) and not any(
            key in response for key in ('products', 'attributes', 'relations'))
        if not result.parse_error and not missing_arrays:
            continue
        records = by_id[result.chunk_id]['context'].get('evidence', [])
        if not records or any('row' not in record for record in records):
            continue
        codes = set(NATIVE_SKU.findall(json.dumps(records, ensure_ascii=False)))
        nonempty = [value for record in records for value in record['row'].values() if str(value).strip()]
        only_orphan_prices = not codes and all(re.fullmatch(r'[$€£]?\s*\d+(?:,\d{3})*\.\d{2}', str(v).strip()) for v in nonempty)
        if not codes.issubset(native_skus) or (nonempty and not codes and not only_orphan_prices):
            continue
        notes.append({'severity': 'warning', 'code': 'ORPHANED_PRICE_FRAGMENT' if only_orphan_prices else 'TABLE_FRAGMENT_RECOVERED_FROM_SOURCE',
                      'chunk': result.chunk_id, 'original_error': result.parse_error or 'Model response omitted extraction arrays',
                      'action': 'Source text supplies exact SKU/price pairs; inspect other table properties.'})
        result.parse_error = None
    return notes


def recover_existing_alpha_prices(pdf_path, results, inputs):
    """Fill an existing SKU's missing USD price from an exact source-text row."""
    import pymupdf
    from catalogbank_ocr.stage2.llm_extractor import ExtractedAttribute

    owners = {product.sku: result for result in results for product in result.products if product.sku}
    priced = {attribute.entity for result in results for attribute in result.attributes
              if attribute.key.casefold() == 'price'}
    by_id = {request['chunk_id']: request for request in inputs}
    notes = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, 1):
            for match in ALPHA_SKU_PRICE_ROW.finditer(page.get_text()):
                codes = [line.strip() for line in match.group('codes').splitlines()]
                amount = match.group('amount').replace(',', '')
                for code in codes:
                    if code not in owners or code in priced:
                        continue
                    result = owners[code]
                    result.attributes.append(ExtractedAttribute(code, 'Price', amount, 'USD'))
                    by_id[result.chunk_id]['context']['evidence'].append({
                        'page': page_number, 'block_id': f'native_p{page_number}_{code}_price',
                        'kind': 'native_text_price', 'row': {'SKU': code, '$': '$' + match.group('amount')}})
                    notes.append({'severity': 'info', 'code': 'PRICE_RECOVERED_FROM_SOURCE_TEXT',
                                  'chunk': result.chunk_id, 'sku': code, 'page': page_number})
                    priced.add(code)
    return notes
