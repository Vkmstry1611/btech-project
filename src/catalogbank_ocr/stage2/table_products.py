"""Reconcile explicit table identifiers without guessing model codes."""
import hashlib
import json
import re

from catalogbank_ocr.stage2.llm_extractor import ExtractedProduct, ExtractedAttribute, ExtractionResult

from catalogbank_ocr.stage2.identifiers import SKU_HEADER, sku_columns, explicit_heading_model
CODE = re.compile(r'[A-Za-z0-9][A-Za-z0-9./_ -]*')
INLINE_PART = re.compile(r'(?<![A-Za-z0-9])(\d{4,5}[A-Z]\d{2,5})(?![A-Za-z0-9])')
INLINE_PRICE = re.compile(r'^\s*([$€£])?\s*(\d+(?:,\d{3})*\.\d{2})(?![\d.])')
SPILLOVER_SKU = re.compile(r'[A-Z]{2,6}\d{2,6}(?:/[A-Z0-9]{1,4})?')


def row_codes(row):
    """Read SKU columns and explicit slash variants displaced by OCR cells."""
    row = realign_marker_row(row)
    columns = sku_columns(row)
    codes = list(dict.fromkeys(v.strip() for k, v in row.items() if k in columns
                              and CODE.fullmatch(v) and not SKU_HEADER.fullmatch(v.strip())
                              and v.strip().lower() not in {'n/a', 'none', 'na', 'new', 'see', 'call', 'contact', 'each'}))
    for value in row.values():
        if any(re.fullmatch(re.escape(code) + r'/[A-Za-z0-9]{1,4}', value) for code in codes):
            if value not in codes: codes.append(value)
    return codes


def spillover_row_codes(row):
    """Find exact SKUs displaced into anonymous OCR columns, without row prices."""
    anchors = row_codes(row)
    if not anchors:
        return []
    prefixes = {re.match(r'[A-Z]+', code).group() for code in anchors
                if re.match(r'[A-Z]+', code)}
    return list(dict.fromkeys(value.strip() for key, value in row.items()
        if re.fullmatch(r'col_\d+', key) and SPILLOVER_SKU.fullmatch(value.strip())
        and re.match(r'[A-Z]+', value.strip()).group() in prefixes
        and value.strip() not in anchors))


def realign_marker_row(row):
    """Remove a leading NEW badge only when it visibly shifts every table cell."""
    cells = list(row.items())
    if (len(cells) >= 4 and SKU_HEADER.fullmatch(cells[0][0].strip())
        and cells[0][1].strip().upper() == 'NEW'
        and cells[1][0].strip() in {'$', '€', '£'}
        and re.fullmatch(r'[A-Z]{2,6}\d{2,6}(?:/[A-Z0-9]{1,4})?', cells[1][1].strip())
        and cells[2][1].strip().startswith(cells[1][0].strip())):
        return dict(zip((key for key, _ in cells[:-1]), (value for _, value in cells[1:])))
    return row


def inline_part_prices(row):
    """Read literal code/price pairs from compressed table cells.

    Price association is accepted only within the cell immediately after a code,
    or in the next cell when that cell contains just the price. Other cells remain
    semantic evidence; their column alignment is not inferred.
    """
    values = list(row.values())
    pairs = []
    for index, value in enumerate(values):
        matches = list(INLINE_PART.finditer(value))
        for number, match in enumerate(matches):
            following = value[match.end():matches[number + 1].start() if number + 1 < len(matches) else len(value)]
            price = INLINE_PRICE.fullmatch(following.strip())
            if not price and not following.strip() and number + 1 == len(matches) and index + 1 < len(values):
                price = INLINE_PRICE.fullmatch(values[index + 1].strip())
            if price:
                pairs.append((match.group(1), price.group(2), {'€':'EUR','£':'GBP'}.get(price.group(1))))
    return list(dict.fromkeys(pairs))


def extract_inline_table_row(request, record):
    """Extract proven part codes and adjacent prices without assigning misaligned headers."""
    row = record.get('row')
    if not row: return None
    selected = sku_columns(row)
    # If a SKU column exists, a code in another column may be the manufacturer's
    # reference. Fall back to semantic review unless the code is in that column.
    searchable = {key:row[key] for key in selected} if selected else row
    pairs = inline_part_prices(searchable)
    if not pairs: return None
    result = ExtractionResult(request['chunk_id'], request['doc_stem'], request.get('section_path', []))
    name = next((v for k,v in row.items() if k.strip().lower() in ('description','product name') and v), None)
    name = name or (request.get('section_path') or ['Catalog item'])[-1]
    group = hashlib.sha256(json.dumps([record['block_id'], row], sort_keys=True).encode()).hexdigest()[:16]
    model = explicit_heading_model(request.get('section_path', []))
    for code, amount, currency in pairs:
        result.products.append(ExtractedProduct(name=name, model=model, sku=code,
            description=name if name != 'Catalog item' else None,
            category=' > '.join(result.section_path) or None, variant_group=group))
        result.attributes.append(ExtractedAttribute(code, 'Price', amount, currency))
    result.backend_used = 'deterministic/inline-table'
    return result


def reconcile_table_products(results, inputs):
    by_id = {request['chunk_id']: request for request in inputs}
    for result in results:
        request = by_id[result.chunk_id]
        for record in request['context'].get('evidence', []):
            row = record.get('row', {})
            row = realign_marker_row(row)
            codes = row_codes(row) + spillover_row_codes(row)
            if not codes: continue
            group = hashlib.sha256(json.dumps([record['block_id'], row], sort_keys=True).encode()).hexdigest()[:16]
            existing = [p for p in result.products if p.sku in codes]
            description = next((v for k, v in row.items() if k.strip().lower() in ('description', 'product name')), None)
            model = next((v for k, v in row.items() if k.strip().lower() == 'model' and v), None) or explicit_heading_model(request.get('section_path', []))
            for code in codes:
                product = next((p for p in existing if p.sku == code), None)
                if product is None:
                    product = ExtractedProduct(name=description or code, model=model, sku=code,
                        description=description, category=' > '.join(request.get('section_path', [])) or None)
                    result.products.append(product)
                product.variant_group = group
                if model: product.model = model
                if not product.description and description: product.description = description
                # Attributes keyed by one exact SKU describe that variant only.
                # Shared table-row values remain evidence for semantic extraction;
                # they are not copied across variants here.
    return results


def extract_explicit_table(request):
    """Read unambiguous product rows directly; otherwise defer the whole request.

    This preserves all row fields and avoids LLM decoding for literal table data.
    Mixed text or missing SKU headers use the normal extraction path. Ambiguous
    price cells are retained verbatim and flagged during export, never guessed.
    """
    from catalogbank_ocr.stage2.catalog_export import parse_price
    records = request['context'].get('evidence', [])
    if not records: return None
    currencies = {'usd':'USD', 'eur':'EUR', '€':'EUR', 'gbp':'GBP', '£':'GBP',
                  'rmb':'CNY', 'cny':'CNY', 'jpy':'JPY', 'inr':'INR', 'aud':'AUD'}
    result = ExtractionResult(request['chunk_id'], request['doc_stem'], request.get('section_path', []))
    for record in records:
        row = record.get('row')
        if not row: return None
        row = realign_marker_row(row)
        codes = row_codes(row)
        if not codes: return extract_inline_table_row(request, record) if len(records) == 1 else None
        description = next((v for k,v in row.items() if k.lower().strip() in ('description','product name')), None)
        model = next((v for k,v in row.items() if k.lower().strip() == 'model'), None) or explicit_heading_model(request.get('section_path', []))
        properties, prices = [], []
        for key, value in row.items():
            if not value or key in sku_columns(row) or key.lower().strip() in ('description','product name','model'):
                continue
            normalized_key = key.lower().strip()
            if normalized_key in currencies or normalized_key in ('$', '¥', 'price', 'list price'):
                try: amount = str(parse_price(value))
                except ValueError: amount = value
                unit = currencies.get(normalized_key)
                if unit is None:
                    prefix = re.match(r'^(USD|EUR|GBP|JPY|CNY|INR|AUD)\b', value, re.I)
                    if prefix: unit = prefix.group(1).upper()
                prices.append((amount, unit))
            else:
                properties.append((key, value))
        for code in codes:
            result.products.append(ExtractedProduct(name=description or code, model=model, sku=code,
                description=description, category=' > '.join(result.section_path) or None))
            for key, value in properties:
                result.attributes.append(ExtractedAttribute(code, key, value))
            for value, unit in prices:
                result.attributes.append(ExtractedAttribute(code, 'Price', value, unit))
    result.backend_used = 'deterministic/explicit-table'
    return result


def extract_with_tables(extractor, requests, progress=False):
    import time
    results = []
    for index, request in enumerate(requests, 1):
        start = time.monotonic()
        literal_products, literal_attributes, remaining = [], [], []
        for record in request['context'].get('evidence', []):
            single = dict(request, context=dict(request['context'], evidence=[record]))
            direct = extract_explicit_table(single)
            if direct is None:
                remaining.append(record)
            else:
                literal_products.extend(direct.products)
                literal_attributes.extend(direct.attributes)
        if remaining:
            context = dict(request['context'], evidence=remaining)
            if literal_products:
                context['product_anchors'] = [{'name':p.name, 'sku':p.sku} for p in literal_products]
            reduced = dict(request, context=context)
            reduced['prompt'] = request['prompt'].split('\nEVIDENCE:\n')[0] + '\nEVIDENCE:\n' + json.dumps(context, ensure_ascii=False)
            result = extractor.extract(reduced)
            result.products = literal_products + result.products
            result.attributes = literal_attributes + result.attributes
            if literal_products: result.backend_used = 'hybrid/' + result.backend_used
        else:
            result = ExtractionResult(request['chunk_id'], request['doc_stem'], request.get('section_path', []))
            result.products, result.attributes = literal_products, literal_attributes
            result.backend_used = 'deterministic/explicit-table'
        result.latency_s = time.monotonic() - start
        results.append(result)
        if progress:
            print(f'[{index}/{len(requests)}] {request["chunk_id"]}: {result.backend_used}, '
                  f'{len(result.products)} products, {result.latency_s:.1f}s', flush=True)
    return results
