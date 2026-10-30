"""Check identifiers against source evidence; retain ambiguous records for review."""
import json
import re
from catalogbank_ocr.stage2.table_products import row_codes, spillover_row_codes


def _heading_key(text):
    words = re.findall(r'\w+', text.casefold())
    return tuple(sorted(word.rstrip('s') if len(word) > 3 else word for word in words))


def _is_heading_name(name, headings):
    stop = {'the', 'of', 'for', 'and', 'page', 'section', 'overview'}
    tokens = set(_heading_key(name)) - stop
    return bool(tokens) and any(tokens <= (set(_heading_key(heading)) - stop) for heading in headings)


def validate_evidence(results, inputs):
    issues = []
    by_id = {v['chunk_id']: v for v in inputs}
    known = {v for r in results for p in r.products for v in (p.name, p.sku) if v}
    model_categories = {}
    alias_owners = {}
    sku_models = {}
    for result in results:
        for product in result.products:
            identity = product.sku or product.model or product.name
            for alias in (product.name, product.sku):
                if alias: alias_owners.setdefault(alias, set()).add(identity)
    for result in results:
        request = by_id[result.chunk_id]
        context = request['context']
        records = context.get('evidence', [])
        text = json.dumps(request.get('section_path', []), ensure_ascii=False) + '\n' + '\n'.join(json.dumps(v, ensure_ascii=False) for v in records)
        headings = set(request.get('section_path', []))
        extracted_skus = {p.sku for p in result.products if p.sku}
        for record in records:
            for value in row_codes(record.get('row', {})):
                if value not in extracted_skus:
                    issues.append({'severity': 'error', 'code': 'MISSING_TABLE_SKU', 'chunk': result.chunk_id, 'value': value})
            for value in spillover_row_codes(record.get('row', {})):
                if value not in extracted_skus:
                    issues.append({'severity': 'error', 'code': 'MISSING_TABLE_SKU', 'chunk': result.chunk_id, 'value': value})
                elif not any(a.entity == value and a.key.casefold() == 'price' for a in result.attributes):
                    issues.append({'severity': 'warning', 'code': 'TABLE_SKU_PRICE_NOT_RECOVERED',
                                   'chunk': result.chunk_id, 'sku': value})
        for product in result.products:
            product.source_evidence = [dict(page=page, block_id=block) for page, block in
                                       dict.fromkeys((r['page'], r['block_id']) for r in records)]
            if product.model and re.fullmatch(r'(?i)(?:\d{3,4}\s+)?(?:stainless steel|steel|brass|aluminum|aluminium|bronze|copper|plastic)', product.model.strip()):
                issues.append({'severity': 'info', 'code': 'MATERIAL_IS_NOT_MODEL',
                               'chunk': result.chunk_id, 'value': product.model})
                product.model = None
            if product.model and product.model == product.sku:
                explicit_model = any(str(record.get('row', {}).get('Model', record.get('row', {}).get('MODEL', ''))) == product.model for record in records)
                explicit_model |= bool(re.search(r'(?i)model\s*(?:number|no\.?|#)?\s*[:=]?\s*' + re.escape(product.model) + r'(?!\w)', text))
                if not explicit_model:
                    # A literal item/part code proves the SKU, not the model role.
                    product.model = None
                    issues.append({'severity': 'info', 'code': 'SKU_IS_NOT_A_CONFIRMED_MODEL',
                                   'chunk': result.chunk_id, 'sku': product.sku})
            concrete_item = any(
                any(key.casefold() in ('description', 'product name', 'name') and value.casefold() == product.name.casefold()
                    for key, value in record.get('row', {}).items())
                and any(key.casefold() in ('price', 'list price', 'usd', 'eur', 'gbp', '$', '€', '£')
                        and re.search(r'\d', value) for key, value in record.get('row', {}).items())
                for record in records)
            if _is_heading_name(product.name, headings) and not (product.sku or product.model or concrete_item):
                issues.append({'severity': 'error', 'code': 'HEADING_AS_PRODUCT', 'chunk': result.chunk_id, 'name': product.name})
            elif not (product.sku or product.model or concrete_item):
                issues.append({'severity': 'error', 'code': 'UNSUPPORTED_SELLABLE_ITEM',
                               'chunk': result.chunk_id, 'name': product.name})
            for field in ('sku', 'model'):
                code = getattr(product, field)
                if code and not re.search(r'(?<![\w])' + re.escape(code) + r'(?![\w])', text):
                    issues.append({'severity': 'error', 'code': 'UNSUPPORTED_IDENTIFIER', 'chunk': result.chunk_id, 'field': field, 'value': code})
            if not product.category and headings:
                product.category = ' > '.join(request['section_path'])
            if product.model:
                model_categories.setdefault(product.model, set()).add(product.category)
                if product.sku: sku_models.setdefault(product.sku, set()).add(product.model)
        for attr in result.attributes:
            if attr.entity not in known:
                issues.append({'severity': 'error', 'code': 'ORPHANED_ATTRIBUTE', 'chunk': result.chunk_id, 'entity': attr.entity})
            elif len(alias_owners.get(attr.entity, set())) > 1:
                issues.append({'severity': 'error', 'code': 'AMBIGUOUS_ATTRIBUTE_OWNER', 'chunk': result.chunk_id,
                               'entity': attr.entity, 'action': 'Use the exact SKU for variant attributes.'})
    for sku, models in sku_models.items():
        if len(models) > 1:
            issues.append({'severity': 'error', 'code': 'CONFLICTING_SKU_MODEL', 'sku': sku, 'models': sorted(models)})
    for model, categories in model_categories.items():
        if len(categories) > 1:
            issues.append({'severity': 'warning', 'code': 'MULTIPLE_MODEL_CATEGORIES', 'model': model,
                           'categories': sorted(str(c) for c in categories),
                           'action': 'Export once in its first observed category; review the alternative placements.'})
    return issues


def remove_heading_only_products(results, inputs):
    """Discard proven family headings that the LLM emitted as sellable products."""
    rejected = {(issue['chunk'], issue['name'], issue['code']) for issue in validate_evidence(results, inputs)
                if issue['code'] in {'HEADING_AS_PRODUCT', 'UNSUPPORTED_SELLABLE_ITEM'}}
    notes = []
    for result in results:
        removed = {name for chunk,name,_ in rejected if chunk == result.chunk_id}
        if not removed: continue
        result.products = [product for product in result.products if product.name not in removed or product.sku or product.model]
        remaining_aliases = {alias for product in result.products for alias in (product.name, product.sku) if alias}
        result.attributes = [attribute for attribute in result.attributes
                             if attribute.entity not in removed or attribute.entity in remaining_aliases]
        notes.extend({'severity':'info','code':'UNSUPPORTED_ITEM_IGNORED' if any(chunk == result.chunk_id and rejected_name == name and code == 'UNSUPPORTED_SELLABLE_ITEM' for chunk,rejected_name,code in rejected) else 'HEADING_FAMILY_IGNORED',
                      'chunk':result.chunk_id,'name':name,
                      'action':'Family text retained in source evidence; no sellable item created.'}
                     for name in sorted(removed))
    return notes


def repair_extractions(extractor, results, inputs, attempts=1):
    """Retry only invalid chunks, keeping failed attempts for audit."""
    history = []
    by_id = {v['chunk_id']: v for v in inputs}
    for _ in range(attempts):
        issues = validate_evidence(results, inputs)
        bad = {v['chunk'] for v in issues if v['severity'] == 'error' and v.get('chunk')}
        bad.update(r.chunk_id for r in results if r.parse_error)
        if not bad: break
        for i, result in enumerate(results):
            if result.chunk_id not in bad: continue
            history.append(result.to_dict())
            request = dict(by_id[result.chunk_id])
            problems = [v for v in issues if v.get('chunk') == result.chunk_id]
            request['prompt'] += '\nCORRECTION REQUIRED: ' + json.dumps(problems) + '\nRe-extract from the original evidence. Return empty arrays for headings or family descriptions without concrete items. Never invent identifiers or remove legitimate supported items.'
            results[i] = extractor.extract(request)
    return results, history
