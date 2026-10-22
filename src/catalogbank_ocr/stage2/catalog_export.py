"""Deterministic sample-schema export; document headings are never products."""
import hashlib
import math
import re
from pathlib import Path


def identifier(prefix, *parts):
    return prefix + '-' + hashlib.sha256('\x1f'.join(str(p) for p in parts).encode()).hexdigest()[:16]


def parse_price(value):
    """Accept one explicit numeric price; reject mixed cells and ambiguous grouping."""
    text = str(value).strip()
    text = re.sub(r'^(?:USD|EUR|GBP|JPY|CNY|INR|AUD|[$€£¥￥])\s*', '', text, flags=re.I)
    text = re.sub(r'\s*(?:USD|EUR|GBP|JPY|CNY|INR|AUD)$', '', text, flags=re.I)
    if not re.fullmatch(r'\d+(?:[.,]\d+)*', text):
        raise ValueError(f'Ambiguous price: {value!r}')
    if ',' in text and '.' in text:
        decimal, grouping = (',', '.') if text.rfind(',') > text.rfind('.') else ('.', ',')
        integer, separator, fraction = text.rpartition(decimal)
        if text.count(decimal) != 1 or not fraction.isdigit() or not re.fullmatch(r'\d{1,3}(?:' + re.escape(grouping) + r'\d{3})+', integer):
            raise ValueError(f'Ambiguous price: {value!r}')
        text = text.replace(grouping, '').replace(decimal, '.')
    elif ',' in text:
        if re.fullmatch(r'\d{1,3}(?:,\d{3})+', text): text = text.replace(',', '')
        elif text.count(',') == 1 and len(text.rsplit(',', 1)[1]) in (1, 2): text = text.replace(',', '.')
        else: raise ValueError(f'Ambiguous price: {value!r}')
    amount = float(text)
    if not math.isfinite(amount) or amount < 0: raise ValueError('Invalid price')
    return amount


def build_catalog_json(document, normalized, backend_used='', pdf_path='', review_issues=None):
    title = document.get('catalog_metadata', {}).get('title') or Path(pdf_path or normalized.doc_stem or 'Document').stem
    root = {'type': 'Product Category', 'title': title, 'identifier': identifier('PC', title)}
    categories, models, items = {}, {}, {}
    for product in normalized.products:
        p = product.to_dict()
        # Group only explicit models; unidentified models are not inferred from headings.
        if p.get('model'):
            model_key = ('model', p['model'])
        elif p.get('variant_group'):
            model_key = ('table-row', p['variant_group'])
        elif p.get('sku'):
            model_key = ('unconfirmed-model', p['sku'])
        else:
            model_key = ('name', p['name'])
        category = p.get('category') or 'Unclassified Products'
        path = tuple(v.strip() for v in category.split(' > ') if v.strip())
        parent = root
        for depth, name in enumerate(path):
            key = path[:depth + 1]
            if key not in categories:
                categories[key] = {'type': 'Product Category', 'title': name, 'identifier': identifier('PC', title, *key)}
                parent.setdefault('subcategories', []).append(categories[key])
            parent = categories[key]
        # A model is placed once globally. Conflicting categories are review issues upstream.
        if model_key not in models:
            models[model_key] = {'type': 'Model', 'title': p.get('model') or p['name'], 'identifier': identifier('MOD', title, model_key), 'sellableItems': []}
            parent.setdefault('models', []).append(models[model_key])
        model = models[model_key]
        if p.get('description'): model.setdefault('description', p['description'])
        item_key = (model_key, p.get('sku') or p['name'])
        if item_key in items: continue
        item = {'type': 'Sellable Item', 'title': p['name'], 'identifier': identifier('SI', title, *item_key)}
        if p.get('sku'): item['articleNumber'] = p['sku']
        if p.get('description'): item['description'] = p['description']
        aliases = {p['name'], p.get('sku')}
        for a in normalized.attributes:
            if a.product_sku:
                if a.product_sku != p.get('sku'): continue
            elif a.entity not in aliases:
                continue
            if 'price' in a.key.lower():
                try:
                    amount = parse_price(a.value)
                except ValueError:
                    if review_issues is None: raise
                    review_issues.append({'severity': 'warning', 'code': 'AMBIGUOUS_PRICE',
                                          'entity': p.get('sku') or p['name'], 'value': a.value,
                                          'unit': a.unit, 'source_chunk': a.source_chunk,
                                          'action': 'Price omitted from final prices; original retained in normalized.json.'})
                    continue
                price = {'type': 'List Price', 'identifier': identifier('LP', item['identifier'], a.value, a.unit or '', a.location or ''), 'title': p['name'], 'amount': amount}
                if a.unit and len(a.unit) == 3 and a.unit.isalpha(): price['currency'] = a.unit.upper()
                if a.location: price['location'] = a.location
                prices = item.setdefault('listPrices', [])
                if not any(v['identifier'] == price['identifier'] for v in prices): prices.append(price)
            else:
                value = a.value + (' ' + a.unit if a.unit else '')
                props = item.setdefault('properties', {})
                if a.key in props and props[a.key] != value:
                    prior = props[a.key]
                    props[a.key] = list(dict.fromkeys((prior if isinstance(prior, list) else [prior]) + [value]))
                else: props[a.key] = value
        model['sellableItems'].append(item); items[item_key] = item
    validate_catalog(root)
    return root


def validate_catalog(root):
    seen = set()
    child_types = {'subcategories': 'Product Category', 'models': 'Model', 'sellableItems': 'Sellable Item', 'listPrices': 'List Price'}
    allowed = {'Product Category': {'subcategories', 'models'}, 'Model': {'sellableItems'}, 'Sellable Item': {'listPrices'}, 'List Price': set()}
    def visit(node, expected):
        if node.get('type') != expected or not node.get('identifier') or not node.get('title'):
            raise ValueError('Invalid catalog node')
        if node['identifier'] in seen: raise ValueError('Duplicate catalog identifier: ' + node['identifier'])
        seen.add(node['identifier'])
        for key, kind in child_types.items():
            if key in node and key not in allowed[expected]: raise ValueError('Invalid containment: ' + key)
            for child in node.get(key, []):
                # The supplied sample also permits a Model directly in subcategories.
                visit(child, 'Model' if key == 'subcategories' and child.get('type') == 'Model' else kind)
        if expected == 'List Price' and (isinstance(node.get('amount'), bool) or not isinstance(node.get('amount'), (float, int)) or not math.isfinite(node['amount']) or node['amount'] < 0):
            raise ValueError('Invalid price amount')
    visit(root, 'Product Category')
