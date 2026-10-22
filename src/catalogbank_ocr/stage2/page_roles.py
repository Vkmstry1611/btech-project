"""Conservative page roles: uncertain pages are always retained for extraction."""
import re


def annotate_pages(document):
    metadata = document.setdefault('catalog_metadata', {})
    metadata['index_entries'] = []
    for page in document.get('pages', []):
        blocks = page.get('blocks', [])
        text = '\n'.join(b.get('text', '') for b in blocks)
        headings = [b.get('text', '').strip() for b in blocks if b.get('type') == 'heading']
        entries = re.findall(r'^(.+?)\s*(?:\.{2,}|\s{3,})\s*(\d+)\s*$', text, re.M)
        index_title = any(re.fullmatch(r'(table of )?contents|index', h, re.I) for h in headings)
        evidence = any(b.get('type') in ('product_card', 'specification') for b in blocks)
        evidence |= bool(re.search(r'\b(SKU|article number|part (?:no|number))\b|[$€£]\s*\d', text, re.I))
        evidence |= any(re.fullmatch(r'[A-Za-z]{1,8}[-/]?\d+[\w/.-]*', h) for h in headings)
        tables = any(b.get('type') == 'table' for b in blocks)
        for block in blocks:
            block_text = block.get('text', '').strip()
            if block.get('source_type') in ('header', 'footer', 'number'):
                block['extraction_role'] = 'page_chrome'
            elif block.get('type') == 'heading' and re.fullmatch(r'(table of )?contents|index|chapters|sections', block_text, re.I):
                block['extraction_role'] = 'navigation'
            elif index_title and re.fullmatch(r'.+?\s*\.{2,}\s*\d+', block_text):
                block['extraction_role'] = 'index_entry'
            else:
                block['extraction_role'] = 'content'
        role, confidence = 'uncertain', 'low'
        if index_title and len(entries) >= 2 and not evidence:
            role, confidence = 'index', 'high'
            metadata['index_entries'].extend({'title': t.strip(), 'printed_page': int(n), 'source_page': page.get('page')} for t, n in entries)
        elif evidence:
            role, confidence = ('mixed' if index_title else 'product'), 'high'
        elif not tables and len(text) < 500 and headings and len(blocks) <= 5:
            if page.get('page') == 1 and re.search(r'catalog(?:ue)?|edition', text, re.I):
                role, confidence = 'cover', 'high'
                metadata.setdefault('title', headings[0])
                metadata['cover_text'] = text
                for key, pattern in (
                    ('manufacturer', r'(?im)^manufacturer\s*[:\-]\s*(.+)$'),
                    ('edition', r'(?im)^edition\s*[:\-]\s*(.+)$'),
                ):
                    match = re.search(pattern, text)
                    if match: metadata[key] = match.group(1).strip()
            elif all(b.get('type') in ('heading', 'figure') or b.get('source_type') in ('header', 'footer', 'number') for b in blocks):
                role, confidence = 'section_divider', 'high'
        page['classification'] = {'role': role, 'confidence': confidence}
    return document
