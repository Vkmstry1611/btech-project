"""Explicit table identifier roles; distributor and maker codes are not variants."""
import re

SKU_HEADER = re.compile(
    r'(?:metric\s+|imperial\s+)?(?:item\s*(?:#|number|no\.?)\s*\*?|sku|'
    r'part\s*(?:number|no\.?|#)|article\s*(?:number|no\.?|#)|'
    r'stock\s*(?:number|no\.?|#)|order\s*codes?|catalog(?:ue)?\s*(?:number|no\.?|#))', re.I)
SUPPLIER_HEADER = re.compile(r'(?:stock\s*(?:number|no\.?|#)|order\s*codes?)', re.I)


def sku_columns(row):
    """Prefer an explicit distributor SKU over a separate manufacturer's part code."""
    supplier = [key for key in row if SUPPLIER_HEADER.fullmatch(key.strip())]
    return supplier or [key for key in row if SKU_HEADER.fullmatch(key.strip())]


def explicit_heading_model(path):
    """Use a heading only when it explicitly labels a single model code."""
    for heading in reversed(path):
        match = re.fullmatch(r'model\s*(?:(?:number|no\.?|#)\s*)?[:=]?\s+([A-Za-z0-9][A-Za-z0-9./_-]*)', heading.strip(), re.I)
        if match: return match.group(1)
    return None
