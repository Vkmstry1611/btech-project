"""Visualize semantic blocks and reconstructed hierarchies."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from catalogbank_ocr.schemas.hierarchy import HierarchyNode
from catalogbank_ocr.schemas.semantic_block import PageSemanticBlocks, SemanticBlock

logger = logging.getLogger(__name__)

SEMANTIC_COLORS = {
    "heading": (255, 87, 34),
    "table": (33, 150, 243),
    "figure": (156, 39, 176),
    "product_card": (76, 175, 80),
    "specification": (255, 193, 7),
    "paragraph": (96, 125, 139),
    "document": (0, 0, 0),
}


def _load_font(size: int = 14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _bbox_is_valid(bbox) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)


def visualize_semantic_blocks(
    source_image_path: Path,
    page_blocks: PageSemanticBlocks,
    output_path: Path,
    show_labels: bool = True,
) -> Path:
    """Overlay semantic blocks on the source image and save the result."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image_path) as image:
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = _load_font(14)

        for block in page_blocks.blocks:
            if not _bbox_is_valid(block.bbox):
                continue
            color = SEMANTIC_COLORS.get(block.type, (0, 0, 0))
            bbox = [float(value) for value in block.bbox]  # type: ignore[arg-type]
            draw.rectangle(bbox, outline=color, width=3)
            label = f"{block.type} #{block.order}"
            if show_labels and block.text:
                short_text = block.text.replace("\n", " ")[:60]
                label = f"{label}: {short_text}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            x1 = bbox[0]
            y1 = max(0, bbox[1] - text_h - 4)
            draw.rectangle([x1, y1, x1 + text_w + 8, y1 + text_h + 4], fill=color)
            draw.text((x1 + 4, y1 + 2), label, fill=(255, 255, 255), font=font)

        canvas.save(output_path)
    logger.info("Saved semantic block visualization to %s", output_path)
    return output_path


def _tree_metrics(root: HierarchyNode) -> Tuple[Dict[int, int], int]:
    levels = defaultdict(int)

    def walk(node: HierarchyNode, depth: int) -> None:
        levels[depth] += 1
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    max_depth = max(levels.keys()) if levels else 0
    return levels, max_depth


def _assign_positions(node: HierarchyNode, depth: int, x_cursor: List[int], positions: Dict[int, Tuple[float, float]], node_order: Dict[int, HierarchyNode]) -> float:
    node_id = id(node)
    node_order[node_id] = node
    if not node.children:
        x = float(x_cursor[0])
        x_cursor[0] += 1
        positions[node_id] = (x, float(-depth))
        return x

    child_xs = []
    for child in node.children:
        child_xs.append(_assign_positions(child, depth + 1, x_cursor, positions, node_order))
    x = sum(child_xs) / float(len(child_xs))
    positions[node_id] = (x, float(-depth))
    return x


def visualize_hierarchy_tree(root: HierarchyNode, output_path: Path) -> Path:
    """Render a hierarchy tree visualization as a simple matplotlib diagram."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for hierarchy visualization") from exc

    positions: Dict[int, Tuple[float, float]] = {}
    node_order: Dict[int, HierarchyNode] = {}
    _assign_positions(root, 0, [0], positions, node_order)

    # Cap figure width to avoid matplotlib's 65535-pixel limit at 200 DPI.
    # A sensible upper bound is ~40 inches (8000 px at 200 DPI).
    node_count = len(positions)
    fig_width = min(40.0, max(8.0, node_count * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    ax.axis("off")

    def draw_node(node: HierarchyNode) -> None:
        node_id = id(node)
        x, y = positions[node_id]
        label = node.name
        if node.semantic_type and node.semantic_type != "document":
            label = f"{label}\n[{node.semantic_type}]"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.2),
            fontsize=9,
        )
        for child in node.children:
            cx, cy = positions[id(child)]
            ax.plot([x, cx], [y - 0.1, cy + 0.1], color="black", linewidth=1)
            draw_node(child)

    draw_node(root)
    ax.relim()
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hierarchy visualization to %s", output_path)
    return output_path
