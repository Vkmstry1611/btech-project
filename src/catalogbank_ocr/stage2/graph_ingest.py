"""Stage 2 — Step 6: Knowledge Graph Ingestion (Neo4j).

Takes a NormalizedResult (from Phase C) and writes it into Neo4j Community
Edition using plain Cypher MERGE statements via the official Python driver.

Graph schema
------------
Nodes
  (:Document  {id, doc_stem})
  (:Section   {id, name, level, page, doc_stem})
  (:Product   {id, name, model, sku, category, doc_stem})
  (:Attribute {id, key, value, unit, value_numeric})
  (:Category  {id, name})

Relationships
  (:Document)   -[:HAS_SECTION]->  (:Section)
  (:Section)    -[:CONTAINS]->     (:Product)
  (:Product)    -[:HAS_ATTRIBUTE]->(:Attribute)
  (:Product)    -[:BELONGS_TO]->   (:Category)
  (:Product)    -[:IS_VARIANT_OF]->(:Product)   # from relation extraction
  (:Product)    -[:HAS_MATERIAL]-> (:Attribute) # or any predicate from LLM

Node IDs are deterministic so repeated ingestion of the same document is safe
(MERGE is idempotent — re-running never creates duplicate nodes).

Design choices for POC
-----------------------
- Uses the synchronous neo4j driver (no async overhead for a batch loader).
- All writes for one NormalizedResult run inside a single write transaction.
- A dry-run mode collects all Cypher statements + parameters into a list and
  returns them without touching Neo4j — used in testing and CI.
- No ORM / OGM layer — just plain parameterised Cypher strings.

Setup (Docker — easiest for local POC)
---------------------------------------
    docker run --name neo4j \\
        -p 7687:7687 -p 7474:7474 \\
        -e NEO4J_AUTH=neo4j/catalogbank \\
        neo4j:5-community

    pip install neo4j>=5.0.0

Then use:
    GraphIngestor(uri="bolt://localhost:7687", user="neo4j", password="catalogbank")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from catalogbank_ocr.stage2.normalization import (
    NormalizedResult,
    NormalizedProduct,
    NormalizedAttribute,
    NormalizedRelation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher templates
# ---------------------------------------------------------------------------

# Constraints / indexes  (run once on first connect if they don't exist)
_CONSTRAINT_STATEMENTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document)  REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT sec_id IF NOT EXISTS FOR (s:Section)   REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT prod_id IF NOT EXISTS FOR (p:Product)  REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT attr_id IF NOT EXISTS FOR (a:Attribute) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT cat_id  IF NOT EXISTS FOR (c:Category) REQUIRE c.id IS UNIQUE",
]

# Node MERGE templates — all parameterised
_MERGE_DOCUMENT = """
MERGE (d:Document {id: $id})
SET d.doc_stem = $doc_stem
"""

_MERGE_SECTION = """
MERGE (s:Section {id: $id})
SET s.name     = $name,
    s.level    = $level,
    s.page     = $page,
    s.doc_stem = $doc_stem
"""

_MERGE_PRODUCT = """
MERGE (p:Product {id: $id})
SET p.name     = $name,
    p.model    = $model,
    p.sku      = $sku,
    p.category = $category,
    p.doc_stem = $doc_stem
"""

_MERGE_ATTRIBUTE = """
MERGE (a:Attribute {id: $id})
SET a.key           = $key,
    a.value         = $value,
    a.unit          = $unit,
    a.value_numeric = $value_numeric
"""

_MERGE_CATEGORY = """
MERGE (c:Category {id: $id})
SET c.name = $name
"""

# Relationship MERGE templates
_MERGE_DOC_SECTION = """
MATCH (d:Document {id: $doc_id})
MATCH (s:Section  {id: $sec_id})
MERGE (d)-[:HAS_SECTION]->(s)
"""

_MERGE_SECTION_PRODUCT = """
MATCH (s:Section {id: $sec_id})
MATCH (p:Product {id: $prod_id})
MERGE (s)-[:CONTAINS]->(p)
"""

_MERGE_PRODUCT_ATTR = """
MATCH (p:Product   {id: $prod_id})
MATCH (a:Attribute {id: $attr_id})
MERGE (p)-[:HAS_ATTRIBUTE]->(a)
"""

_MERGE_PRODUCT_CATEGORY = """
MATCH (p:Product  {id: $prod_id})
MATCH (c:Category {id: $cat_id})
MERGE (p)-[:BELONGS_TO]->(c)
"""

# Dynamic relationship for LLM-extracted relations (subject → predicate → object)
# Predicate is embedded in the Cypher string — safe because we normalise it
# to an uppercase identifier before use (no user-controlled SQL injection risk).
_MERGE_PRODUCT_RELATION_TMPL = """
MATCH (a:Product {{id: $subj_id}})
MATCH (b:Product {{id: $obj_id}})
MERGE (a)-[:{predicate}]->(b)
"""


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _node_id(prefix: str, *parts: str) -> str:
    """Stable, short node ID derived from content — safe for MERGE keys."""
    raw = ":".join(p.lower().strip() for p in parts if p)
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{prefix}_{digest}"


# ---------------------------------------------------------------------------
# Statement collector (used by both real and dry-run paths)
# ---------------------------------------------------------------------------

@dataclass
class _Statement:
    cypher: str
    params: Dict[str, Any]


def _build_statements(normalized: NormalizedResult) -> List[_Statement]:
    """Convert a NormalizedResult into a flat list of (cypher, params) pairs.

    This is the single source of truth for what gets written to the graph —
    used by both the live driver and the dry-run collector.
    """
    stmts: List[_Statement] = []
    doc_stem = normalized.doc_stem

    # ---- Document node --------------------------------------------------
    doc_id = _node_id("doc", doc_stem)
    stmts.append(_Statement(_MERGE_DOCUMENT, {"id": doc_id, "doc_stem": doc_stem}))

    # ---- Section nodes (one per source chunk) ---------------------------
    # We reconstruct Section nodes from the source_chunk_ids so that even
    # documents with zero extracted products still get their section nodes.
    chunk_sec_ids: Dict[str, str] = {}
    for chunk_id in normalized.source_chunk_ids:
        sec_id = _node_id("sec", doc_stem, chunk_id)
        chunk_sec_ids[chunk_id] = sec_id
        # Parse level + name from chunk_id pattern: "<stem>__p<page>__c<n>"
        parts = chunk_id.split("__")
        sec_name = parts[0] if len(parts) == 1 else chunk_id
        level = 1
        page = 1
        if len(parts) >= 2:
            try:
                page = int(parts[1].lstrip("p"))
            except ValueError:
                pass
        stmts.append(_Statement(_MERGE_SECTION, {
            "id": sec_id,
            "name": sec_name,
            "level": level,
            "page": page,
            "doc_stem": doc_stem,
        }))
        stmts.append(_Statement(_MERGE_DOC_SECTION, {
            "doc_id": doc_id,
            "sec_id": sec_id,
        }))

    # ---- Product nodes --------------------------------------------------
    prod_id_map: Dict[str, str] = {}   # normalized name+sku key → neo4j id

    for product in normalized.products:
        prod_id = _node_id("prod", doc_stem, product.name, product.sku or "")
        # Track by (name, sku) so we can link attributes/relations later
        lookup_key = f"{product.name.lower()}::{(product.sku or '').lower()}"
        prod_id_map[lookup_key] = prod_id

        stmts.append(_Statement(_MERGE_PRODUCT, {
            "id":       prod_id,
            "name":     product.name,
            "model":    product.model,
            "sku":      product.sku,
            "category": product.category,
            "doc_stem": doc_stem,
        }))

        # Link product → section (use first source chunk)
        if product.source_chunks:
            first_chunk = product.source_chunks[0]
            sec_id = chunk_sec_ids.get(first_chunk)
            if sec_id:
                stmts.append(_Statement(_MERGE_SECTION_PRODUCT, {
                    "sec_id":  sec_id,
                    "prod_id": prod_id,
                }))

        # Category node + relationship
        if product.category:
            cat_name = product.category.strip()
            cat_id = _node_id("cat", cat_name)
            stmts.append(_Statement(_MERGE_CATEGORY, {
                "id": cat_id, "name": cat_name,
            }))
            stmts.append(_Statement(_MERGE_PRODUCT_CATEGORY, {
                "prod_id": prod_id,
                "cat_id":  cat_id,
            }))

    # ---- Attribute nodes ------------------------------------------------
    for attr in normalized.attributes:
        attr_id = _node_id("attr", doc_stem, attr.entity, attr.key)
        stmts.append(_Statement(_MERGE_ATTRIBUTE, {
            "id":            attr_id,
            "key":           attr.key,
            "value":         attr.value,
            "unit":          attr.unit,
            "value_numeric": attr.value_numeric,
        }))

        # Link to product if we can find it
        entity_key = f"{attr.entity.lower()}::"
        # Try exact match first, then SKU-agnostic
        matching_prod_id = None
        for k, pid in prod_id_map.items():
            if k.startswith(entity_key) or k.split("::")[0] == attr.entity.lower():
                matching_prod_id = pid
                break

        if matching_prod_id:
            stmts.append(_Statement(_MERGE_PRODUCT_ATTR, {
                "prod_id": matching_prod_id,
                "attr_id": attr_id,
            }))

    # ---- LLM-extracted relations ----------------------------------------
    _VALID_REL_PREDICATES = {
        "HAS_ATTRIBUTE", "IS_VARIANT_OF", "BELONGS_TO_CATEGORY", "HAS_MATERIAL",
    }

    for rel in normalized.relations:
        predicate = rel.predicate.upper().strip()
        # Only emit typed relationships between known products
        if predicate not in _VALID_REL_PREDICATES:
            continue
        if predicate in {"HAS_ATTRIBUTE", "BELONGS_TO_CATEGORY"}:
            # These are handled structurally above; skip duplicates
            continue

        # Find subject and object product IDs
        subj_key = rel.subject.lower()
        obj_key  = rel.object.lower()
        subj_id  = next((pid for k, pid in prod_id_map.items() if k.split("::")[0] == subj_key), None)
        obj_id   = next((pid for k, pid in prod_id_map.items() if k.split("::")[0] == obj_key),  None)

        if subj_id and obj_id and subj_id != obj_id:
            cypher = _MERGE_PRODUCT_RELATION_TMPL.format(predicate=predicate)
            stmts.append(_Statement(cypher, {"subj_id": subj_id, "obj_id": obj_id}))

    return stmts


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Summary of what was written (or would be written) to Neo4j."""

    doc_stem: str
    statements_executed: int = 0
    nodes_merged: int = 0        # estimated from statement types
    rels_merged: int = 0
    dry_run: bool = False
    error: Optional[str] = None
    statements: List[Dict[str, Any]] = field(default_factory=list)  # populated in dry-run

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "doc_stem":            self.doc_stem,
            "dry_run":             self.dry_run,
            "statements_executed": self.statements_executed,
            "nodes_merged":        self.nodes_merged,
            "rels_merged":         self.rels_merged,
            "error":               self.error,
        }
        if self.dry_run:
            d["statements"] = self.statements
        return d

    def __repr__(self) -> str:
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        err = f"  ERROR: {self.error}" if self.error else ""
        return (
            f"IngestionResult({mode}, doc={self.doc_stem!r}, "
            f"stmts={self.statements_executed}, "
            f"nodes~{self.nodes_merged}, rels~{self.rels_merged}){err}"
        )


def _count_nodes_rels(stmts: List[_Statement]) -> Tuple[int, int]:
    """Rough count of node vs relationship statements for reporting."""
    node_keywords = {"MERGE (d:Document", "MERGE (s:Section", "MERGE (p:Product",
                     "MERGE (a:Attribute", "MERGE (c:Category"}
    nodes = sum(1 for s in stmts if any(s.cypher.strip().startswith(k) for k in node_keywords))
    rels  = len(stmts) - nodes
    return nodes, rels


# ---------------------------------------------------------------------------
# GraphIngestor — public entry point
# ---------------------------------------------------------------------------

class GraphIngestor:
    """Ingests NormalizedResult objects into Neo4j.

    Parameters
    ----------
    uri
        Bolt URI, e.g. ``"bolt://localhost:7687"``.
    user / password
        Neo4j credentials.
    database
        Target database name (default: ``"neo4j"``).
    dry_run
        If True, collect all statements but do NOT connect to Neo4j.
        Useful for testing and inspecting what would be written.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "catalogbank",
        database: str = "neo4j",
        dry_run: bool = False,
    ) -> None:
        self.uri      = uri
        self.user     = user
        self.password = password
        self.database = database
        self.dry_run  = dry_run
        self._driver  = None   # lazy — only opened when first needed

        if not dry_run:
            self._open_driver()

    # ------------------------------------------------------------------
    # Driver lifecycle
    # ------------------------------------------------------------------

    def _open_driver(self) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError(
                "neo4j package not installed. Run: pip install neo4j>=5.0.0"
            ) from exc

        self._driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        logger.info("Neo4j driver opened: %s", self.uri)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "GraphIngestor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def ensure_constraints(self) -> None:
        """Create uniqueness constraints if they don't exist (run once)."""
        if self.dry_run:
            logger.info("dry_run=True — skipping constraint creation")
            return
        with self._driver.session(database=self.database) as session:
            for stmt in _CONSTRAINT_STATEMENTS:
                try:
                    session.run(stmt)
                except Exception as exc:
                    # Older Neo4j versions may not support IF NOT EXISTS — log and continue
                    logger.warning("Constraint statement failed (non-fatal): %s — %s", stmt[:60], exc)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, normalized: NormalizedResult) -> IngestionResult:
        """Write one NormalizedResult to Neo4j (or collect statements in dry-run).

        All writes for this document execute inside a single write transaction
        so a failure leaves the graph unchanged.

        Returns
        -------
        IngestionResult
            Summary of what was written.
        """
        stmts = _build_statements(normalized)
        nodes_est, rels_est = _count_nodes_rels(stmts)

        result = IngestionResult(
            doc_stem=normalized.doc_stem,
            statements_executed=len(stmts),
            nodes_merged=nodes_est,
            rels_merged=rels_est,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            result.statements = [
                {"cypher": s.cypher.strip(), "params": s.params}
                for s in stmts
            ]
            logger.info("dry_run: collected %d statements for %s", len(stmts), normalized.doc_stem)
            return result

        # Live write
        try:
            with self._driver.session(database=self.database) as session:
                with session.begin_transaction() as tx:
                    for stmt in stmts:
                        tx.run(stmt.cypher, **stmt.params)
                    tx.commit()
            logger.info(
                "Ingested %s: %d stmts, ~%d nodes, ~%d rels",
                normalized.doc_stem, len(stmts), nodes_est, rels_est,
            )
        except Exception as exc:
            result.error = str(exc)
            logger.error("Neo4j ingestion failed for %s: %s", normalized.doc_stem, exc)

        return result

    def ingest_batch(
        self,
        normalized_results: List[NormalizedResult],
        progress: bool = True,
    ) -> List[IngestionResult]:
        """Ingest a list of NormalizedResults sequentially."""
        all_results: List[IngestionResult] = []
        total = len(normalized_results)
        for i, nr in enumerate(normalized_results, 1):
            if progress:
                print(f"  [{i}/{total}] ingesting {nr.doc_stem!r} …", end=" ", flush=True)
            result = self.ingest(nr)
            if progress:
                if result.error:
                    print(f"✗ {result.error}")
                else:
                    mode = "(dry-run)" if result.dry_run else ""
                    print(
                        f"✓ {mode}  "
                        f"stmts={result.statements_executed}  "
                        f"nodes~{result.nodes_merged}  "
                        f"rels~{result.rels_merged}"
                    )
            all_results.append(result)
        return all_results
