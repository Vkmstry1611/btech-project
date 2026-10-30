"""Smoke test for the FastAPI catalog service (no server needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient
from catalogbank_ocr.api.app import app, store

client = TestClient(app)

print(f"loaded catalogs: {len(store.catalogs)} -> {list(store.catalogs)}")

checks = [
    ("GET /", client.get("/")),
    ("GET /health", client.get("/health")),
    ("GET /catalogs", client.get("/catalogs")),
    ("GET /metrics", client.get("/metrics")),
]
for label, r in checks:
    print(f"  {label:<20} {r.status_code}")
    assert r.status_code == 200, r.text

stem = list(store.catalogs)[0]
r = client.get(f"/catalogs/{stem}")
print(f"  GET /catalogs/{stem}  {r.status_code}  items_in_tree=ok")
assert r.status_code == 200

r = client.get(f"/catalogs/{stem}/search?q=1455N117")
print(f"  search '1455N117'      {r.status_code}  hits={len(r.json()['items'])}")
assert r.status_code == 200

r = client.get("/catalogs/does_not_exist")
assert r.status_code == 404
print("  404 for missing stem   ok")

print("\nALL SMOKE TESTS PASSED")
