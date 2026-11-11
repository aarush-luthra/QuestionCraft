"""
scripts/test_api_client.py
==========================
Verifies all FastAPI inference endpoints in-memory using TestClient.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.serving.app import app

with TestClient(app) as client:
    print("\n--- 1. Testing GET /health ---")
    r = client.get("/health")
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    print("\n--- 2. Testing GET /artifacts (11-category ML Artifact Manifest) ---")
    r = client.get("/artifacts")
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    assert "dataset" in r.json()
    assert "monitoring_artifacts" in r.json()

    print("\n--- 3. Testing POST /predict ---")
    sample_q = {
        "question_text": "Design and implement a cache-aware B-Tree in C++ for large-scale storage."
    }
    r = client.post("/predict", json=sample_q)
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    pred = r.json()
    assert pred["difficulty"] in ["Easy", "Medium", "Hard"]
    assert pred["blooms_level"] in ["L1", "L2", "L3", "L4", "L5", "L6"]

    print("\n--- 4. Testing POST /batch/classify ---")
    batch_q = {
        "questions": [
            {"question_text": "Define a binary search tree and state its properties."},
            {
                "question_text": "Apply Dijkstra algorithm to compute shortest paths in a directed graph."
            },
            {
                "question_text": "Synthesize a Red-Black tree deletion routine handling double-black rebalancing."
            },
        ]
    }
    r = client.post("/batch/classify", json=batch_q)
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    assert len(r.json()["results"]) == 3

    print("\n--- 5. Testing POST /deploy/canary ---")
    r = client.post("/deploy/canary?ratio=0.2")
    print("Status:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200

    print("\n--- 6. Testing GET /metrics (Prometheus) ---")
    r = client.get("/metrics")
    print("Status:", r.status_code)
    print(r.text[:300] + "...")
    assert "questioncraft_requests_total" in r.text

    print("\n[+] All API endpoint integration checks PASSED successfully!")
