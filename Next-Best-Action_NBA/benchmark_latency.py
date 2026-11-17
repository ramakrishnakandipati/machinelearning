"""In-process latency benchmark for /v1/nba/recommend (reports p50/p95/p99).

Usage: python benchmark_latency.py [--n 2000]
Uses the in-memory contact store, so Redis network latency is not included.
"""
from __future__ import annotations

import argparse
import random

import numpy as np
from fastapi.testclient import TestClient

from nba.service import app, state


def main(n: int) -> None:
    triggers = ["app_open", "browse_abandon", "post_purchase"]
    with TestClient(app) as client:
        ids = list(state.features)[:500] or list(range(500))
        for _ in range(100):  # warm-up
            client.post("/v1/nba/recommend", json={"customer_id": random.choice(ids), "trigger": "app_open"})
        lat = []
        for _ in range(n):
            r = client.post("/v1/nba/recommend", json={"customer_id": random.choice(ids), "trigger": random.choice(triggers)})
            lat.append(r.json()["latency_ms"])
    p = np.percentile(lat, [50, 95, 99])
    print(f"handler latency over {n} calls (ms): p50={p[0]:.2f} p95={p[1]:.2f} p99={p[2]:.2f}  (target p99 < 40)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    main(ap.parse_args().n)
