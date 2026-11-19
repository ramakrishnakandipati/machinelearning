"""Online feature store: real Redis when a daemon answers, otherwise fakeredis (in-process)."""
from __future__ import annotations

import json
import os

import pandas as pd

from src.config import DATA


class FeatureStore:
    def __init__(self) -> None:
        self.backend = "uninitialised"
        self.r = None
        self.n_users = 0

    def connect(self) -> "FeatureStore":
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=0.3, socket_timeout=2)
            r.ping()
            self.r, self.backend = r, "redis"
        except Exception:
            import fakeredis
            self.r, self.backend = fakeredis.FakeRedis(decode_responses=True), "fakeredis"
        return self

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def load_snapshot(self) -> None:
        """Publish the offline as-of-now user features (+ purchase sets) into the store."""
        snap = pd.read_parquet(DATA / "user_features_snapshot.parquet")
        purchased = pd.read_parquet(DATA / "purchased.parquet")
        pipe = self.r.pipeline()
        for row in snap.itertuples(index=False):
            pipe.hset(f"user:{row.user_id}", mapping={
                "tier": row.loyalty_tier, "channel": row.preferred_channel, "region": row.home_region,
                "recency_days": float(row.recency_days), "frequency_90d": float(row.frequency_90d),
                "monetary_value_90d": float(row.monetary_value_90d), "avg_basket_size": float(row.avg_basket_size),
                "history": json.dumps([int(x) for x in row.recent_item_history])})
        for u, g in purchased.groupby("user_id"):
            pipe.sadd(f"purchased:{u}", *[int(i) for i in g.item_id])
        pipe.execute()
        self.n_users = len(snap)

    def get_user(self, user_id: int) -> dict | None:
        h = self.r.hgetall(f"user:{user_id}")
        if not h:
            return None
        h["history"] = json.loads(h["history"])
        return h

    def purchased(self, user_id: int) -> set[int]:
        return {int(x) for x in self.r.smembers(f"purchased:{user_id}")}
