"""Structured stage-by-stage logging: console + JSON lines."""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path


class StageLogger:
    def __init__(self, path: str | Path = "outputs/logs/pipeline_execution.jsonl", reset: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()

    def log(self, stage: str, event: str, **fields) -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        extra = " ".join(f"{k}={v}" for k, v in fields.items() if not isinstance(v, (dict, list)))
        print(f"[{rec['ts']}] [{stage}] {event} {extra}", file=sys.stdout, flush=True)

    @contextmanager
    def stage(self, name: str, **fields):
        self.log(name, "start", **fields)
        t0 = time.time()
        try:
            yield
        except Exception as e:
            self.log(name, "error", error=repr(e), seconds=round(time.time() - t0, 2))
            raise
        self.log(name, "end", seconds=round(time.time() - t0, 2))
