"""Structured stage-by-stage logging (console + .log + JSON summary)."""
from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List


def get_logger(name: str = "coldstart", log_dir: str | Path | None = None) -> logging.Logger:
    """Create a console logger, optionally also writing ``pipeline_execution.log``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fp = str((Path(log_dir) / "pipeline_execution.log").resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == fp for h in logger.handlers):
            fh = logging.FileHandler(fp, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class StageLogger:
    """Records timed pipeline stages and dumps ``pipeline_summary.json``."""

    def __init__(self, log_dir: str | Path, name: str = "coldstart") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(name, self.log_dir)
        self.stages: List[Dict[str, Any]] = []
        self.t0 = time.perf_counter()

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    @contextlib.contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[Dict[str, Any]]:
        """Context manager timing a stage; extra info can be added to the yielded dict."""
        rec: Dict[str, Any] = {"stage": name, "status": "running", **meta}
        self.logger.info(f"> START  {name}")
        start = time.perf_counter()
        try:
            yield rec
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = repr(exc)
            self.logger.exception(f"XX FAILED {name}: {exc}")
            raise
        finally:
            rec["seconds"] = round(time.perf_counter() - start, 3)
            self.stages.append(rec)
            if rec["status"] == "ok":
                self.logger.info(f"OK DONE  {name} ({rec['seconds']}s)")
            self._flush_stage_log(rec)

    def _flush_stage_log(self, rec: Dict[str, Any]) -> None:
        with open(self.log_dir / "pipeline_stages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def write_summary(self, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        summary = {
            "total_seconds": round(time.perf_counter() - self.t0, 3),
            "stages": self.stages,
            "all_ok": all(s["status"] == "ok" for s in self.stages),
            **(extra or {}),
        }
        (self.log_dir / "pipeline_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        return summary
