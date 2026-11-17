"""Module 1 - Output directory & artifact manager.

Creates the ``outputs/`` tree at runtime and provides one standardized helper
for exporting 300-dpi PNG figures (title, axis labels, legend, UTC timestamp)
that always closes the figure to avoid matplotlib memory leaks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: safe on servers / CI
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

SUBDIRS: tuple[str, ...] = (
    "data_plots",
    "validation_plots",
    "comparison_plots",
    "saved_models",
)
DEFAULT_ROOT: Path = Path(__file__).resolve().parent.parent / "outputs"


@dataclass
class ArtifactManager:
    """Owns the on-disk layout for every artifact the pipeline produces."""

    root: Path = DEFAULT_ROOT
    dpi: int = 300
    paths: dict[str, Path] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.initialize()

    # ---------------------------------------------------------------- layout
    def initialize(self) -> None:
        """Create the directory structure (idempotent)."""
        self.root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRS:
            path = self.root / name
            path.mkdir(parents=True, exist_ok=True)
            self.paths[name] = path

    def dir(self, subdir: str) -> Path:
        """Return one of the managed sub-directories."""
        if subdir not in self.paths:
            raise KeyError(f"Unknown artifact directory '{subdir}'. Valid: {SUBDIRS}")
        return self.paths[subdir]

    # ---------------------------------------------------------------- figures
    def save_figure(
        self,
        fig: Figure,
        subdir: str,
        filename: str,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> Path:
        """Finalize and export a figure as a 300-dpi PNG, then close it.

        ``xlabel``/``ylabel`` are applied to every axes that has no label yet,
        a legend is added to every axes with labelled artists, and a UTC
        generation timestamp is stamped in the footer.
        """
        if not filename.lower().endswith(".png"):
            filename += ".png"
        out = self.dir(subdir) / filename
        try:
            if title:
                fig.suptitle(title, fontsize=14, fontweight="bold")
            for ax in fig.get_axes():
                if xlabel and not ax.get_xlabel():
                    ax.set_xlabel(xlabel)
                if ylabel and not ax.get_ylabel():
                    ax.set_ylabel(ylabel)
                handles, labels = ax.get_legend_handles_labels()
                if labels and ax.get_legend() is None:
                    ax.legend(fontsize=8, frameon=False)
            stamp = datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC")
            fig.text(0.995, 0.005, stamp, ha="right", va="bottom", fontsize=7, color="gray")
            fig.savefig(out, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)  # always release the figure
        return out

    # ---------------------------------------------------------------- json
    def write_json(self, subdir: str, filename: str, payload: dict[str, Any]) -> Path:
        """Write a JSON artifact (used for the model manifest)."""
        out = self.dir(subdir) / filename
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out

    def read_json(self, subdir: str, filename: str) -> dict[str, Any]:
        """Read a previously written JSON artifact."""
        return json.loads((self.dir(subdir) / filename).read_text(encoding="utf-8"))


if __name__ == "__main__":
    am = ArtifactManager()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="demo")
    print("Saved", am.save_figure(fig, "data_plots", "_smoke_test", "Smoke test", "x", "y"))
