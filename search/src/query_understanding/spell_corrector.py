"""Domain-aware spell correction (SymSpell trained on the catalog vocabulary)."""
from __future__ import annotations

from collections import Counter

import pandas as pd
from symspellpy import SymSpell, Verbosity

from src.common.text import raw_tokens


PROTECTED = {"under", "below", "less", "than", "max", "up", "to", "cheaper", "or", "and", "for", "with", "the"}


class SpellCorrector:
    def __init__(self, max_edit: int = 2):
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=7)
        self.vocab: set[str] = set()
        self.max_edit = max_edit

    def fit(self, catalog: pd.DataFrame, extra_terms: list[str] | None = None) -> "SpellCorrector":
        cnt: Counter = Counter()
        for col, w in (("title", 3), ("brand", 3), ("attributes", 2), ("description", 1)):
            for text in catalog[col].astype(str).values:
                for t in raw_tokens(text):
                    if t.startswith("$") or t.replace(".", "").isdigit():
                        continue
                    cnt[t] += w
                    if "-" in t:
                        for p in t.split("-"):
                            if len(p) > 1:
                                cnt[p] += w
        for term in extra_terms or []:
            for t in raw_tokens(term):
                cnt[t] += 5
        for term, c in cnt.items():
            self.sym.create_dictionary_entry(term, c)
        self.vocab = set(cnt)
        return self

    def correct(self, query: str) -> tuple[str, list[tuple[str, str]]]:
        """Return corrected query and a list of (original, corrected) edits."""
        out, edits = [], []
        for t in raw_tokens(query):
            if t in self.vocab or t in PROTECTED or len(t) <= 3 or t.startswith("$") or t.replace(".", "").isdigit():
                out.append(t)
                continue
            sugg = self.sym.lookup(t, Verbosity.CLOSEST, max_edit_distance=self.max_edit, include_unknown=True)
            best = sugg[0].term if sugg else t
            if best != t:
                edits.append((t, best))
            out.append(best)
        return " ".join(out), edits
