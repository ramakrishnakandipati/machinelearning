"""Tokenisation shared by indexing and query-time code."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\$?\d+(?:\.\d+)?|[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOP = {"a", "an", "the", "for", "with", "and", "of", "in", "to", "on", "s", "-", "under", "below", "than", "less", "max"}


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        return w[:-1]
    return w


def raw_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def tokens(text: str, expand_hyphen: bool = True, drop_stop: bool = True) -> list[str]:
    """Lower-cased, light-stemmed tokens. Hyphen compounds yield compound + parts."""
    out: list[str] = []
    for t in raw_tokens(text):
        if t.startswith("$") or t.replace(".", "").isdigit():
            continue
        parts = [t]
        if expand_hyphen and "-" in t:
            parts += [p for p in t.split("-") if p]
        for p in parts:
            if drop_stop and p in STOP:
                continue
            out.append(stem(p))
    return out
