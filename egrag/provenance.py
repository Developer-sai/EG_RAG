"""Step 4 - Deterministic provenance.

The LLM gives a paragraph id and a quote. Here we check that the quote really is in that
paragraph and read document, page, paragraph number and character span from the index.
If the quote is in a different retrieved paragraph we move the citation there; if it is in
none, the claim has no location and is treated as unsupported. An LLM-written location can
be hallucinated; a location found by string matching cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .ingest import Paragraph


@dataclass
class Location:
    pid: str
    doc: str
    page: int | None
    paragraph: int          # n-th paragraph on the page
    start: int              # character span of the quote inside the document text
    end: int
    snippet: str            # "...text before [quote] text after..."
    method: str             # exact | fuzzy | relocated


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def find_quote(quote: str, text: str, min_ratio: float) -> tuple[int, int, float] | None:
    """Return (start, end, ratio) of the quote inside text, exact first, then fuzzy."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return None
    i = t.find(q)
    if i >= 0:
        return i, i + len(q), 1.0
    # fuzzy: compare the quote with every window of the same number of words
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", t)]
    n = len(q.split())
    best = (0.0, 0, 0)
    sm = SequenceMatcher(None, q, None, autojunk=False)
    for k in range(max(1, len(words) - n + 1)):
        s, e = words[k][0], words[min(k + n, len(words)) - 1][1]
        sm.set_seq2(t[s:e])
        if sm.quick_ratio() >= min_ratio and sm.quick_ratio() > best[0]:
            r = sm.ratio()
            if r > best[0]:
                best = (r, s, e)
    return (best[1], best[2], best[0]) if best[0] >= min_ratio else None


def snippet(text: str, start: int, end: int, context: int = 60) -> str:
    """Show the quote in [brackets] with some context, in the document's own spelling."""
    t = text if len(text) == len(_norm(text)) else _norm(text)   # offsets come from the normalised text
    left, right = max(0, start - context), min(len(t), end + context)
    return ("…" if left else "") + t[left:start] + "[" + t[start:end] + "]" + t[end:right] + ("…" if right < len(t) else "")


def locate(claim: dict, candidates: list[Paragraph], min_ratio: float = 0.85, use_quote: bool = True) -> Location | None:
    """Resolve a claim's citation to a verified location, or None."""
    by_id = {p.pid: p for p in candidates}
    cited = by_id.get(claim.get("cite") or "")
    if not use_quote:                        # ablation / citation-only: trust the cited id blindly
        return _loc(cited, 0, len(cited.text), "cited") if cited else None
    quote = claim.get("quote") or ""
    order = ([cited] if cited else []) + [p for p in candidates if p is not cited]
    for p in order:
        hit = find_quote(quote, p.text, min_ratio)
        if hit:
            method = "relocated" if p is not cited else ("exact" if hit[2] == 1.0 else "fuzzy")
            return _loc(p, hit[0], hit[1], method)
    return None


def _loc(p: Paragraph, s: int, e: int, method: str) -> Location:
    para_no = int(p.pid.rsplit("-", 1)[1])
    return Location(p.pid, p.doc, p.page, para_no, p.start + s, p.start + e, snippet(p.text, s, e), method)
