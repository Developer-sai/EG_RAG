"""Step 3 - Generation: prompts for the five systems.

The EG-RAG prompt asks for structured JSON: a list of short claims, each citing ONE paragraph
id and copying a verbatim quote from it. The model never writes a page number; the page and
document are looked up from the paragraph id later (provenance.py).
"""
from __future__ import annotations

from .ingest import Paragraph
from .retrieve import Hit

ROLE = "You answer employees' questions about workplace policy documents."

LLM_ONLY = ROLE + """ Answer from your general knowledge.
Return JSON: {"answer": "<2-4 sentences>", "claims": ["<one factual statement per item>"]}"""

NAIVE_RAG = ROLE + """ Use the passages below to answer.
Return JSON: {"answer": "<2-4 sentences>", "claims": ["<one factual statement per item>"]}"""

CITATION_RAG = ROLE + """ Use only the passages below. Cite the passage id in square brackets for each claim.
If the passages do not contain the answer, set "not_found" to true.
Return JSON: {"answer": "<2-4 sentences>", "not_found": false,
              "claims": [{"text": "<one factual statement>", "cite": "<passage id>"}]}"""

EG_RAG = ROLE + """ Use only the passages below.
Rules:
1. Split the answer into short claims. Each claim cites exactly ONE passage id.
2. For each claim copy a "quote": the exact words (one sentence or less) from that passage that support it.
   Copy them character for character; do not paraphrase the quote.
3. Do not use outside knowledge. If no passage answers the question, return "not_found": true and no claims.
Return JSON: {"answer": "<2-4 sentences>", "not_found": false,
              "claims": [{"text": "<claim>", "cite": "<passage id>", "quote": "<verbatim words>"}]}"""

GENERAL_FALLBACK = ROLE + """ The user's documents do not contain the answer. Give a short, cautious general answer
and say that it must be checked against the organisation's own policy.
Return JSON: {"answer": "<2-3 sentences>"}"""


def passages(hits: list[Hit], with_ids: bool) -> str:
    lines = []
    for h in hits:
        p: Paragraph = h.para
        lines.append(f"[{p.pid}] {p.text}" if with_ids else p.text)
    return "\n\n".join(lines)


def build_prompt(system_id: str, question: str, hits: list[Hit] | None = None) -> tuple[str, str]:
    """Return (system prompt, user prompt) for S1..S5."""
    if system_id == "S1":
        return LLM_ONLY, f"Question: {question}"
    if system_id == "S2":
        return NAIVE_RAG, f"Passages:\n{passages(hits, False)}\n\nQuestion: {question}"
    if system_id == "S3":
        return CITATION_RAG, f"Passages:\n{passages(hits, True)}\n\nQuestion: {question}"
    return EG_RAG, f"Passages:\n{passages(hits, True)}\n\nQuestion: {question}"


def normalise_claims(raw) -> list[dict]:
    """Accept both ["text", ...] and [{"text":..., "cite":..., "quote":...}, ...]."""
    claims = []
    if isinstance(raw, (str, dict)):          # a single claim instead of a list
        raw = [raw]
    if not isinstance(raw, list):
        return claims
    for c in raw:
        if isinstance(c, str):
            claims.append({"text": c, "cite": None, "quote": None})
        elif isinstance(c, dict) and c.get("text"):
            cite = c.get("cite")
            if isinstance(cite, list):
                cite = cite[0] if cite else None
            if c.get("quote") is not None and not isinstance(c.get("quote"), str):
                c["quote"] = str(c["quote"])
            claims.append({"text": str(c["text"]), "cite": str(cite).strip("[] ") if cite else None,
                           "quote": c.get("quote")})
    return claims
