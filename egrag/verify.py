"""Step 5 - Claim verification and evidence status.

A natural language inference (NLI) cross-encoder reads (evidence paragraph, claim) and gives
probabilities for entailment, contradiction and neutral.

Claim status
    grounded      quote located AND the cited paragraph entails the claim
    contradicted  the cited paragraph contradicts the claim
    unsupported   no located quote, or the paragraph neither entails nor contradicts
Answer status (shown to the user)
    grounded      at least one grounded claim and nothing contradicted
    conflict      a claim is entailed by one paragraph and contradicted by another
                  (e.g. two versions of a policy)
    contradicted  a claim is contradicted by its own evidence
    not_found     no evidence: the answer is labelled as unverified general knowledge
"""
from __future__ import annotations

import re

import numpy as np

STOP = {"the", "and", "for", "are", "that", "with", "this", "from", "have", "each", "must", "may", "will"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9\"(])", text.strip()) if s]


def evidence_for(paragraph: str, quote: str) -> str:
    """The sentence(s) of the paragraph that hold the quoted words: what the claim is really checked against.
    A small NLI model is reliable on a sentence, but not on a whole multi-sentence paragraph."""
    q = words(quote)
    sents = sentences(paragraph)
    if not q or not sents:
        return paragraph
    scored = [(len(words(s) & q) / len(q), s) for s in sents]
    chosen = [s for r, s in scored if r >= 0.3] or [max(scored)[1]]
    return " ".join(chosen)


def best_sentence(paragraph: str, claim: str, min_overlap: float = 0.4) -> str | None:
    """The sentence of another paragraph that talks about the same thing as the claim, or None."""
    c = words(claim)
    sents = sentences(paragraph)
    if not c or not sents:
        return None
    r, s = max((len(words(s) & c) / len(c), s) for s in sents)
    return s if r >= min_overlap else None


class NLI:
    """Wraps a sentence-transformers CrossEncoder trained on MNLI/SNLI."""

    def __init__(self, model):
        self.model = model
        labels = {int(k): v.lower() for k, v in model.model.config.id2label.items()}
        self.idx = {name: i for i, name in labels.items()}

    def probs(self, pairs: list[tuple[str, str]]) -> list[dict]:
        if not pairs:
            return []
        scores = np.asarray(self.model.predict(pairs, apply_softmax=True))
        return [{"entail": float(s[self.idx["entailment"]]), "contradict": float(s[self.idx["contradiction"]])}
                for s in np.atleast_2d(scores)]


def claim_status(p: dict, entail_t: float, contra_t: float) -> str:
    if p["contradict"] >= contra_t and p["contradict"] > p["entail"]:
        return "contradicted"
    if p["entail"] >= entail_t:
        return "grounded"
    return "unsupported"


def answer_status(claims: list[dict]) -> str:
    statuses = [c["status"] for c in claims]
    if any(c.get("conflict") for c in claims):
        return "conflict"
    if "contradicted" in statuses:
        return "contradicted"
    if "grounded" in statuses:
        return "grounded"
    return "not_found"
