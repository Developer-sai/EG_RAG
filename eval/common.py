"""Shared helpers for the evaluation: corpus, question set and the LLM judge."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from egrag import ROOT, load_config            # noqa: E402
from egrag.ingest import Paragraph, load_folder  # noqa: E402
from egrag.provenance import find_quote         # noqa: E402

CFG = load_config()
RUNS = ROOT / CFG["paths"]["results"] / "runs"


def load_corpus() -> tuple[list[Paragraph], dict]:
    paras, pdfs = [], {}
    for folder in (CFG["paths"]["raw"], CFG["paths"]["synthetic"]):
        p, f, warnings = load_folder(ROOT / folder)
        for w in warnings:
            print("WARNING:", w)
        paras, pdfs = paras + p, pdfs | f
    keep = set(CFG["corpus"])                        # only the two PDFs of the study
    return [p for p in paras if p.doc in keep], {d: f for d, f in pdfs.items() if d in keep}


def corpus_fingerprint(paras: list[Paragraph]) -> str:
    return hashlib.sha256("".join(p.pid + p.text for p in paras).encode()).hexdigest()[:12]


def load_questions() -> list[dict]:
    with open(ROOT / CFG["paths"]["questions"], encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_locations(q: dict, paras: list[Paragraph]) -> list[dict]:
    """Map each gold quote to its paragraph and document-level character span."""
    out = []
    for g in q.get("gold", []):
        for p in paras:
            if p.doc == g["doc"] and (g.get("page") in (None, p.page)):
                hit = find_quote(g["quote"], p.text, 0.9)
                if hit:
                    out.append({"pid": p.pid, "doc": p.doc, "page": p.page,
                                "start": p.start + hit[0], "end": p.start + hit[1]})
                    break
    return out


# ---------------------------------------------------------------- LLM judge
JUDGE_ANSWER = """You grade answers to questions about workplace policy documents.
Given the question, the gold answer and a system answer, decide if the system answer is correct.
If the gold answer is NOT_FOUND, the system answer is correct only if it says the documents do not contain
the answer (a general answer clearly labelled as not from the documents also counts as correct).
Return JSON: {"correct": true or false, "reason": "<one sentence>"}"""

JUDGE_CLAIM = """You check whether a claim is supported by an evidence passage.
supported = the passage states the claim; contradicted = the passage states something incompatible;
unsupported = the passage does not say it.
Return JSON: {"label": "supported" | "unsupported" | "contradicted"}"""


def judge_answer(judge, q: dict, answer: str) -> bool:
    gold = q["gold_answer"] if q["answerable"] else "NOT_FOUND"
    r = judge.chat(JUDGE_ANSWER, f"Question: {q['question']}\nGold answer: {gold}\nSystem answer: {answer}")
    return bool((r["json"] or {}).get("correct", False))


def judge_claim(judge, claim: str, evidence: str) -> str:
    r = judge.chat(JUDGE_CLAIM, f"Evidence passage: {evidence}\nClaim: {claim}")
    label = str((r["json"] or {}).get("label", "unsupported")).lower()
    return label if label in {"supported", "unsupported", "contradicted"} else "unsupported"
