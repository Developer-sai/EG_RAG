"""The five systems compared in the evaluation (Section 3.3 of the report).

S1  LLM only                      no documents
S2  naive RAG                     retrieval, no citations
S3  citation-only RAG             LLM cites paragraph ids, nothing is checked
S4  EG-RAG                        quote-matched provenance + NLI status for every claim
S5  EG-RAG + guardrails           S4 + sufficiency gate + removal of unsupported claims;
                                  with no evidence it still answers, labelled "not found"

Ablation switches (all on by default): use_quote, use_verifier, use_gate.
Revised verifier (S4v2 / S5v2): sentence_check=True checks each claim against the quoted sentence instead of the
whole paragraph, and looks for conflicts only in sentences of the same document that talk about the same thing.
"""
from __future__ import annotations

from dataclasses import asdict

from .generate import GENERAL_FALLBACK, build_prompt, normalise_claims
from .ingest import Paragraph
from .provenance import locate
from .retrieve import Retriever
from .verify import NLI, answer_status, best_sentence, claim_status, evidence_for

SYSTEMS = ["S1", "S2", "S3", "S4", "S5"]
NOT_FOUND_TEXT = "No evidence found in your documents."


class EGRAG:
    def __init__(self, paragraphs: list[Paragraph], cfg: dict, embedder=None, reranker=None, nli=None,
                 use_quote=True, use_verifier=True, use_gate=True, sentence_check=False):
        r, g = cfg["retrieval"], cfg["guardrails"]
        self.cfg, self.top_k = cfg, r["top_k"]
        self.retriever = Retriever(paragraphs, embedder, reranker, r["candidates"], r["rrf_k"])
        self.nli = NLI(nli) if (nli is not None and use_verifier) else None
        self.use_quote, self.use_gate = use_quote, use_gate
        self.sentence_check = sentence_check        # revised verifier: sentence-level evidence, same-document conflicts
        self.gate_t, self.entail_t = g["sufficiency_threshold"], g["entail_threshold"]
        self.contra_t, self.quote_min = g["contradict_threshold"], g["quote_match_min"]

    def answer(self, question: str, system: str, llm) -> dict:
        hits = [] if system == "S1" else self.retriever.search(question, self.top_k)
        out = {"system": system, "question": question, "hits": [(h.para.pid, round(h.score, 4)) for h in hits],
               "best_score": round(hits[0].score, 4) if hits else None, "general_answer": None}

        # sufficiency gate (S5): weak retrieval -> do not ask the LLM to ground anything
        if system == "S5" and self.use_gate and (not hits or hits[0].score < self.gate_t):
            return self._not_found(out, question, llm, reason="gate")

        sys_prompt, user_prompt = build_prompt(system, question, hits)
        resp = llm.chat(sys_prompt, user_prompt)
        data = resp["json"] or {}
        out.update(answer=str(data.get("answer", resp["text"])), model=resp["model"],
                   latency_s=resp["latency_s"], usage=resp.get("usage"))
        claims = normalise_claims(data.get("claims"))
        paras = [h.para for h in hits]

        if system in ("S1", "S2"):                       # no citations: nothing to locate
            out.update(claims=[{**c, "location": None, "status": "uncited"} for c in claims], status="n/a")
            return out

        if data.get("not_found") or not claims:
            out.update(claims=[], status="not_found")
            return self._not_found(out, question, llm, reason="model") if system == "S5" else out

        for c in claims:
            loc = locate(c, paras, self.quote_min, use_quote=(system != "S3" and self.use_quote))
            c["location"] = asdict(loc) if loc else None

        if system == "S3":                               # citation-only: presented as grounded, unchecked
            for c in claims:
                c["status"] = "grounded" if c["location"] else "unsupported"
            out.update(claims=claims, status="grounded" if any(c["location"] for c in claims) else "not_found")
            return out

        self._verify(claims, paras)
        if system == "S5":                               # guardrail: keep only claims with evidence
            kept = [c for c in claims if c["status"] != "unsupported"]
            out["removed_claims"] = [c["text"] for c in claims if c["status"] == "unsupported"]
            claims = kept
            if not any(c["status"] in ("grounded", "contradicted") for c in claims):
                out.update(claims=[], status="not_found")
                return self._not_found(out, question, llm, reason="no_grounded_claims")
            out["answer"] = " ".join(c["text"] for c in claims)
        out.update(claims=claims, status=answer_status(claims))
        return out

    def _verify(self, claims: list[dict], paras: list[Paragraph]) -> None:
        by_id = {p.pid: p for p in paras}
        for c in claims:
            c["conflict"] = False
            if c["location"] is None:
                c.update(status="unsupported", entail=0.0, contradict=0.0)
                continue
            if self.nli is None:                        # ablation: located quote = grounded
                c.update(status="grounded", entail=None, contradict=None)
                continue
            own = by_id[c["location"]["pid"]]
            premise = evidence_for(own.text, c.get("quote", "")) if self.sentence_check else own.text
            p = self.nli.probs([(premise, c["text"])])[0]
            c.update(status=claim_status(p, self.entail_t, self.contra_t), **p)
            if c["status"] == "grounded":               # does another retrieved paragraph disagree?
                others = [q for q in paras if q.pid != own.pid and (not self.sentence_check or q.doc == own.doc)]
                pairs = [(q, q.text if not self.sentence_check else best_sentence(q.text, c["text"])) for q in others]
                pairs = [(q, t) for q, t in pairs if t]
                for (q, _), pq in zip(pairs, self.nli.probs([(t, c["text"]) for _, t in pairs])):
                    if pq["contradict"] >= self.contra_t and pq["contradict"] > pq["entail"]:
                        c["conflict"], c["conflict_with"] = True, q.pid
                        break

    def _not_found(self, out: dict, question: str, llm, reason: str) -> dict:
        resp = llm.chat(GENERAL_FALLBACK, f"Question: {question}")
        general = (resp["json"] or {}).get("answer", resp["text"])
        out.update(status="not_found", claims=[], not_found_reason=reason,
                   answer=f"{NOT_FOUND_TEXT} General answer (unverified): {general}", general_answer=general)
        out.setdefault("model", resp["model"])
        return out
