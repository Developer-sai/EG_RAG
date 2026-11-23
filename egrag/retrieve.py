"""Step 2 - Retrieval: hybrid search over paragraph units.

1. BM25 (Robertson & Zaragoza, 2009) finds exact words such as "annual leave" or "26 days".
2. A dense embedding model finds paraphrases ("holidays" ~ "annual leave").
3. Reciprocal Rank Fusion (Cormack et al., 2009) merges the two rankings:
       score(p) = sum over rankings of 1 / (k + rank(p))
4. A cross-encoder re-ranker reads (question, paragraph) together and orders the shortlist.
   Its best score also feeds the sufficiency gate: if even the best paragraph scores low,
   the documents probably do not answer the question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Paragraph


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Hit:
    para: Paragraph
    score: float          # re-ranker score


class Retriever:
    def __init__(self, paragraphs: list[Paragraph], embedder=None, reranker=None,
                 candidates: int = 20, rrf_k: int = 60):
        self.paras = paragraphs
        self.candidates, self.rrf_k = candidates, rrf_k
        self.bm25 = BM25Okapi([tokenize(p.text) or ["_"] for p in paragraphs]) if paragraphs else None
        self.embedder = embedder
        self.reranker = reranker
        self.vectors = None
        if self.embedder is not None and paragraphs:
            self.vectors = np.asarray(self.embedder.encode([p.text for p in paragraphs], normalize_embeddings=True))

    # --- individual rankers -------------------------------------------------
    def bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[: self.candidates])

    def dense_rank(self, query: str) -> list[int]:
        q = np.asarray(self.embedder.encode([query], normalize_embeddings=True))[0]
        return list(np.argsort(-(self.vectors @ q))[: self.candidates])

    # --- fusion + re-ranking --------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self.paras or not query.strip():
            return []
        rankings = [self.bm25_rank(query)]
        if self.vectors is not None:
            rankings.append(self.dense_rank(query))
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        shortlist = sorted(fused, key=fused.get, reverse=True)[: self.candidates]
        if self.reranker is None:
            return [Hit(self.paras[i], fused[i]) for i in shortlist[:top_k]]
        scores = np.asarray(self.reranker.predict([(query, self.paras[i].text) for i in shortlist]), dtype=float)
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.paras[shortlist[j]], float(scores[j])) for j in order]


def load_local_models(cfg: dict):
    """Embedder, re-ranker and NLI model (downloaded from Hugging Face on first use)."""
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        m = cfg["local_models"]
        return SentenceTransformer(m["embedder"]), CrossEncoder(m["reranker"]), CrossEncoder(m["nli"])
    except Exception as err:
        raise RuntimeError("Could not load the local models. Run 'pip install -r requirements.txt' and make sure "
                           f"huggingface.co is reachable the first time (they are cached afterwards). Detail: {err}") from err
