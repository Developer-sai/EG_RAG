"""Small offline stand-ins for the models, used only by the tests (no network, no GPU).

They make the tests check the plumbing (ids, locations, statuses, metrics), not model quality.
"""
import re
import zlib

import numpy as np


def _words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


class StubEmbedder:
    """Bag-of-words hashing vectors."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in _words(t):
                out[i, zlib.crc32(w.encode()) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class StubReranker:
    """Score = word overlap minus 3, so unrelated questions fall below a gate of 0."""

    def predict(self, pairs):
        return [len(_words(q) & _words(p)) - 3.0 for q, p in pairs]


class _Cfg:
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}


class StubNLI:
    """Entailment if every number in the claim appears in the premise; contradiction if the claim
    has numbers and the premise shares words but different numbers; otherwise neutral."""

    class model:  # noqa: N801  (mimics CrossEncoder.model.config)
        config = _Cfg()

    def predict(self, pairs, apply_softmax=True):
        out = []
        for premise, claim in pairs:
            nums_c, nums_p = set(re.findall(r"\d+", claim)), set(re.findall(r"\d+", premise))
            overlap = len(_words(premise) & _words(claim)) / max(1, len(_words(claim)))
            if nums_c and nums_c <= nums_p and overlap > 0.4:
                out.append([0.05, 0.9, 0.05])
            elif nums_c and nums_p and not nums_c & nums_p and overlap > 0.5:
                out.append([0.9, 0.05, 0.05])
            else:
                out.append([0.1, 0.2, 0.7])
        return np.array(out)


class FakeLLM:
    """Returns scripted JSON keyed by a substring of the user prompt."""

    def __init__(self, script: dict, default: dict | None = None):
        self.script, self.default, self.calls = script, default or {"answer": "I don't know.", "claims": []}, []

    def chat(self, system, user, json_mode=True):
        self.calls.append((system, user))
        data = next((v for k, v in self.script.items() if k in user), self.default)
        return {"text": str(data), "json": data, "model": "fake", "latency_s": 0.0, "usage": None}
