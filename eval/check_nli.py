"""Sanity check of the NLI verifier: a claim copied from its paragraph must come out as entailed.

    python eval/check_nli.py
"""
import numpy as np

from common import CFG

from egrag.retrieve import load_local_models
from egrag.verify import NLI

_, _, model = load_local_models(CFG)
print("model labels:", model.model.config.id2label)
nli = NLI(model)
print("label -> column used:", nli.idx)

P = ("The standard working week is 37.5 hours, worked Monday to Friday between 8:00 and 18:00. "
     "Core hours, when all staff must be available, are 10:00 to 16:00.")
pairs = [
    ("copied sentence (expect entail high)", P, "The standard working week is 37.5 hours, worked Monday to Friday between 8:00 and 18:00."),
    ("identical short text (expect entail ~1)", "Requests are answered in writing within ten working days.",
     "Requests are answered in writing within ten working days."),
    ("paraphrase (expect entail high)", P, "Staff normally work 37.5 hours a week from Monday to Friday."),
    ("wrong number (expect contradict high)", P, "The standard working week is 40 hours."),
    ("unrelated (expect neutral high)", P, "The office is located in Paris."),
]
raw = np.asarray(model.predict([(p, h) for _, p, h in pairs]))
soft = np.asarray(model.predict([(p, h) for _, p, h in pairs], apply_softmax=True))
for (name, p, h), r, s, d in zip(pairs, raw, soft, nli.probs([(p, h) for _, p, h in pairs])):
    print(f"{name:42s} entail={d['entail']:.3f} contradict={d['contradict']:.3f} | raw={np.round(r, 2)} softmax={np.round(s, 3)}")
