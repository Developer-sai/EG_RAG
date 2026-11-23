"""Agreement checks reported in the evaluation.

    python eval/agreement.py export      # writes results/judge_sample.csv (30 claims) for hand labelling
    python eval/agreement.py score       # Cohen's kappa: LLM judge vs human

judge_sample.csv  : fill the column "human" with supported / unsupported / contradicted.
"""
from __future__ import annotations

import json
import random
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import CFG, RUNS

OUT = RUNS.parent


def export(n: int = 30) -> None:
    rows = []
    for path in RUNS.glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            for c in rec.get("claims", []):
                if c.get("judge") and c.get("location"):
                    rows.append({"file": path.stem, "qid": rec["qid"], "claim": c["text"],
                                 "evidence_pid": c["location"]["pid"], "judge": c["judge"], "human": ""})
    random.Random(CFG["seed"]).shuffle(rows)
    pd.DataFrame(rows[:n]).to_csv(OUT / "judge_sample.csv", index=False)
    print(f"wrote {min(n, len(rows))} claims to {OUT / 'judge_sample.csv'}")


def score() -> None:
    sample = pd.read_csv(OUT / "judge_sample.csv").dropna(subset=["human"])
    if len(sample):
        k = cohen_kappa_score(sample["judge"], sample["human"])
        print(f"LLM judge vs human: kappa = {k:.3f} on {len(sample)} claims, "
              f"raw agreement = {(sample.judge == sample.human).mean():.3f}")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
