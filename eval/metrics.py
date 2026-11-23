"""Score the runs of the 10-question study.

    python eval/metrics.py                 # -> results/summary.csv, per_question.csv, outcomes.md, figure

Metrics per question (Section 3.4 of the report)
    correct            the judge accepts the answer (gold answer, or "not found" for unanswerable)
    doc/page/para_hit  a located claim points to the gold document / page / paragraph (answerable only)
    span_f1            character overlap between the cited quote and the gold span
    unsupported_rate   share of claims shown as grounded that the judge marks unsupported or contradicted
    status_correct     evidence status right: answerable -> evidence shown, unanswerable -> "not found"
    false_grounding    a claim is shown as grounded but the documents do not support it
    false_abstention   answerable question, but the system says "not found"
With 10 questions no significance tests are run: every outcome is reported per question (outcomes.md).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import RUNS, gold_locations, load_corpus, load_questions

OUT = RUNS.parent
SHOWN_GROUNDED = {"grounded"}            # claim statuses that the user sees as backed by evidence


def span_f1(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap == 0:
        return 0.0
    p, r = overlap / (a[1] - a[0]), overlap / (b[1] - b[0])
    return 2 * p * r / (p + r)


def score(rec: dict, q: dict, gold: list[dict]) -> dict:
    claims = rec.get("claims", [])
    located = [c for c in claims if c.get("location")]
    shown = [c for c in located if c.get("status") in SHOWN_GROUNDED]
    has_citations = rec["system"] not in ("S1", "S2")
    row = {"qid": q["qid"], "system": rec["name"], "generator": rec["generator"], "type": q["type"],
           "answerable": q["answerable"], "correct": int(rec["correct"]), "status": rec.get("status"),
           "n_claims": len(claims), "latency_s": rec.get("wall_s"), "error": int(rec.get("status") == "error")}
    if not has_citations:
        return row
    locs = [c["location"] for c in shown]
    if q["answerable"] and gold:
        row["doc_hit"] = int(any(l["doc"] == g["doc"] for l in locs for g in gold))
        row["page_hit"] = int(any(l["doc"] == g["doc"] and l["page"] == g["page"] for l in locs for g in gold))
        row["para_hit"] = int(any(l["pid"] == g["pid"] for l in locs for g in gold))
        row["span_f1"] = max([span_f1((l["start"], l["end"]), (g["start"], g["end"]))
                              for l in locs for g in gold if l["doc"] == g["doc"]] or [0.0])
    bad = [c for c in shown if c.get("judge") in ("unsupported", "contradicted")]
    row["n_shown"], row["n_unsupported"] = len(shown), len(bad)
    shows_evidence = rec.get("status") not in ("not_found", "error")
    row["status_correct"] = int(shows_evidence == q["answerable"])
    row["false_grounding"] = int(bool(bad) or (not q["answerable"] and bool(shown)))
    row["false_abstention"] = int(q["answerable"] and not shows_evidence)
    if q["type"] == "conflict":
        row["conflict_detected"] = int(rec.get("status") == "conflict")
    return row


def load_rows() -> pd.DataFrame:
    paras, _ = load_corpus()
    questions = load_questions()
    qmap = {q["qid"]: q for q in questions}
    golds = {qid: gold_locations(q, paras) for qid, q in qmap.items()}
    missing = [qid for qid, q in qmap.items() if q["answerable"] and not golds[qid]]
    if missing:
        print(f"WARNING: gold quote not found in corpus for {missing}")
    rows = []
    for path in sorted(RUNS.glob("*.jsonl")):
        gen, name = path.stem.split("__")
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            rec.update(generator=gen, name=name)
            rows.append(score(rec, qmap[rec["qid"]], golds[rec["qid"]]))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["generator", "system"])
    out = g.agg(questions=("qid", "count"), errors=("error", "sum"), accuracy=("correct", "mean"))
    for col in ["doc_hit", "page_hit", "para_hit", "span_f1", "status_correct",
                "false_grounding", "false_abstention", "conflict_detected", "latency_s"]:
        if col in df:
            out[col] = g[col].mean()
    if "n_shown" in df:
        out["unsupported_rate"] = g["n_unsupported"].sum() / g["n_shown"].sum().replace(0, np.nan)
    return out.reset_index()


def outcomes(df: pd.DataFrame) -> str:
    """One table per generator: for every question, was each system's answer judged correct (and the status of S4)."""
    parts = []
    for gen in df.generator.unique():
        d = df[(df.generator == gen) & ~df.system.str.contains("_")]
        ok = d.pivot(index="qid", columns="system", values="correct").replace({1: "yes", 0: "no"})
        for s in [c for c in ok.columns if c.startswith(("S4", "S5"))]:
            ok[f"{s} status"] = d[d.system == s].set_index("qid")["status"]
        ok.insert(0, "type", d.drop_duplicates("qid").set_index("qid")["type"])
        parts.append(f"### {gen}: answer judged correct, and evidence status\n\n{ok.to_markdown()}\n")
    return "\n".join(parts)


def figure(summ: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cit = summ[summ.system.isin(["S3", "S4", "S5"])]
    metrics = ["para_hit", "status_correct", "false_grounding", "unsupported_rate"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3), sharey=True)
    for ax, m in zip(axes, metrics):
        for i, gen in enumerate(cit.generator.unique()):
            sub = cit[cit.generator == gen]
            ax.bar(np.arange(len(sub)) + i * 0.38, sub[m], width=0.38, label=gen)
            ax.set_xticks(np.arange(len(sub)) + 0.19, sub.system)
        ax.set_title(m.replace("_", " "))
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main() -> None:
    df = load_rows()
    if df.empty:
        raise SystemExit(f"No runs in {RUNS}. Run eval/run_eval.py first.")
    df.to_csv(OUT / "per_question.csv", index=False)
    summ = summary(df)
    summ.to_csv(OUT / "summary.csv", index=False)
    (OUT / "figures").mkdir(exist_ok=True)
    main_systems = summ[~summ.system.str.contains("_")]
    if {"S3", "S4", "S5"} <= set(main_systems.system):
        figure(main_systems, OUT / "figures" / "main_metrics.png")
    with open(OUT / "tables.md", "w", encoding="utf-8") as f:
        f.write(f"# Results (10 questions)\n\n## Summary\n\n{summ.round(3).to_markdown(index=False)}\n\n")
        f.write(f"## Per question\n\n{outcomes(df)}\n")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
