"""Run the systems on the question set and judge the outputs.

    python eval/run_eval.py                       # S1-S5, all 10 questions, both generators
    python eval/run_eval.py --revised             # S4v2 / S5v2: revised verifier (sentence-level)
    python eval/run_eval.py --ablations           # S5 with one component switched off at a time
    python eval/run_eval.py --limit 5             # quick smoke test

Writes one JSON line per question to results/runs/<generator>__<system>.jsonl.
LLM calls are cached, so re-running is free and gives identical outputs.
"""
from __future__ import annotations

import argparse
import json
import time

from common import CFG, RUNS, corpus_fingerprint, judge_answer, judge_claim, load_corpus, load_questions

from egrag.llm import LLM
from egrag.pipeline import EGRAG, SYSTEMS
from egrag.retrieve import load_local_models

ABLATIONS = {"no_quote": {"use_quote": False}, "no_verifier": {"use_verifier": False},
             "no_gate": {"use_gate": False}}


def run(rag: EGRAG, system: str, gen_tag: str, llm: LLM, judge: LLM, questions: list[dict],
        paras_by_id: dict, name: str) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{gen_tag}__{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            t0 = time.time()
            try:
                out = rag.answer(q["question"], system, llm)
                out["correct"] = judge_answer(judge, q, out.get("answer", ""))
                for c in out.get("claims", []):                   # judge every located claim
                    if c.get("location"):
                        c["judge"] = judge_claim(judge, c["text"], paras_by_id[c["location"]["pid"]].text)
            except Exception as err:                              # keep going; the error is recorded
                out = {"system": system, "status": "error", "error": str(err), "claims": [], "correct": False}
                print(f"\n[{q['qid']}] {err}")
            out["wall_s"] = round(time.time() - t0, 2)
            out.update(qid=q["qid"], type=q["type"], answerable=q["answerable"])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"\r{name:>14} {gen_tag}: {i}/{len(questions)}", end="", flush=True)
    print(f"  -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--generators", nargs="+", default=list(CFG["models"]["generators"]))
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--revised", action="store_true", help="S4v2 and S5v2: sentence-level verifier")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    paras, _ = load_corpus()
    questions = load_questions()
    questions = questions[: args.limit]
    print(f"{len(paras)} paragraphs (corpus {corpus_fingerprint(paras)}), {len(questions)} questions")

    embedder, reranker, nli = load_local_models(CFG)
    judge = LLM(CFG["models"]["judge"], CFG)
    by_id = {p.pid: p for p in paras}
    for tag in args.generators:
        llm = LLM(CFG["models"]["generators"][tag], CFG)
        if args.ablations:
            for name, flags in ABLATIONS.items():
                run(EGRAG(paras, CFG, embedder, reranker, nli, **flags), "S5", tag, llm, judge, questions, by_id,
                    f"S5_{name}")
        elif args.revised:
            rag = EGRAG(paras, CFG, embedder, reranker, nli, sentence_check=True)
            for s in ("S4", "S5"):
                run(rag, s, tag, llm, judge, questions, by_id, f"{s}v2")
        else:
            rag = EGRAG(paras, CFG, embedder, reranker, nli)
            for s in args.systems:
                run(rag, s, tag, llm, judge, questions, by_id, s)


if __name__ == "__main__":
    main()
