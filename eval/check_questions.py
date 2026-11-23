"""Check the gold annotations against the parsed corpus before any system is run.

    python eval/check_questions.py

For every answerable question, each gold quote must be found in its document; the page and
paragraph are then read from the index (so gold pages never need typing by hand). Questions
drafted from the public documents carry "verified": false until the author has read them
against the PDF; the script lists them so they can be checked and set to true.
"""
import collections

import pandas as pd

from common import CFG, gold_locations, load_corpus, load_questions

from egrag import ROOT


def main() -> None:
    paras, _ = load_corpus()
    docs = {p.doc for p in paras}
    qs = load_questions()
    rows = []
    for q in qs:
        locs = gold_locations(q, paras)
        rows.append({"qid": q["qid"], "type": q["type"], "verified": q.get("verified", False),
                     "gold_quotes": len(q["gold"]), "found": len(locs),
                     "pages": "; ".join(f"{l['doc']} p{l['page']}" for l in locs),
                     "missing_docs": ", ".join(sorted({g["doc"] for g in q["gold"]} - docs))})
    df = pd.DataFrame(rows)
    out = ROOT / CFG["paths"]["results"]
    out.mkdir(exist_ok=True)
    df.to_csv(out / "question_check.csv", index=False)
    answerable = df[df.type != "unanswerable"]
    problems = answerable[answerable.found == 0]
    print(f"{len(qs)} questions: {collections.Counter(q['type'] for q in qs)}")
    print(f"documents in corpus: {sorted(docs)}")
    print(f"answerable questions with no gold quote found: {len(problems)}")
    if len(problems):
        print(problems[["qid", "missing_docs"]].to_string(index=False))
    print(f"not yet verified by the author: {int((~df.verified).sum())}  (see results/question_check.csv)")


if __name__ == "__main__":
    main()
