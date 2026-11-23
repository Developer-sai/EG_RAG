"""Ask questions about any document from the command line (no interface needed).

    python ask.py "C:/Users/me/Downloads/paper.pdf" "What is comprehension debt?"
    python ask.py policies_folder "How many days of annual leave?" --system S4 --save-images out

Prints the evidence status, every claim with document / page / paragraph / snippet, and saves
the highlighted page of each claim as a PNG when --save-images is given.
"""
import argparse
import sys
from pathlib import Path

from egrag import load_config
from egrag.highlight import RED, YELLOW, render_evidence
from egrag.ingest import IngestError, load_document, load_folder
from egrag.llm import LLM
from egrag.pipeline import EGRAG
from egrag.retrieve import load_local_models


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a file or a folder of files")
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--system", default="S5", choices=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--model", default=cfg["models"]["generators"]["proprietary"])
    ap.add_argument("--save-images", metavar="FOLDER")
    args = ap.parse_args()

    src = Path(args.source)
    try:
        if src.is_dir():
            paras, pdfs, warnings = load_folder(src)
            for w in warnings:
                print("skipped:", w)
        else:
            paras, pdf = load_document(src)
            pdfs = {paras[0].doc: pdf} if pdf else {}
    except IngestError as err:
        print("Cannot read the document:", err)
        return 1
    if not paras:
        print("No readable text found.")
        return 1
    print(f"{len(paras)} paragraphs from {len({p.doc for p in paras})} document(s)\n")

    try:
        models = load_local_models(cfg)
    except RuntimeError as err:                     # local models could not be loaded (e.g. offline first run)
        print("Cannot load the local models:", err)
        return 1
    rag = EGRAG(paras, cfg, *models, sentence_check=True)   # revised sentence-level verifier (S4v2 / S5v2)
    by_id = {p.pid: p for p in paras}
    llm = LLM(args.model, cfg)
    for qn, question in enumerate(args.questions, 1):
        try:
            out = rag.answer(question, args.system, llm)
        except Exception as err:
            print(f"Q{qn}: {question}\n  ERROR: {err}\n")
            continue
        print(f"Q{qn}: {question}\n  STATUS: {out['status']}\n  ANSWER: {out.get('answer', '')}")
        for k, c in enumerate(out.get("claims", []), 1):
            loc = c.get("location")
            where = f"{loc['doc']} p.{loc['page']} para {loc['paragraph']}" if loc else "no location"
            print(f"   [{k}] {c['status']:<12} {where}\n       claim:   {c['text']}")
            if loc:
                print(f"       snippet: {loc['snippet']}")
            if loc and args.save_images and loc["doc"] in pdfs and loc["page"]:
                folder = Path(args.save_images)
                folder.mkdir(parents=True, exist_ok=True)
                png = render_evidence(pdfs[loc["doc"]], loc["page"], c.get("quote") or "",
                                      by_id[loc["pid"]].bbox, colour=YELLOW if c["status"] == "grounded" else RED)
                (folder / f"Q{qn}_claim{k}.png").write_bytes(png)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
