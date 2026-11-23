"""EG-RAG: Evidence-Grounded question answering with located provenance.

Pipeline (one module per step):
    ingest.py      documents -> paragraph units with doc / page / char offsets / bbox
    retrieve.py    BM25 + dense search, fused with RRF, then cross-encoder re-ranking
    llm.py         OpenRouter client with a disk cache (every call is logged)
    generate.py    prompts for the five systems S1-S5
    provenance.py  match each cited quote to its paragraph (location comes from the index)
    verify.py      NLI check of every claim -> evidence status
    pipeline.py    the five systems put together
    highlight.py   draw the evidence on the PDF page for the UI
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
