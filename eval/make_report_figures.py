"""Draw the figures of the report from the logged results (no model calls needed).

    python eval/make_report_figures.py

Figure 2  results/figures/fig_main.png     paragraph hit, span F1, status accuracy and false abstention (S3, S4, S4v2)
Figure 3  results/figures/ev_d3_06.png     evidence view of question D3-06 (circular, page 3)
Figure 4  results/figures/ev_syn10.png     evidence view of the conflict question SYN-10 (handbook, pages 2 and 3)

The evidence views use the same highlight function as the web interface (egrag/highlight.py) and the
location stored in the run records, so they show exactly what the prototype returned.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from egrag.highlight import render_evidence  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ZOOM = 1.6


def figure_main() -> None:
    s = pd.read_csv(RES / "summary.csv")
    systems = ["S3", "S4", "S4v2"]
    labels = ["S3\ncitation-only", "S4\noriginal", "S4v2\nrevised"]
    mets = [("para_hit", "Paragraph hit"), ("span_f1", "Span F1"),
            ("status_correct", "Status correct"), ("false_abstention", "False abstention")]
    colours = {"proprietary": "#2b6cb0", "open_weight": "#dd8452"}
    names = {"proprietary": "GPT-4.1 mini", "open_weight": "Llama 3.3 70B"}
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True)
    x = np.arange(3)
    for ax, (m, title) in zip(axes, mets):
        for k, (g, c) in enumerate(colours.items()):
            v = [float(s[(s.generator == g) & (s.system == sy)][m].iloc[0]) for sy in systems]
            ax.bar(x + (k - .5) * .38, v, .38, color=c, label=names[g])
            for xi, vi in zip(x + (k - .5) * .38, v):
                ax.text(xi, vi + .02, f"{vi:.2f}", ha="center", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0, 1.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, .93))
    plt.savefig(FIG / "fig_main.png", dpi=220)
    plt.close()


def runs(generator: str = "proprietary", system: str = "S4v2") -> dict:
    return {json.loads(l)["qid"]: json.loads(l) for l in open(RES / "runs" / f"{generator}__{system}.jsonl", encoding="utf-8")}


def evidence_crop(pdf: Path, claim: dict, above: int = 120, below: int = 60) -> Image.Image:
    page, quote = claim["location"]["page"], claim["quote"]
    png = render_evidence(pdf, page, quote, zoom=ZOOM)
    doc = pymupdf.open(pdf)
    rects = doc[page - 1].search_for(quote.strip()[:300])
    im = Image.open(io.BytesIO(png))
    if not rects:
        return im
    y0, y1 = min(r.y0 for r in rects) * ZOOM, max(r.y1 for r in rects) * ZOOM
    return im.crop((0, max(0, int(y0 - above)), im.width, min(im.height, int(y1 + below))))


def figure_evidence() -> None:
    r = runs()
    d3 = ROOT / "data" / "raw" / "D3.pdf"
    syn = ROOT / "data" / "synthetic" / "SYN1_staff_handbook.pdf"
    evidence_crop(d3, r["D3-06"]["claims"][0], above=170, below=170).save(FIG / "ev_d3_06.png")
    parts = [evidence_crop(syn, c, a, b) for c, (a, b) in zip(r["SYN-10"]["claims"][:2], [(90, 40), (95, 70)])]
    w, h = parts[0].width, sum(p.height for p in parts) + 24
    sheet = Image.new("RGB", (w, h), "white")
    draw, y = ImageDraw.Draw(sheet), 0
    for k, p in enumerate(parts):
        sheet.paste(p, (0, y))
        y += p.height
        if k == 0:
            draw.line((40, y + 12, w - 40, y + 12), fill=(150, 150, 150), width=2)
            y += 24
    sheet.save(FIG / "ev_syn10.png")


if __name__ == "__main__":
    figure_main()
    figure_evidence()
    print("figures written to", FIG)
