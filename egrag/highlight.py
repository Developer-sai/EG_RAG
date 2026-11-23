"""Step 6 - Show the evidence: render the cited PDF page with the quote highlighted.

PyMuPDF's search_for() returns the exact rectangles of the quote on the page. If the quote
wraps oddly and is not found, the whole paragraph box (stored at ingestion) is outlined instead.
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

YELLOW, RED = (1, 0.85, 0), (0.9, 0.2, 0.2)


def render_evidence(pdf_path: str | Path, page: int, quote: str, bbox: tuple | None = None,
                    zoom: float = 1.6, colour=YELLOW) -> bytes:
    """Return a PNG of the page (1-based) with the quote highlighted."""
    doc = pymupdf.open(pdf_path)
    if not page or not 1 <= page <= doc.page_count:
        count = doc.page_count
        doc.close()
        raise ValueError(f"page {page} is outside the document (1-{count})")
    pg = doc[page - 1]
    rects = pg.search_for(quote.strip()[:300]) if quote else []
    if not rects and quote:                       # try the first half: quotes can span odd line breaks
        rects = pg.search_for(quote.strip()[: max(20, len(quote) // 2)])
    if rects:
        annot = pg.add_highlight_annot(rects)
        annot.set_colors(stroke=colour)
        annot.update()
    elif bbox and pymupdf.Rect(bbox).is_valid and not pymupdf.Rect(bbox).is_empty:
        annot = pg.add_rect_annot(pymupdf.Rect(bbox))
        annot.set_colors(stroke=colour)
        annot.set_border(width=2)
        annot.update()
    png = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")
    doc.close()
    return png
