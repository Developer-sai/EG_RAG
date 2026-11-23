"""Step 1 - Ingestion: turn uploaded files into paragraph units that know where they are.

A paragraph unit keeps document, page, its position on the page (bbox) and its character
offsets in the document text. Later steps cite only the paragraph id, so the location shown
to the user is always read from here and never written by the LLM.

* PDF            -> PyMuPDF text blocks (one block = one paragraph, with page and bbox);
                    two-column pages are read left column first, then right column
* DOCX/PPTX/XLSX -> converted to PDF with LibreOffice when it is installed (keeps pages),
                    otherwise Microsoft MarkItDown text (no page numbers)
* anything else  -> MarkItDown text (TXT, MD, HTML, CSV, JSON ...), split on blank lines
Files that cannot be read (password-protected, scanned without text, corrupt) raise
IngestError with a plain-English reason, so the interface can skip them and say why.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf

MIN_CHARS = 20          # shorter blocks are page furniture (numbers, labels)
OFFICE = {".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls", ".odt", ".odp", ".ods", ".rtf"}


class IngestError(Exception):
    """A file could not be turned into text; the message says why."""


@dataclass
class Paragraph:
    pid: str            # e.g. "D1-p12-3" = document D1, page 12, 3rd paragraph in reading order
    doc: str            # document id (safe version of the file name)
    page: int | None    # 1-based page number, None if the format has no pages
    text: str
    start: int          # character offset of this paragraph in the document text
    end: int
    bbox: tuple | None  # (x0, y0, x1, y1) on the page, for highlighting

    def as_dict(self) -> dict:
        return asdict(self)


def safe_id(name: str) -> str:
    """File name -> id the LLM can copy reliably (letters, digits, underscores)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:60] or "doc"


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)        # ligatures (ﬁ -> fi), full-width characters
    text = text.replace("­", "").replace("​", "")   # soft hyphens, zero-width spaces
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)        # words split across lines
    return re.sub(r"\s+", " ", text).strip()


def _repeated_lines(pages: list[list[str]]) -> set[str]:
    """Header/footer text = the same short block on more than half of the pages."""
    if len(pages) < 3:
        return set()
    counts = Counter(t for blocks in pages for t in set(blocks) if len(t) < 120)
    return {t for t, n in counts.items() if n > len(pages) / 2}


def reading_order(blocks: list, width: float) -> list:
    """Order blocks as a person reads them.

    On a two-column page, blocks that sit entirely in the left or right half are columns.
    Full-width blocks (titles, wide figures) above the columns come first, then the left
    column top to bottom, then the right column, then any full-width block below them.
    """
    mid = width / 2
    left = [b for b in blocks if b[2] <= mid + 0.05 * width and b[0] < mid]
    right = [b for b in blocks if b[0] >= mid - 0.05 * width and b not in left]
    if len(left) < 2 or len(right) < 2:            # single column: top to bottom
        return sorted(blocks, key=lambda b: (round(b[1]), b[0]))
    wide = [b for b in blocks if b not in left and b not in right]
    body_top = min(b[1] for b in left + right)
    above = [b for b in wide if b[1] < body_top]
    below = [b for b in wide if b[1] >= body_top]
    by_y = lambda bs: sorted(bs, key=lambda b: b[1])
    return by_y(above) + by_y(left) + by_y(right) + by_y(below)


def read_pdf(path: Path, doc_id: str) -> list[Paragraph]:
    try:
        pdf = pymupdf.open(path)
    except Exception as err:
        raise IngestError(f"{path.name}: not a readable PDF ({err})") from err
    if pdf.needs_pass:
        raise IngestError(f"{path.name}: the PDF is password-protected")
    raw = []
    for page in pdf:
        h, w = page.rect.height, page.rect.width
        blocks = [b for b in page.get_text("blocks") if b[6] == 0
                  and b[3] > 0.06 * h and b[1] < 0.94 * h]          # drop running headers / page numbers
        raw.append([(clean(b[4]), tuple(round(v, 1) for v in b[:4])) for b in reading_order(blocks, w)])
    pdf.close()
    furniture = _repeated_lines([[t for t, _ in blocks] for blocks in raw])
    paras, offset = [], 0
    for page_no, blocks in enumerate(raw, start=1):
        n = 0
        for text, bbox in blocks:
            if len(text) < MIN_CHARS or text in furniture:
                continue
            n += 1
            paras.append(Paragraph(f"{doc_id}-p{page_no}-{n}", doc_id, page_no, text,
                                   offset, offset + len(text), bbox))
            offset += len(text) + 1
    if not paras:
        raise IngestError(f"{path.name}: no text found (a scanned PDF needs OCR first)")
    return paras


def read_text(text: str, doc_id: str) -> list[Paragraph]:
    paras, offset, n = [], 0, 0
    for chunk in (clean(c) for c in re.split(r"\n\s*\n", text)):
        if len(chunk) >= MIN_CHARS:
            n += 1
            paras.append(Paragraph(f"{doc_id}-p0-{n}", doc_id, None, chunk, offset, offset + len(chunk), None))
            offset += len(chunk) + 1
    return paras


def office_to_pdf(path: Path) -> Path | None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    out = Path(tempfile.mkdtemp())
    try:
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out), str(path)],
                       check=False, capture_output=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return None
    pdf = out / (path.stem + ".pdf")
    return pdf if pdf.exists() else None


def load_document(path: str | Path, doc_id: str | None = None) -> tuple[list[Paragraph], Path | None]:
    """Return (paragraphs, pdf_path). pdf_path is used by the viewer to highlight evidence."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"{path}: file not found")
    doc_id = doc_id or safe_id(path.stem)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path, doc_id), path
    if suffix in OFFICE:
        pdf = office_to_pdf(path)
        if pdf:
            return read_pdf(pdf, doc_id), pdf
    try:
        from markitdown import MarkItDown          # imported lazily: only needed for non-PDF files
        text = MarkItDown(enable_plugins=False).convert(str(path)).text_content   # plugins off: safer
    except Exception as err:
        raise IngestError(f"{path.name}: this file type could not be read ({type(err).__name__})") from err
    paras = read_text(text or "", doc_id)
    if not paras:
        raise IngestError(f"{path.name}: no text found")
    return paras, None


def load_folder(folder: str | Path) -> tuple[list[Paragraph], dict[str, Path], list[str]]:
    """Read every file in a folder. Returns (paragraphs, pdf paths by doc id, warnings)."""
    paras, pdfs, warnings, used = [], {}, [], set()
    folder = Path(folder)
    if not folder.is_dir():
        return paras, pdfs, warnings
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith((".", "~$")) or path.name == "manifest.csv":
            continue
        doc_id = unique_id(safe_id(path.stem), used)
        try:
            p, pdf = load_document(path, doc_id)
        except IngestError as err:
            warnings.append(str(err))
            continue
        paras += p
        if pdf:
            pdfs[doc_id] = pdf
    return paras, pdfs, warnings


def unique_id(doc_id: str, used: set[str]) -> str:
    """Two files called report.pdf and report.docx must not share an id."""
    base, k = doc_id, 2
    while doc_id in used:
        doc_id, k = f"{base}_{k}", k + 1
    used.add(doc_id)
    return doc_id
