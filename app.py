"""EG-RAG demo interface.   Run:  streamlit run app.py

Upload PDFs or Office files, ask a question, and see each claim with its status and the
exact source text highlighted on the PDF page. Every failure (unreadable file, missing key,
network problem) is shown as a message instead of stopping the app.
"""
import tempfile
from pathlib import Path

import streamlit as st

from egrag import ROOT, load_config
from egrag.highlight import RED, YELLOW, render_evidence
from egrag.ingest import IngestError, load_document, load_folder, safe_id, unique_id
from egrag.llm import LLM
from egrag.pipeline import EGRAG
from egrag.retrieve import load_local_models

BADGE = {"grounded": "🟢 Grounded in your documents", "conflict": "🟠 Conflicting evidence in your documents",
         "contradicted": "🔴 Contradicted by your documents", "not_found": "⚪ No evidence found in your documents",
         "n/a": "ℹ️ Baseline system: no evidence check"}

st.set_page_config(page_title="EG-RAG", layout="wide")
cfg = load_config()


@st.cache_resource(show_spinner="Loading local models (the first run downloads them)…")
def models():
    return load_local_models(cfg)


@st.cache_resource(show_spinner="Reading and indexing documents…")
def build_index(files: tuple[tuple[str, bytes], ...], use_corpus: bool):
    paras, pdfs, warnings, used = [], {}, [], set()
    if use_corpus:
        for folder in (cfg["paths"]["raw"], cfg["paths"]["synthetic"]):
            p, f, w = load_folder(ROOT / folder)
            p = [x for x in p if x.doc in cfg["corpus"]]
            paras, warnings = paras + p, warnings + w
            pdfs |= {d: path for d, path in f.items() if d in cfg["corpus"]}
            used |= {x.doc for x in p}
    tmp = Path(tempfile.mkdtemp())
    for name, data in files:
        path = tmp / Path(name).name
        path.write_bytes(data)
        try:
            p, pdf = load_document(path, unique_id(safe_id(path.stem), used))
        except IngestError as err:
            warnings.append(str(err))
            continue
        paras += p
        if pdf:
            pdfs[p[0].doc] = pdf
    embedder, reranker, nli = models()
    return EGRAG(paras, cfg, embedder, reranker, nli, sentence_check=True), pdfs, {p.pid: p for p in paras}, warnings


st.title("EG-RAG · answers with located evidence")
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF, DOCX, PPTX, XLSX, TXT, MD or HTML files", accept_multiple_files=True)
    use_corpus = st.checkbox("Also use the evaluation corpus (data/raw + data/synthetic)", value=not uploads)
    st.header("Model")
    model = st.selectbox("Generator (via OpenRouter)", list(cfg["models"]["generators"].values()))
    system = st.radio("System", ["S5", "S4", "S3", "S2", "S1"],
                      help="S5 = EG-RAG with guardrails (the proposed system, revised sentence-level verifier); S1-S4 are the baselines.")

files = tuple((u.name, u.getvalue()) for u in uploads or [])
if not files and not use_corpus:
    st.info("Upload at least one document, or tick the evaluation corpus.")
    st.stop()

try:
    rag, pdfs, para_by_id, warnings = build_index(files, use_corpus)
except RuntimeError as err:                       # local models could not be loaded
    st.error(str(err))
    st.stop()
for w in warnings:
    st.warning(f"Skipped: {w}")
if not para_by_id:
    st.error("No readable text was found in the documents. Scanned PDFs need OCR before upload.")
    st.stop()
st.caption(f"{len(para_by_id)} paragraphs indexed from {len({p.doc for p in para_by_id.values()})} documents.")

question = st.text_input("Ask a question", placeholder="How many days of annual leave do I get?").strip()
if not question:
    st.stop()

with st.spinner("Retrieving, answering and verifying…"):
    try:
        result = rag.answer(question, system, LLM(model, cfg))
    except Exception as err:                      # missing key, network, rate limit, model not found
        st.error(f"The answer could not be produced: {err}")
        st.stop()

st.subheader(BADGE.get(result["status"], result["status"]))
st.write(result.get("answer") or "")
if result.get("removed_claims"):
    st.caption(f"{len(result['removed_claims'])} claim(s) removed because no evidence supported them.")

claims = [c for c in result.get("claims", []) if c.get("location")]
if claims:
    st.subheader("Evidence")
    rows = [{"#": i + 1, "claim": c["text"], "status": c["status"] + (" (conflict)" if c.get("conflict") else ""),
             "document": c["location"]["doc"], "page": c["location"]["page"],
             "paragraph": c["location"]["paragraph"], "snippet": c["location"]["snippet"]}
            for i, c in enumerate(claims)]
    st.dataframe(rows, hide_index=True)
    pick = st.selectbox("Show claim on the page", range(len(claims)), format_func=lambda i: f"Claim {i + 1}")
    loc = claims[pick]["location"]
    para = para_by_id.get(loc["pid"])
    if para and loc["doc"] in pdfs and loc["page"]:
        quote = claims[pick].get("quote") or para.text
        colour = YELLOW if claims[pick]["status"] == "grounded" else RED
        try:
            st.image(render_evidence(pdfs[loc["doc"]], loc["page"], quote, para.bbox, colour=colour),
                     caption=f"{loc['doc']} · page {loc['page']} · paragraph {loc['paragraph']}")
        except Exception as err:
            st.warning(f"Could not draw the page ({err}). Source text: {para.text}")
    elif para:
        st.info(f"{loc['doc']} (no page view for this file type): {para.text}")
