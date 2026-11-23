# EG-RAG — Evidence-Grounded Question Answering with Located Provenance

MSc in Artificial Intelligence research project, National College of Ireland.
Lakshmi Ranga Sai Gajulapalli (25205692). Supervisor: Syed Abidi.

Ask a question about your own documents. EG-RAG answers, shows **where** each claim comes from
(document, page, paragraph, highlighted text) and labels every answer with an evidence status:
🟢 grounded · 🟠 conflicting evidence · 🔴 contradicted · ⚪ no evidence found (an unverified general answer is still given).

## How it works (one file per step)

```
upload ─► ingest.py ─► retrieve.py ─► generate.py ─► provenance.py ─► verify.py ─► app.py
          paragraphs    BM25 + dense    LLM writes      quote → doc,     NLI checks     page with the
          with page,    + RRF fusion    claims citing   page, paragraph  each claim;    quote
          bbox, offsets + re-ranker     paragraph ids   (from the index, evidence       highlighted
                                        + exact quotes  never the LLM)   status         (highlight.py)
```

| Step | File | Technique | Why |
|---|---|---|---|
| Ingest | `egrag/ingest.py` | PyMuPDF text blocks; Microsoft MarkItDown for non-PDF; LibreOffice keeps page numbers for Office files | every paragraph keeps its page, position and character offsets |
| Retrieve | `egrag/retrieve.py` | BM25 + MiniLM embeddings, Reciprocal Rank Fusion (k = 60), MiniLM cross-encoder re-ranker | keywords and paraphrases both found; best score feeds the sufficiency gate |
| Generate | `egrag/generate.py`, `egrag/llm.py` | OpenRouter LLM, JSON output, temperature 0, disk cache | the LLM cites a paragraph id and copies a quote; it never writes a page number |
| Provenance | `egrag/provenance.py` | exact, then fuzzy string matching of the quote | location is looked up, so it cannot be hallucinated |
| Verify | `egrag/verify.py` | DeBERTa NLI cross-encoder | entailed → grounded, contradicted, unsupported; disagreement between paragraphs → conflict |
| Guardrails | `egrag/pipeline.py` | sufficiency gate + removal of unsupported claims | with no evidence: "No evidence found in your documents" + labelled general answer |
| Show | `app.py`, `egrag/highlight.py` | Streamlit + PyMuPDF highlight | the user sees the exact text on the page |

The five systems compared in the report are in `egrag/pipeline.py`:
S1 LLM only · S2 naive RAG · S3 citation-only RAG · S4 EG-RAG · S5 EG-RAG with guardrails.

## Set-up (Windows, macOS or Linux, Python 3.10+)

```bash
cd Code
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt    # first install downloads PyTorch (~1-2 GB)
copy .env.example .env             # macOS/Linux: cp .env.example .env
# open .env and paste your OpenRouter key: OPENROUTER_API_KEY=sk-or-...
python -m egrag.llm --check        # confirms the model ids in config.yaml exist on OpenRouter
python -m pytest -q                # 25 offline tests, no key needed
```

The three local models (embedder, re-ranker, NLI) download from Hugging Face on first use (~400 MB).
LibreOffice is optional; without it Office files are read with MarkItDown and have no page numbers.

## Run the demo

```bash
streamlit run app.py
```
Upload PDFs (or tick "use the evaluation corpus"), ask a question, pick a claim to see it highlighted.

Or ask from the command line about any file or folder:

```bash
python ask.py "C:\Users\ranga\Downloads\Generated_Not_Understood.pdf" "What is comprehension debt?" --save-images out
```

## What it handles

* PDF (one or two columns), DOCX/PPTX/XLSX (page numbers kept when LibreOffice is installed), TXT, MD, HTML, CSV.
* Ligatures, hyphenation, running headers and page numbers are cleaned; file names with spaces or symbols are fine.
* Password-protected, scanned (no text layer), empty or corrupt files are skipped with a message.
* Missing API key, network errors, rate limits and malformed model output give a clear message, never a crash;
  in the evaluation a failed question is recorded as an error and the run continues.
* Known limits: scanned PDFs need OCR first; table cells and equations are read as plain text.
  `results/demo_generated_not_understood/report.md` shows a test on a two-column research paper.

## Reproduce the evaluation

The study is small on purpose: **2 PDFs, 10 questions, 2 generator models**, so every answer can be read and checked.

* `D3` – Circular 08/2023, Annual Leave in the Civil Service (public, see `data/sources.csv`)
* `SYN1_staff_handbook` – a synthetic staff handbook of a fictional company, with two clauses that conflict (25 vs 23 days)
* 7 single-passage questions, 1 conflict question, 2 questions the documents cannot answer (`eval/questions.jsonl`)

```bash
python run_all.py --smoke     # 3 questions per system: checks the set-up (~1 minute)
python run_all.py             # S1-S5 x 2 generators, revised verifier S4v2/S5v2, three ablations, tables (a few minutes with the cache, 40-60 without)
```

Outputs in `results/`:

| File | Content |
|---|---|
| `tables.md` | summary of all metrics, and for every question whether each system answered correctly and the evidence status |
| `summary.csv`, `per_question.csv` | the same numbers as tables, one row per question and system |
| `figures/main_metrics.png` | the metrics script's own chart |
| `figures/fig_main.png`, `ev_d3_06.png`, `ev_syn10.png` | Figures 2, 3 and 4 of the report (`python eval/make_report_figures.py`) |
| `runs/*.jsonl` | every answer, claim, location, status and judge label |
| `judge_sample.csv` | 30 claims to label by hand; then `python eval/agreement.py score` gives Cohen's kappa |

Every LLM request and response is stored in `data/cache/`, so a re-run gives identical outputs at no cost.
With 10 questions no significance tests are run; all outcomes are shown question by question.

## Data

* `data/sources.csv` – the public document D3 (Table 1 of the report); `data/download_corpus.py` saves it to
  `data/raw/` and records its SHA-256 in `data/raw/manifest.csv`. If the site blocks the script, save the PDF
  from the browser as `data/raw/D3.pdf` and run the script again.
* `data/make_synthetic.py` – builds the synthetic staff handbook of the fictional Northbridge Analytics Ltd.
* `eval/questions.jsonl` – the 10 questions. Each gold answer points to a verbatim quote; page and paragraph
  are found automatically. Check them with `python eval/check_questions.py`.

## Configuration

All settings (corpus, models, retrieval depth, thresholds, seed = 42) are in `config.yaml`. The thresholds are
conventional values fixed in advance; nothing is tuned on the 10 questions.

## Project layout

```
Code/
├── app.py                Streamlit interface
├── run_all.py            one command for the whole evaluation
├── config.yaml           every setting
├── egrag/                the method (ingest, retrieve, llm, generate, provenance, verify, pipeline, highlight)
├── eval/                 questions, run, metrics, agreement, checks
├── data/                 sources, download and synthetic-document scripts
├── tests/                offline tests with stub models
└── results/              written by the evaluation
```
