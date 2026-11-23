"""Edge cases: the system must give a clear message, never crash.   python -m pytest -q"""
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(Path(__file__).parent)]

from egrag import load_config                                            # noqa: E402
from egrag.generate import normalise_claims                              # noqa: E402
from egrag.highlight import render_evidence                              # noqa: E402
from egrag.ingest import IngestError, clean, load_document, load_folder  # noqa: E402
from egrag.llm import LLM, parse_json                                    # noqa: E402
from egrag.pipeline import EGRAG                                         # noqa: E402
from stubs import FakeLLM, StubEmbedder, StubNLI, StubReranker           # noqa: E402

CFG = load_config()


def two_column_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(50, 60, 560, 90), "A Wide Title That Spans Both Columns Of The Page", fontsize=14)
    for i, y in enumerate((120, 220, 320)):
        page.insert_textbox(pymupdf.Rect(50, y, 290, y + 80), f"Left column paragraph number {i + 1} about annual leave rules.")
        page.insert_textbox(pymupdf.Rect(322, y - 40, 560, y + 40), f"Right column paragraph number {i + 1} about sick pay rules.")
    doc.save(path)


def test_two_column_reading_order(tmp_path):
    two_column_pdf(tmp_path / "paper.pdf")
    paras, _ = load_document(tmp_path / "paper.pdf")
    texts = [p.text.split(" paragraph")[0] for p in paras]
    assert texts == ["A Wide Title That Spans Both Columns Of The Page"] + ["Left column"] * 3 + ["Right column"] * 3


def test_ligatures_and_hyphenation_are_cleaned():
    assert clean("the ﬁrst ﬂoor is suf-\nficient") == "the first floor is sufficient"


def test_blank_scanned_encrypted_and_corrupt_files_raise_clear_errors(tmp_path):
    blank = pymupdf.open(); blank.new_page(); blank.save(tmp_path / "scan.pdf")
    locked = pymupdf.open(); locked.new_page().insert_text((72, 72), "secret text " * 5)
    locked.save(tmp_path / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    for name, reason in [("scan.pdf", "no text"), ("locked.pdf", "password"), ("broken.pdf", "not a readable")]:
        with pytest.raises(IngestError, match=reason):
            load_document(tmp_path / name)
    with pytest.raises(IngestError, match="not found"):
        load_document(tmp_path / "missing.pdf")


def test_folder_skips_bad_files_and_keeps_ids_unique(tmp_path):
    two_column_pdf(tmp_path / "report.pdf")
    (tmp_path / "report.txt").write_text("A plain text policy paragraph that is long enough.\n\nSecond one, also long enough.")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    (tmp_path / "~$lockfile.docx").write_bytes(b"x")
    paras, pdfs, warnings = load_folder(tmp_path)
    assert {p.doc for p in paras} == {"report", "report_2"} and list(pdfs) == ["report"]
    assert len(warnings) == 1 and "broken.pdf" in warnings[0]


def test_file_names_with_spaces_and_symbols_become_safe_ids(tmp_path):
    two_column_pdf(tmp_path / "HR Policy (v2) – final.pdf")
    paras, _ = load_document(tmp_path / "HR Policy (v2) – final.pdf")
    assert paras[0].doc == "HR_Policy_v2_final" and " " not in paras[0].pid


def test_empty_index_and_empty_question_do_not_crash():
    rag = EGRAG([], CFG, StubEmbedder(), StubReranker(), StubNLI())
    assert rag.retriever.search("anything") == [] and rag.retriever.search("   ") == []
    out = rag.answer("How many days of leave?", "S5", FakeLLM({}, {"answer": "General answer."}))
    assert out["status"] == "not_found" and "No evidence found" in out["answer"]


def test_messy_llm_output_is_tolerated():
    assert parse_json("not json at all") == {}
    assert parse_json('["a", "b"]') == {"answer": "['a', 'b']"}
    assert parse_json('{"answer": "x", "claims": [') == {}           # truncated JSON
    assert normalise_claims("one claim") == [{"text": "one claim", "cite": None, "quote": None}]
    assert normalise_claims({"text": "c", "cite": ["[D1-p1-1]"], "quote": 25})[0] == \
        {"text": "c", "cite": "D1-p1-1", "quote": "25"}
    assert normalise_claims(42) == [] and normalise_claims([{"no_text": 1}, None]) == []


def test_highlight_rejects_bad_page(tmp_path):
    two_column_pdf(tmp_path / "p.pdf")
    with pytest.raises(ValueError, match="outside the document"):
        render_evidence(tmp_path / "p.pdf", 5, "x")
    assert render_evidence(tmp_path / "p.pdf", 1, "text that is not on the page", bbox=(50, 120, 290, 200))[:4] == b"\x89PNG"


def test_missing_api_key_gives_clear_message(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("egrag.llm.ROOT", tmp_path)                    # no .env file there
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is missing"):
        LLM("any/model", CFG, cache_dir=tmp_path).chat("s", "u")


def test_run_eval_records_errors_and_continues(tmp_path, monkeypatch):
    import run_eval

    class Broken:
        def chat(self, *a, **k):
            raise RuntimeError("network down")

    monkeypatch.setattr(run_eval, "RUNS", tmp_path)
    rag = EGRAG([], CFG, StubEmbedder(), StubReranker(), StubNLI())
    qs = [{"qid": f"Q{i}", "question": "q?", "type": "single", "answerable": True} for i in range(3)]
    run_eval.run(rag, "S3", "g", Broken(), Broken(), qs, {}, "S3")
    lines = (tmp_path / "g__S3.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all('"status": "error"' in line for line in lines)
