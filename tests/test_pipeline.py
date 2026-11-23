"""Offline tests:  python -m pytest -q"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(Path(__file__).parent)]

from egrag import load_config                                  # noqa: E402
from egrag.highlight import render_evidence                    # noqa: E402
from egrag.ingest import load_folder, read_text                # noqa: E402
from egrag.llm import parse_json                               # noqa: E402
from egrag.pipeline import EGRAG                               # noqa: E402
from egrag.provenance import find_quote, locate                # noqa: E402
from stubs import FakeLLM, StubEmbedder, StubNLI, StubReranker  # noqa: E402

CFG = load_config()
SYN = ROOT / "data" / "synthetic"
HB = "SYN1_staff_handbook"


@pytest.fixture(scope="module")
def corpus():
    if not any(SYN.glob("*.pdf")):
        import runpy
        runpy.run_path(str(ROOT / "data" / "make_synthetic.py"), run_name="__main__")
    return load_folder(SYN)[:2]


@pytest.fixture(scope="module")
def rag(corpus):
    return EGRAG(corpus[0], CFG, StubEmbedder(), StubReranker(), StubNLI())


def test_ingest_keeps_page_and_removes_headers(corpus):
    paras, pdfs = corpus
    assert HB in pdfs and len(paras) > 5
    assert not any("fictional organisation" in p.text for p in paras)       # running header removed
    leave = next(p for p in paras if "25 working days" in p.text)
    assert (leave.doc, leave.page, leave.pid) == (HB, 2, f"{HB}-p2-1") and leave.bbox


def test_read_text_splits_paragraphs():
    paras = read_text("First paragraph with enough characters.\n\nSecond paragraph, also long enough.", "T")
    assert [p.pid for p in paras] == ["T-p0-1", "T-p0-2"] and paras[0].page is None


def test_find_quote_exact_and_fuzzy():
    text = "Full-time employees are entitled to 25 working days of annual leave per leave year."
    assert find_quote("entitled to 25 working days", text, 0.85)[2] == 1.0
    assert find_quote("entitled to 25 workng days", text, 0.85) is not None     # typo still matches
    assert find_quote("entitled to 40 holidays", text, 0.85) is None


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope this helps') == {"a": 2}


def test_retrieval_finds_relevant_paragraph(rag):
    hits = rag.retriever.search("medical certificate sick leave absence", 3)
    assert hits[0].para.doc == HB and "medical certificate" in hits[0].para.text


def test_s5_grounded_answer_with_location(rag):
    llm = FakeLLM({"Core hours": {"answer": "Core hours are 10:00 to 16:00.", "not_found": False, "claims": [
        {"text": "Core hours are 10:00 to 16:00.", "cite": f"{HB}-p1-2",
         "quote": "Core hours, when all staff must be available, are 10:00 to 16:00."}]}})
    out = rag.answer("What are the core hours when all staff must be available?", "S5", llm)
    c = out["claims"][0]
    assert out["status"] == "grounded"
    assert c["location"]["doc"] == HB and c["location"]["page"] == 1 and c["location"]["method"] == "exact"


def test_wrong_id_is_relocated_and_bad_quote_is_removed(rag):
    llm = FakeLLM({"medical": {"answer": "x", "claims": [
        {"text": "A certificate is needed after 3 days.", "cite": f"{HB}-p1-1",           # wrong id
         "quote": "A medical certificate is required for absences longer than three consecutive days."},
        {"text": "Sick pay lasts 40 days.", "cite": f"{HB}-p3-2", "quote": "sick pay lasts forty days"}]}})  # invented
    out = rag.answer("When is a medical certificate required for sick leave?", "S5", llm)
    assert len(out["claims"]) == 1 and out["claims"][0]["location"]["method"] == "relocated"
    assert out["removed_claims"] == ["Sick pay lasts 40 days."]


def test_conflict_is_detected(rag):
    q = "How many working days of annual leave do full-time employees get?"
    llm = FakeLLM({"annual leave": {"answer": "25 days.", "claims": [
        {"text": "Full-time employees get 25 working days of annual leave.", "cite": f"{HB}-p2-1",
         "quote": "Full-time employees are entitled to 25 working days of annual leave per leave year"}]}})
    out = rag.answer(q, "S5", llm)
    assert out["status"] == "conflict" and out["claims"][0]["conflict_with"] == f"{HB}-p3-4"


def test_not_found_still_answers_with_label(rag):
    llm = FakeLLM({}, default={"answer": "Many firms offer this.", "claims": []})
    out = rag.answer("Does the company pay for gym membership?", "S5", llm)
    assert out["status"] == "not_found" and out["answer"].startswith("No evidence found in your documents.")
    assert "unverified" in out["answer"]


def test_s3_trusts_cited_id_without_checking(rag):
    llm = FakeLLM({"Core": {"answer": "x", "claims": [{"text": "Core hours are 9 to 5.", "cite": f"{HB}-p1-2"}]}})
    out = rag.answer("What are the Core hours?", "S3", llm)
    assert out["status"] == "grounded" and out["claims"][0]["location"]["method"] == "cited"


def test_s1_and_s2_have_no_locations(rag):
    llm = FakeLLM({}, default={"answer": "25 days.", "claims": ["You get 25 days."]})
    for s in ("S1", "S2"):
        out = rag.answer("How much annual leave?", s, llm)
        assert out["status"] == "n/a" and out["claims"][0]["location"] is None


def test_highlight_renders_png(corpus):
    _, pdfs = corpus
    png = render_evidence(pdfs[HB], 2, "25 working days of annual leave")
    assert png[:4] == b"\x89PNG"


def test_metrics_scoring():
    from metrics import score, span_f1
    assert span_f1((0, 10), (5, 15)) == pytest.approx(0.5)
    q = {"qid": "Q", "type": "single", "answerable": True}
    gold = [{"pid": "D-p1-1", "doc": "D", "page": 1, "start": 0, "end": 10}]
    rec = {"system": "S5", "name": "S5", "generator": "g", "correct": True, "status": "grounded", "claims": [
        {"status": "grounded", "judge": "supported",
         "location": {"pid": "D-p1-1", "doc": "D", "page": 1, "start": 0, "end": 10}}]}
    row = score(rec, q, gold)
    assert row["para_hit"] == 1 and row["span_f1"] == 1.0 and row["false_grounding"] == 0
    rec["claims"][0]["judge"] = "unsupported"
    assert score(rec, q, gold)["false_grounding"] == 1


def test_eval_scripts_end_to_end(tmp_path, monkeypatch):
    """run_eval -> metrics on the synthetic questions with stub models (checks the plumbing only)."""
    import metrics
    import run_eval

    class JudgeOrGen(FakeLLM):
        def chat(self, system, user, json_mode=True):
            if "grade answers" in system:
                return {"text": "", "json": {"correct": True}, "model": "judge", "latency_s": 0, "usage": None}
            if "check whether a claim" in system:
                return {"text": "", "json": {"label": "supported"}, "model": "judge", "latency_s": 0, "usage": None}
            return super().chat(system, user, json_mode)

    monkeypatch.setattr(run_eval, "load_local_models", lambda cfg: (StubEmbedder(), StubReranker(), StubNLI()))
    monkeypatch.setattr(run_eval, "LLM", lambda model, cfg: JudgeOrGen({}, {"answer": "n/a", "claims": []}))
    monkeypatch.setattr(run_eval, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(metrics, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(metrics, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--generators", "proprietary"])
    run_eval.main()
    monkeypatch.setattr(sys, "argv", ["metrics.py"])
    metrics.main()
    summary = (tmp_path / "summary.csv").read_text()
    assert "S5" in summary and "Per question" in (tmp_path / "tables.md").read_text()


def test_sentence_helpers():
    from egrag.verify import best_sentence, evidence_for
    para = "Staff receive 25 days of leave. Requests go to the manager. Unused days expire in March."
    assert evidence_for(para, "Staff receive 25 days of leave") == "Staff receive 25 days of leave."
    assert best_sentence(para, "unused days expire in March") == "Unused days expire in March."
    assert best_sentence(para, "the parking policy covers bicycles") is None
    assert evidence_for(para, "") == para
