"""Reproduce every number in the Evaluation section with one command.

    python run_all.py            # full run (needs OPENROUTER_API_KEY in .env)
    python run_all.py --smoke    # 3 questions per system, to check the set-up first

Steps (each can also be run on its own):
  1. data/make_synthetic.py      build the synthetic staff handbook
  2. data/download_corpus.py     fetch the public circular (D3) and record its SHA-256 hash
  3. eval/check_questions.py     every gold quote must be found in the two PDFs
  4. eval/run_eval.py            S1-S5, 10 questions, two generators
  5. eval/run_eval.py --revised  S4v2 / S5v2 with the sentence-level verifier
  5b. eval/run_eval.py --ablations
  6. eval/metrics.py             tables and figure in results/
  7. eval/agreement.py export    30 judged claims for hand labelling (judge validation)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def step(*args: str) -> None:
    print("\n=== " + " ".join(args))
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    step("data/make_synthetic.py")
    step("data/download_corpus.py")
    step("eval/check_questions.py")
    if smoke:
        step("eval/run_eval.py", "--limit", "3")
        step("eval/metrics.py")
        sys.exit(0)
    step("eval/run_eval.py")
    step("eval/run_eval.py", "--revised")
    step("eval/run_eval.py", "--ablations")
    step("eval/metrics.py")
    step("eval/make_report_figures.py")
    step("eval/agreement.py", "export")
    print("\nDone. Results are in results/. Label results/judge_sample.csv, then run: python eval/agreement.py score")
