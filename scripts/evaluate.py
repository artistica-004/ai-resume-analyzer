"""Run the full pipeline on the sample resumes + JDs and print before/after scores.

Run: python scripts/make_samples.py   (once)
     python scripts/evaluate.py
Uses the Groq API (about 6 calls per pair). Outputs go to eval_output/.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.parser import parse_resume  # noqa: E402
from core.pipeline import analyze, analyze_tailored, compare, generate_tailored, get_client, prepare_inputs  # noqa: E402

DATA = ROOT / "sample_data"
OUT = ROOT / "eval_output"
PAIRS = [
    ("resume_fresher.docx", "jd_ml_engineer.txt", True),
    ("resume_experienced.pdf", "jd_backend_developer.txt", False),
]


def run_pair(client, resume_file: str, jd_file: str, fresher: bool) -> None:
    print(f"\n=== {resume_file}  vs  {jd_file} ===")
    parsed = parse_resume(DATA / resume_file)
    jd_text = (DATA / jd_file).read_text(encoding="utf-8")
    prep = prepare_inputs(parsed, jd_text, client)
    before = analyze(prep.resume, prep.job, client, parsed.quality, parsed.raw_text,
                     n_rewrite=3, fresher=fresher, warnings=prep.warnings)
    bundle = generate_tailored(prep.resume, prep.job, [], client, fresher=fresher)
    after = analyze_tailored(bundle, prep.job, client, fresher=fresher)
    cmp = compare(before, after)

    print(f"{'Category':40s} {'Before':>8s} {'After':>8s}")
    for b, a in zip(before.category_scores, after.category_scores):
        print(f"{b.name:40s} {b.points:8.1f} {a.points:8.1f}")
    print(f"{'OVERALL':40s} {before.overall_score:8.1f} {after.overall_score:8.1f}   ({cmp.delta:+.1f})")
    print("Newly matched:", ", ".join(cmp.newly_matched) or "-")
    print("Still missing:", ", ".join(cmp.still_missing) or "-")
    print("Validation:", "PASSED - no fabricated facts" if bundle.validation.passed
          else f"{len(bundle.validation.issues)} issue(s)")
    print("Guard reverted:", len(bundle.result.reverted_items), "| placeholders:",
          len(bundle.result.placeholders_to_fill))

    stem = Path(resume_file).stem
    (OUT / f"{stem}_tailored.docx").write_bytes(bundle.docx_bytes)
    (OUT / f"{stem}_tailored.pdf").write_bytes(bundle.pdf_bytes)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    client = get_client()
    for i, (r, j, fresher) in enumerate(PAIRS):
        if not (DATA / r).exists():
            sys.exit("Sample resumes missing. Run: python scripts/make_samples.py")
        if i:
            print("\n(waiting 20 s to respect free-tier rate limits…)")
            time.sleep(20)
        run_pair(client, r, j, fresher)
    print(f"\nTailored files saved in {OUT}")


if __name__ == "__main__":
    main()