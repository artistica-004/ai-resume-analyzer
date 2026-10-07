"""Write pinned, UTF-8 requirements files from the CURRENT working venv.

Run: python scripts/pin_requirements.py
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ["streamlit", "groq", "pydantic", "pdfplumber", "python-docx", "python-dotenv", "reportlab", "pandas"]
DEV = ["pytest"]


def pins(packages):
    lines = []
    for p in packages:
        try:
            lines.append(f"{p}=={version(p)}")
        except PackageNotFoundError:
            raise SystemExit(f"'{p}' is not installed in this venv. Run: pip install {p}")
    return lines


runtime = pins(RUNTIME)
(ROOT / "requirements.txt").write_text("\n".join(runtime) + "\n", encoding="utf-8")
(ROOT / "requirements-dev.txt").write_text("-r requirements.txt\n" + "\n".join(pins(DEV)) + "\n",
                                           encoding="utf-8")
print("requirements.txt:\n  " + "\n  ".join(runtime))
print(f"\nStreamlit version (use as sdk_version in README.md): {version('streamlit')}")