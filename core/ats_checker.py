"""ATS format / parse-ability lint rules (pure code, no LLM).

Each issue has a severity:
  error   -> -2 points (out of 10)
  warning -> -1 point
  info    ->  0 points (advice only)
"""

from __future__ import annotations

import math
import re
from typing import List, Optional

from .models import ATSIssue, ATSReport, ParseQuality, ResumeData
from .resume_structurer import KNOWN_HEADINGS, parse_month_year

ERROR_PENALTY = 2.0
WARNING_PENALTY = 1.0
MAX_POINTS = 10.0

# Emoji, dingbats, private-use icon fonts (FontAwesome etc.), geometric shapes, arrows
_ODD_CHARS_RE = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\ue000-\uf8ff\u25a0-\u25ff\u2190-\u21ff]"
)
_DATE_OK_RE = re.compile(r"^([A-Z][a-z]{2} \d{4}|\d{4}|Present)$")
_MONTH_DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{4}$")


def _issue(severity: str, code: str, message: str, evidence: str = "") -> ATSIssue:
    return ATSIssue(severity=severity, code=code, message=message, evidence=evidence)  # type: ignore[arg-type]


def points_from_issues(issues: List[ATSIssue]) -> float:
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    return max(0.0, MAX_POINTS - ERROR_PENALTY * errors - WARNING_PENALTY * warnings)


def audit(resume: ResumeData, quality: Optional[ParseQuality], raw_text: str,
          years: float) -> ATSReport:
    """Run all ATS lint rules and return a report with a 0-100 percentage."""
    issues: List[ATSIssue] = []
    c = resume.contact

    # ---- Contact ----
    if not c.name:
        issues.append(_issue("error", "MISSING_NAME", "No name detected at the top of the resume."))
    if not c.email:
        issues.append(_issue("error", "MISSING_EMAIL", "No email address found."))
    if not c.phone:
        issues.append(_issue("warning", "MISSING_PHONE", "No phone number found."))
    if not c.location:
        issues.append(_issue("info", "MISSING_LOCATION",
                             "No city/location found; recruiters often filter by location."))
    if not any(("linkedin" in ln.lower() or "github" in ln.lower()) for ln in c.links):
        issues.append(_issue("info", "MISSING_PROFILE_LINKS", "No LinkedIn or GitHub URL found."))

    # ---- Sections ----
    if not resume.skills and not resume.skill_groups:
        issues.append(_issue("error", "MISSING_SKILLS", "No Skills section detected."))
    if not resume.experience and not resume.projects:
        issues.append(_issue("error", "MISSING_EXPERIENCE", "No Experience or Projects section detected."))
    if not resume.education:
        issues.append(_issue("warning", "MISSING_EDUCATION", "No Education section detected."))
    if not resume.summary:
        issues.append(_issue("info", "MISSING_SUMMARY", "No Summary; a 3-4 line targeted summary helps."))

    # ---- Headings ----
    odd_headings = []
    for h in resume.detected_headings:
        low = re.sub(r"[^a-z& ]", "", h.lower()).strip()
        if low and low not in KNOWN_HEADINGS:
            odd_headings.append(h)
    for h in odd_headings[:5]:
        issues.append(_issue("info", "NONSTANDARD_HEADING",
                             f"Heading '{h}' is non-standard. Prefer Summary, Skills, Experience, "
                             "Projects, Education, Certifications.", h))

    # ---- Length / pages ----
    words = len(raw_text.split())
    page_limit = 1 if years < 8 else 2
    pages = quality.num_pages if quality and quality.num_pages else max(1, math.ceil(words / 600))
    if words < 250:
        issues.append(_issue("warning", "TOO_SHORT", f"Only about {words} words; the resume looks thin."))
    if pages > page_limit + 1:
        issues.append(_issue("error", "TOO_MANY_PAGES",
                             f"{pages} pages; aim for {page_limit} at your experience level."))
    elif pages > page_limit:
        issues.append(_issue("warning", "TOO_MANY_PAGES",
                             f"{pages} pages; aim for {page_limit} at your experience level."))
    elif words > 1100 and years < 8:
        issues.append(_issue("warning", "TOO_DENSE", f"About {words} words; consider trimming."))

    # ---- Characters ----
    odd = _ODD_CHARS_RE.findall(raw_text)
    if odd:
        sample = "".join(sorted(set(odd)))[:10]
        issues.append(_issue("warning", "UNUSUAL_CHARACTERS",
                             "Icons/emoji/symbol-font characters found; ATS may render them as garbage.",
                             sample))

    # ---- Dates ----
    bad_dates, missing_dates, precisions = [], [], set()
    for e in resume.experience:
        if not e.start_date:
            missing_dates.append(e.company or e.title)
        for d in (e.start_date, e.end_date):
            if d and not _DATE_OK_RE.match(d):
                bad_dates.append(d)
            if d and d != "Present":
                precisions.add("month" if _MONTH_DATE_RE.match(d) else "year")
        start = parse_month_year(e.start_date)
        end = parse_month_year(e.end_date, is_end=True)
        if start and end and end < start:
            issues.append(_issue("warning", "DATE_ORDER",
                                 f"End date is before start date for '{e.company or e.title}'.",
                                 f"{e.start_date} - {e.end_date}"))
    for p in resume.projects:
        for d in (p.start_date, p.end_date):
            if d and not _DATE_OK_RE.match(d):
                bad_dates.append(d)
    for ed in resume.education:
        for d in (ed.start_date, ed.end_date):
            if d and not _DATE_OK_RE.match(d):
                bad_dates.append(d)
    if bad_dates:
        issues.append(_issue("warning", "INCONSISTENT_DATES",
                             "Some dates use an unusual format. Use 'Mar 2024 - Present'.",
                             ", ".join(bad_dates[:4])))
    if missing_dates:
        issues.append(_issue("warning", "MISSING_DATES", "Some experience entries have no dates.",
                             ", ".join(missing_dates[:3])))
    if len(precisions) > 1:
        issues.append(_issue("info", "MIXED_DATE_PRECISION",
                             "Mix of 'Mon YYYY' and 'YYYY' dates; keep one style."))

    starts = [parse_month_year(e.start_date) for e in resume.experience]
    valid = [s for s in starts if s]
    if len(valid) >= 2 and valid != sorted(valid, reverse=True):
        issues.append(_issue("warning", "NOT_REVERSE_CHRONOLOGICAL",
                             "Experience is not in reverse-chronological order (latest first)."))

    # ---- Bullets ----
    empty = [e.company or e.title for e in resume.experience if not e.bullets]
    if empty:
        issues.append(_issue("warning", "EXPERIENCE_WITHOUT_BULLETS",
                             "Some roles have no bullet points.", ", ".join(empty[:3])))
    long_bullets = [b for b in resume.all_bullets() if len(b.split()) > 40]
    if long_bullets:
        issues.append(_issue("info", "LONG_BULLETS",
                             f"{len(long_bullets)} bullet(s) exceed ~2 lines.", long_bullets[0][:80]))

    # ---- Parse quality ----
    if quality:
        if quality.multi_column_suspected:
            issues.append(_issue("warning", "MULTI_COLUMN", "Multi-column layout detected."))
        if quality.has_tables:
            issues.append(_issue("warning", "TABLES", "Tables detected; content may be read out of order."))
        if quality.has_images:
            issues.append(_issue("info", "IMAGES", "Images/graphics detected; ATS ignores them."))
        if quality.header_footer_text:
            issues.append(_issue("warning", "HEADER_FOOTER_CONTACT",
                                 "Contact info is inside the page header/footer."))
        if any("Text boxes" in w for w in quality.warnings):
            issues.append(_issue("warning", "TEXT_BOXES", "Text boxes detected; ATS often skips them."))

    points = points_from_issues(issues)
    return ATSReport(issues=issues, score_percent=round(points / MAX_POINTS * 100, 1))