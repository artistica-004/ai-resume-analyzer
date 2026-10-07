"""Raw resume text -> ResumeData.

The LLM does the heavy lifting of splitting text into sections, then
deterministic code:
- re-extracts contact info with regex (and rejects invented emails),
- normalises dates to "Mon YYYY",
- de-duplicates skills by canonical alias,
- drops any experience entry whose company AND title are absent from the
  source text (extraction-time hallucination guard),
- detects the real section headings for the ATS audit.

Date helpers (normalize_date / parse_month_year) also live here and are
reused by matcher, validator and ats_checker.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from .aliases import canonical, normalize
from .llm import LLMClient, truncate_text
from .models import ResumeData
from .prompts import RESUME_STRUCTURE_PROMPT

logger = logging.getLogger(__name__)

MAX_RESUME_CHARS = 14_000

STANDARD_HEADINGS: Dict[str, List[str]] = {
    "summary": ["summary", "professional summary", "profile", "about me", "objective",
                "career objective", "professional profile"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "skills & tools",
               "technologies", "tech stack", "skill set"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "internships", "internship", "internship experience"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "education": ["education", "academic background", "academics", "educational qualifications",
                  "qualifications"],
    "certifications": ["certifications", "certificates", "licenses & certifications", "courses",
                       "certifications & courses"],
}
KNOWN_HEADINGS = {h for hs in STANDARD_HEADINGS.values() for h in hs}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,16}\d")
URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in/|github\.com/)[\w\-/.%]+|https?://[^\s,|]+",
    re.IGNORECASE,
)

_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_FULL_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december"]
_PRESENT = {"present", "current", "now", "till date", "to date", "ongoing", "today", "currently"}


# --------------------------------------------------------------------------- #
# Dates                                                                       #
# --------------------------------------------------------------------------- #
def _to_year(y: str) -> int:
    return int(y) if len(y) == 4 else 2000 + int(y)


def normalize_date(raw: str) -> str:
    """Convert common date formats to 'Mon YYYY', 'YYYY' or 'Present'.

    Unknown formats are returned unchanged (so we never invent a date).
    """
    s = (raw or "").strip()
    if not s:
        return ""
    low = s.lower().strip(" .")
    if low in _PRESENT:
        return "Present"

    m = re.match(r"^([a-z]{3,9})\.?[\s,'’-]*(\d{2}|\d{4})$", low)
    if m:
        word = m.group(1)
        idx = next((i for i, full in enumerate(_FULL_MONTHS)
                    if full.startswith(word) and len(word) >= 3), None)
        if idx is not None:
            return f"{_MONTHS[idx].title()} {_to_year(m.group(2))}"

    m = re.match(r"^(\d{1,2})[/.-](\d{4}|\d{2})$", low)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{_MONTHS[int(m.group(1)) - 1].title()} {_to_year(m.group(2))}"

    m = re.match(r"^(\d{4})[/.-](\d{1,2})$", low)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{_MONTHS[int(m.group(2)) - 1].title()} {m.group(1)}"

    if re.match(r"^(19|20)\d{2}$", low):
        return low
    return s


def parse_month_year(raw: str, is_end: bool = False,
                     today: Optional[date] = None) -> Optional[Tuple[int, int]]:
    """Return (year, month) for a date string, or None if unparseable."""
    s = normalize_date(raw)
    if not s:
        return None
    if s == "Present":
        t = today or date.today()
        return (t.year, t.month)
    m = re.match(r"^([A-Z][a-z]{2}) (\d{4})$", s)
    if m:
        return (int(m.group(2)), _MONTHS.index(m.group(1).lower()) + 1)
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return (int(m.group(0)), 12 if is_end else 1)
    return None


# --------------------------------------------------------------------------- #
# Headings & contact                                                          #
# --------------------------------------------------------------------------- #
def detect_headings(raw_text: str) -> List[str]:
    """Find section-heading lines in the raw text (rule-based)."""
    headings: List[str] = []
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if i == 0:  # first line is almost always the candidate's name
            continue
        s = line.rstrip(":").strip()
        if not s or len(s) > 40 or s.startswith("-") or len(s.split()) > 5:
            continue
        low = re.sub(r"[^a-z& ]", "", s.lower()).strip()
        letters = [c for c in s if c.isalpha()]
        is_caps = len(letters) >= 4 and all(c.isupper() for c in letters)
        if low in KNOWN_HEADINGS or (is_caps and not EMAIL_RE.search(s) and not re.search(r"\d", s)):
            if s not in headings:
                headings.append(s)
    return headings


def extract_contact_regex(raw_text: str) -> Dict[str, object]:
    """Regex-based contact extraction used to verify / fill the LLM output."""
    head = raw_text[:1500]  # contact info lives near the top
    email = EMAIL_RE.search(raw_text)
    phone = None
    for m in PHONE_RE.finditer(head):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13:
            phone = m.group(0).strip()
            break
    links = []
    for m in URL_RE.finditer(raw_text):
        url = m.group(0).rstrip(".,;)")
        if url not in links:
            links.append(url)
    return {"email": email.group(0) if email else "", "phone": phone or "", "links": links[:4]}


# --------------------------------------------------------------------------- #
# Post-processing                                                             #
# --------------------------------------------------------------------------- #
def _strip_bullet(text: str) -> str:
    return re.sub(r"^[-•*·\s]+", "", text or "").strip()


def _in_source(value: str, source_norm: str) -> bool:
    v = normalize(value)
    return bool(v) and v in source_norm


def post_process(data: ResumeData, raw_text: str) -> Tuple[ResumeData, List[str]]:
    """Deterministic clean-up and hallucination checks on extracted data."""
    warnings: List[str] = []
    source_norm = normalize(raw_text)
    contact = extract_contact_regex(raw_text)

    # Contact: trust regex over the model; never keep an email that isn't in the text
    if data.contact.email and data.contact.email.lower() not in raw_text.lower():
        data.contact.email = ""
    data.contact.email = data.contact.email or str(contact["email"])
    data.contact.phone = data.contact.phone or str(contact["phone"])
    links = [ln for ln in data.contact.links if ln and ln.lower() in raw_text.lower()]
    for ln in contact["links"]:  # type: ignore[union-attr]
        if ln not in links:
            links.append(ln)
    data.contact.links = links

    # Experience: drop entries we cannot find in the source
    kept = []
    for e in data.experience:
        if _in_source(e.company, source_norm) or _in_source(e.title, source_norm):
            e.start_date = normalize_date(e.start_date)
            e.end_date = normalize_date(e.end_date)
            e.bullets = [_strip_bullet(b) for b in e.bullets if _strip_bullet(b)]
            kept.append(e)
        else:
            warnings.append("One extracted experience entry could not be verified in the text and was dropped.")
            logger.warning("Dropped unverifiable experience entry")
    data.experience = kept

    for p in data.projects:
        p.start_date = normalize_date(p.start_date)
        p.end_date = normalize_date(p.end_date)
        p.bullets = [_strip_bullet(b) for b in p.bullets if _strip_bullet(b)]
    for ed in data.education:
        ed.start_date = normalize_date(ed.start_date)
        ed.end_date = normalize_date(ed.end_date)

    # Skills: flatten groups if needed, de-duplicate by canonical alias
    skills = list(data.skills)
    if not skills and data.skill_groups:
        skills = [s for items in data.skill_groups.values() for s in items]
    seen, deduped = set(), []
    for s in skills:
        s = s.strip()
        key = canonical(s)
        if s and key not in seen:
            seen.add(key)
            deduped.append(s)
    data.skills = deduped

    # Headings: rule-based detection is ground truth; keep LLM ones only if verbatim
    headings = detect_headings(raw_text)
    for h in data.detected_headings:
        if h and h in raw_text and h not in headings:
            headings.append(h)
    data.detected_headings = headings
    return data, warnings


def structure_resume(raw_text: str, client: LLMClient) -> Tuple[ResumeData, List[str]]:
    """Convert raw resume text into ResumeData. Returns (data, warnings)."""
    warnings: List[str] = []
    text, truncated = truncate_text(raw_text, MAX_RESUME_CHARS)
    if truncated:
        warnings.append(
            f"Your resume is very long; only the first {MAX_RESUME_CHARS:,} characters were analyzed."
        )
    user_prompt = f"RESUME TEXT:\n<<<\n{text}\n>>>"
    data = client.chat_json(RESUME_STRUCTURE_PROMPT, user_prompt, ResumeData,
                            temperature=0.1, max_tokens=4000)
    data, more = post_process(data, raw_text)
    return data, warnings + more