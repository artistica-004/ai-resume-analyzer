"""Job description text -> JobData (LLM extraction + rule-based grounding).

After the LLM call, code:
- merges must-have / nice-to-have / tools lists into one weighted keyword list,
- de-duplicates by canonical alias (so "JS" and "JavaScript" become one),
- DROPS any keyword not grounded in the JD text (hallucination guard),
- falls back to regex for years of experience and title words for seniority.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

from .aliases import canonical, normalize, surface_forms
from .llm import LLMClient, truncate_text
from .matcher import content_tokens
from .models import Importance, JDKeyword, JobData
from .prompts import JD_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)

MIN_JD_CHARS = 150
MAX_JD_CHARS = 10_000
MAX_KEYWORDS = 40
_RANK = {"critical": 3, "important": 2, "nice": 1}


class JDError(Exception):
    """User-facing job-description error."""


def _grounded(term: str, category: str, jd_norm: str) -> bool:
    """True if the term (or an alias / all its words) actually appears in the JD."""
    if any(f" {form} " in jd_norm for form in surface_forms(term)):
        return True
    tokens = content_tokens(term)
    if not tokens:
        return False
    present = sum(1 for t in tokens if f" {t} " in jd_norm)
    if category in ("education", "certification"):
        return present / len(tokens) >= 0.5
    return present == len(tokens)


def extract_years(jd_text: str) -> float | None:
    """Regex fallback: '3+ years', '2-4 yrs', '5 to 7 years' -> lowest number."""
    hits = re.findall(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:-|–|to)?\s*\d{0,2}\s*\+?\s*(?:years?|yrs?)",
                      jd_text.lower())
    nums = [int(h) for h in hits if 0 < int(h) < 30]
    return float(min(nums)) if nums else None


def infer_seniority(title: str, years: float | None) -> str:
    t = title.lower()
    if "intern" in t:
        return "intern"
    if re.search(r"\b(junior|jr|entry|graduate|fresher|trainee|associate)\b", t):
        return "entry"
    if re.search(r"\b(lead|principal|staff|head|manager|architect)\b", t):
        return "lead"
    if re.search(r"\b(senior|sr)\b", t):
        return "senior"
    if years is None:
        return "unknown"
    if years < 2:
        return "entry"
    return "mid" if years < 5 else "senior"


def post_process_job(job: JobData, jd_text: str) -> JobData:
    """Ground, merge and de-duplicate the LLM's JD extraction."""
    jd_norm = f" {normalize(jd_text)} "
    pool: Dict[str, JDKeyword] = {}

    def add(term: str, importance: str, category: str, upgrade: bool = True) -> None:
        term = (term or "").strip(" .,;:")
        if not term or len(term) > 60:
            return
        if not _grounded(term, category, jd_norm):
            logger.info("Dropped ungrounded JD keyword (category=%s)", category)
            return
        key = canonical(term)
        existing = pool.get(key)
        if existing is None:
            pool[key] = JDKeyword(term=term, importance=importance, category=category)
        elif upgrade and _RANK[importance] > _RANK[existing.importance.value]:
            existing.importance = Importance(importance)

    for kw in job.keywords:
        add(kw.term, kw.importance.value, kw.category)
    for s in job.must_have_skills:
        add(s, "critical", "hard_skill")
    for s in job.nice_to_have_skills:
        add(s, "nice", "hard_skill", upgrade=False)
    for s in job.tools:
        add(s, "important", "tool", upgrade=False)
    for s in job.education_requirements:
        add(s, "important", "education", upgrade=False)
    for s in job.certifications:
        add(s, "important", "certification", upgrade=False)
    for s in job.soft_skills:
        add(s, "nice", "soft_skill", upgrade=False)

    keywords = sorted(pool.values(), key=lambda k: -_RANK[k.importance.value])
    job.keywords = keywords[:MAX_KEYWORDS]

    # Keep the plain lists consistent with grounded keywords
    grounded_keys = set(pool)
    job.must_have_skills = [s for s in job.must_have_skills if canonical(s) in grounded_keys]
    job.nice_to_have_skills = [s for s in job.nice_to_have_skills if canonical(s) in grounded_keys]

    if job.min_years_experience is None:
        job.min_years_experience = extract_years(jd_text)
    if job.seniority == "unknown":
        job.seniority = infer_seniority(job.job_title, job.min_years_experience)  # type: ignore[assignment]
    return job


def analyze_jd(jd_text: str, client: LLMClient) -> Tuple[JobData, List[str]]:
    """Analyze a job description. Returns (JobData, warnings)."""
    jd_text = (jd_text or "").strip()
    if len(jd_text) < MIN_JD_CHARS:
        raise JDError(
            "The job description is too short. Paste the full JD including requirements "
            f"(at least {MIN_JD_CHARS} characters)."
        )
    warnings: List[str] = []
    text, truncated = truncate_text(jd_text, MAX_JD_CHARS)
    if truncated:
        warnings.append(f"The job description was long; only the first {MAX_JD_CHARS:,} characters were analyzed.")

    job = client.chat_json(JD_ANALYSIS_PROMPT, f"JOB DESCRIPTION:\n<<<\n{text}\n>>>", JobData,
                           temperature=0.1, max_tokens=3000)
    job = post_process_job(job, jd_text)
    if not job.keywords:
        warnings.append("No clear skills or requirements were found in the job description.")
    return job, warnings