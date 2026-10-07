"""Anti-fabrication validator.

Compares a tailored ResumeData with the original and flags any:
  company / title / date / degree / institution / certification
  that does not exist in the original, any skill or tech term that is not
  supported by the original (or by a user-confirmed skill), and any number
  in the summary/bullets that the original never contained.

Fully deterministic, so it is unit-testable and cannot itself hallucinate.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from .aliases import ALIASES, canonical, is_ambiguous_short_form, normalize, surface_forms
from .matcher import content_tokens
from .models import JobData, ResumeData, UserConfirmedSkill, ValidationIssue, ValidationReport
from .resume_structurer import normalize_date

_PLACEHOLDER_RE = re.compile(r"\[add metric[^\]]*\]", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SOFT_KEYS = {"communication", "teamwork", "problem solving", "agile", "user interface",
              "user experience", "application programming interface"}
# (canonical key, detectable forms, all forms) for tech terms
_TECH_TERMS: List[Tuple[str, List[str], List[str]]] = []
for _key, _forms in ALIASES.items():
    if _key in _SOFT_KEYS:
        continue
    _all = sorted({normalize(f) for f in _forms + [_key]}, key=len, reverse=True)
    _detect = [f for f in _all if not is_ambiguous_short_form(f)]
    if _detect:
        _TECH_TERMS.append((_key, _detect, _all))


# --------------------------------------------------------------------------- #
# Reusable helpers (also used by bullets.py and tailor.py)                    #
# --------------------------------------------------------------------------- #
def extract_numbers(text: str) -> Set[str]:
    t = _PLACEHOLDER_RE.sub(" ", text or "")
    return {n.replace(",", "") for n in _NUM_RE.findall(t)}


def unsupported_numbers(text: str, source_text: str) -> List[str]:
    """Numbers in `text` that never appear in `source_text`."""
    allowed = extract_numbers(source_text)
    return sorted(n for n in extract_numbers(text) if n not in allowed)


def unsupported_tech_terms(text: str, source_text: str) -> List[str]:
    """Known tech terms present in `text` but absent (in every alias form) from `source_text`."""
    norm = f" {normalize(text or '')} "
    src = f" {normalize(source_text or '')} "
    out = []
    for key, detect, all_forms in _TECH_TERMS:
        if any(f" {f} " in norm for f in detect) and not any(f" {f} " in src for f in all_forms):
            out.append(key)
    return out


def unsupported_jd_terms(text: str, source_text: str, job: JobData) -> List[str]:
    """JD skill/tool keywords in `text` that the source does not support."""
    norm = f" {normalize(text or '')} "
    src = f" {normalize(source_text or '')} "
    out = []
    for kw in job.keywords:
        if kw.category not in ("hard_skill", "tool", "certification"):
            continue
        forms = surface_forms(kw.term)
        detect = [f for f in forms if not is_ambiguous_short_form(f)]
        if any(f" {f} " in norm for f in detect) and not any(f" {f} " in src for f in forms):
            out.append(kw.term)
    return out


def text_issues(text: str, source_text: str, job: Optional[JobData] = None) -> List[Tuple[str, str]]:
    """All (kind, value) fabrication problems in a free-text field."""
    issues: List[Tuple[str, str]] = [("number", n) for n in unsupported_numbers(text, source_text)]
    seen = set()
    terms = unsupported_tech_terms(text, source_text)
    if job is not None:
        terms += unsupported_jd_terms(text, source_text, job)
    for t in terms:
        if canonical(t) not in seen:
            seen.add(canonical(t))
            issues.append(("skill", t))
    return issues


def allowed_skill_keys(original: ResumeData, confirmed: Sequence[UserConfirmedSkill]) -> Set[str]:
    skills: List[str] = list(original.skills)
    skills += [s for g in original.skill_groups.values() for s in g]
    skills += [t for p in original.projects for t in p.tech_stack]
    skills += [c.skill for c in confirmed]
    return {canonical(s) for s in skills if s}


def skill_supported(skill: str, allowed_keys: Set[str], source_norm_padded: str) -> bool:
    """True if a skill is in the allowed set or literally present in the source."""
    if canonical(skill) in allowed_keys:
        return True
    if any(f" {f} " in source_norm_padded for f in surface_forms(skill)):
        return True
    toks = content_tokens(skill)
    return bool(toks) and all(f" {t} " in source_norm_padded for t in toks)


def source_text(original: ResumeData, confirmed: Sequence[UserConfirmedSkill]) -> str:
    """Everything the tailored resume is allowed to draw facts from."""
    extra = [f"{c.skill}. {c.description}" for c in confirmed]
    return original.full_text() + "\n" + "\n".join(extra)


# --------------------------------------------------------------------------- #
# Main validator                                                              #
# --------------------------------------------------------------------------- #
def _norm_set(values: Iterable[str]) -> Set[str]:
    return {normalize(v) for v in values if v}


def validate(original: ResumeData, tailored: ResumeData,
             confirmed: Optional[Sequence[UserConfirmedSkill]] = None,
             job: Optional[JobData] = None) -> ValidationReport:
    confirmed = list(confirmed or [])
    issues: List[ValidationIssue] = []

    def add(kind: str, value: str, location: str, message: str, severity: str = "error") -> None:
        issues.append(ValidationIssue(kind=kind, value=value, location=location,  # type: ignore[arg-type]
                                      severity=severity, message=message))  # type: ignore[arg-type]

    src = source_text(original, confirmed)
    src_norm = f" {normalize(src)} "

    companies = _norm_set(e.company for e in original.experience)
    titles = _norm_set(e.title for e in original.experience)
    dates = {normalize_date(d) for d in
             [x for e in original.experience for x in (e.start_date, e.end_date)]
             + [x for p in original.projects for x in (p.start_date, p.end_date)]
             + [x for ed in original.education for x in (ed.start_date, ed.end_date)] if d}
    degrees = _norm_set(ed.degree for ed in original.education)
    institutions = _norm_set(ed.institution for ed in original.education)

    # ---- Experience ----
    for e in tailored.experience:
        loc = f"Experience – {e.company or e.title}"
        if e.company and normalize(e.company) not in companies:
            add("company", e.company, loc, "Company not found in the original resume.")
        if e.title and normalize(e.title) not in titles:
            add("title", e.title, loc, "Job title differs from the original resume.")
        for d in (e.start_date, e.end_date):
            if d and normalize_date(d) not in dates:
                add("date", d, loc, "Date not found in the original resume.")
    for p in tailored.projects:
        for d in (p.start_date, p.end_date):
            if d and normalize_date(d) not in dates:
                add("date", d, f"Project – {p.name}", "Date not found in the original resume.")

    # ---- Education & certifications ----
    for ed in tailored.education:
        loc = f"Education – {ed.institution}"
        if ed.degree and normalize(ed.degree) not in degrees:
            add("degree", ed.degree, loc, "Degree not found in the original resume.")
        if ed.institution and normalize(ed.institution) not in institutions:
            add("institution", ed.institution, loc, "Institution not found in the original resume.")
        for d in (ed.start_date, ed.end_date):
            if d and normalize_date(d) not in dates:
                add("date", d, loc, "Date not found in the original resume.")
    for c in tailored.certifications:
        if normalize(c) not in src_norm:
            add("certification", c, "Certifications", "Certification not found in the original resume.")

    # ---- Skills list ----
    keys = allowed_skill_keys(original, confirmed)
    listed = list(tailored.skills) + [s for g in tailored.skill_groups.values() for s in g]
    for s in dict.fromkeys(listed):
        if not skill_supported(s, keys, src_norm):
            add("skill", s, "Skills", "Skill is not supported by the original resume or confirmed skills.")

    # ---- Free text: summary + bullets ----
    texts: List[Tuple[str, str]] = [("Summary", tailored.summary)]
    texts += [(f"Experience – {e.company or e.title}", b) for e in tailored.experience for b in e.bullets]
    texts += [(f"Project – {p.name}", b) for p in tailored.projects for b in p.bullets]
    for loc, text in texts:
        for kind, value in text_issues(text, src, job):
            msg = ("Number does not appear in the original resume." if kind == "number"
                   else "Skill/tool mentioned without support in the original resume.")
            add(kind, value, loc, msg)

    # De-duplicate
    unique, seen = [], set()
    for i in issues:
        key = (i.kind, normalize(i.value), i.location)
        if key not in seen:
            seen.add(key)
            unique.append(i)
    return ValidationReport(passed=not any(i.severity == "error" for i in unique), issues=unique)