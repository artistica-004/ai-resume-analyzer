"""Deterministic keyword / skill matching with aliases and evidence.

No LLM calls here, so the same resume + JD always gives the same result.
Every JD requirement gets one status:
  found_exact | found_synonym | implied_by_related_experience | missing
plus the resume snippet that justifies it. ("implied" is set later by
semantic.py, and only with a verified quote.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from .aliases import is_ambiguous_short_form, normalize, surface_forms
from .models import Importance, JDKeyword, JobData, MatchStatus, RequirementMatch, ResumeData
from .resume_structurer import parse_month_year

_STOPWORDS_RAW = [
    "and", "or", "the", "a", "an", "of", "in", "on", "for", "with", "to", "experience", "knowledge",
    "skill", "skills", "strong", "good", "excellent", "proficiency", "proficient", "understanding",
    "familiarity", "familiar", "working", "hands", "using", "ability", "etc", "related", "field",
    "years", "year", "plus", "solid", "basic", "advanced", "concepts", "tools", "such", "as",
]
STOPWORDS = {normalize(w) for w in _STOPWORDS_RAW}

# Order in which evidence is preferred (usage in a bullet beats a skills list)
_KIND_PRIORITY = ["bullet", "title", "project_stack", "skills", "education", "cert", "summary", "other"]


def content_tokens(text: str) -> List[str]:
    """Meaningful normalized tokens of a phrase (stopwords removed)."""
    return [t for t in normalize(text).split() if t not in STOPWORDS and len(t) > 1]


@dataclass(frozen=True)
class EvidenceUnit:
    """One searchable piece of the resume (a bullet, a skill, an education line...)."""

    label: str
    text: str
    kind: str
    norm: str  # " normalized text " (space-padded for whole-word matching)


def _unit(label: str, text: str, kind: str) -> EvidenceUnit:
    return EvidenceUnit(label=label, text=text.strip(), kind=kind, norm=f" {normalize(text)} ")


def build_units(resume: ResumeData) -> List[EvidenceUnit]:
    """Split a resume into evidence units, sorted by evidence priority."""
    units: List[EvidenceUnit] = []
    if resume.summary:
        for sent in re.split(r"(?<=[.!?])\s+", resume.summary):
            if sent.strip():
                units.append(_unit("Summary", sent, "summary"))

    seen = set()
    all_skills = list(resume.skills) + [s for items in resume.skill_groups.values() for s in items]
    for s in all_skills:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            units.append(_unit("Skills", f"Skills: {s}", "skills"))

    for e in resume.experience:
        label = f"Experience – {e.company or e.title}"
        if e.title or e.company:
            units.append(_unit(label, f"{e.title} at {e.company}".strip(), "title"))
        for b in e.bullets:
            units.append(_unit(label, b, "bullet"))

    for p in resume.projects:
        label = f"Project – {p.name}"
        head = p.name + (f" (Tech: {', '.join(p.tech_stack)})" if p.tech_stack else "")
        units.append(_unit(label, head, "project_stack"))
        for b in p.bullets:
            units.append(_unit(label, b, "bullet"))

    for ed in resume.education:
        parts = [f"{ed.degree} {ed.field}".strip(), ed.institution, ed.grade]
        txt = ", ".join(x for x in parts if x)
        if txt:
            units.append(_unit("Education", txt, "education"))

    for c in resume.certifications:
        units.append(_unit("Certifications", c, "cert"))
    for heading, items in resume.other_sections.items():
        for it in items:
            units.append(_unit(heading, it, "other"))

    units.sort(key=lambda u: _KIND_PRIORITY.index(u.kind))  # stable sort keeps order within kind
    return units


def _evidence(u: EvidenceUnit, max_len: int = 200) -> str:
    text = u.text if u.kind == "skills" else f"[{u.label}] {u.text}"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _find(units: List[EvidenceUnit], form: str) -> Optional[EvidenceUnit]:
    """First unit containing `form` as whole words. Ambiguous short forms
    (e.g. 'go', 'ts', 'cv') only count inside the skills list or a tech stack."""
    if not form.strip():
        return None
    strict = is_ambiguous_short_form(form)
    for u in units:
        if strict and u.kind not in ("skills", "project_stack"):
            continue
        if f" {form} " in u.norm:
            return u
    return None


# --------------------------------------------------------------------------- #
# Education                                                                   #
# --------------------------------------------------------------------------- #
_DEGREE_PATTERNS = [
    (4, r" (phd|ph d|doctorate|doctor of philosophy) "),
    (3, r" (master|m tech|mtech|m e|m sc|msc|mca|mba|m s|ms|postgraduate) "),
    (2, r" (bachelor|b tech|btech|b e|be|b sc|bsc|bca|bs|b s|undergraduate) "),
    (1, r" (diploma|associate) "),
]
_DEGREE_WORDS = {normalize(w) for w in [
    "bachelor", "bachelors", "master", "masters", "degree", "b", "m", "tech", "e", "sc", "s", "btech",
    "mtech", "bsc", "msc", "bca", "mca", "mba", "phd", "ph", "d", "diploma", "equivalent", "relevant",
    "similar", "discipline", "preferred", "required", "qualification", "graduate", "undergraduate",
]}
_TECH_FIELDS = {normalize(w) for w in [
    "computer", "information", "software", "electronics", "electrical", "data", "mathematics",
    "statistics", "it", "cse", "ece", "ai", "engineering", "science",
]}


def degree_level(norm_padded: str) -> int:
    """0 = none found, 1 = diploma, 2 = bachelor, 3 = master, 4 = PhD."""
    for level, pattern in _DEGREE_PATTERNS:
        if re.search(pattern, norm_padded):
            return level
    return 0


def _field_ok(req_norm: str, edu_units: List[EvidenceUnit]) -> bool:
    field_tokens = [t for t in content_tokens(req_norm) if t not in _DEGREE_WORDS]
    if "c" in req_norm.split() or "cs" in field_tokens:
        field_tokens.append("computer")
    if not field_tokens:
        return True
    edu_text = " ".join(u.norm for u in edu_units)
    if any(f" {t} " in edu_text for t in field_tokens):
        return True
    if " related " in f" {req_norm} " or " equivalent " in f" {req_norm} ":
        return any(f" {t} " in edu_text for t in _TECH_FIELDS)
    return False


def _match_education(term: str, units: List[EvidenceUnit]) -> Tuple[MatchStatus, str, str]:
    req_norm = normalize(term)
    padded = f" {req_norm} "
    edu_units = [u for u in units if u.kind == "education"]
    if not edu_units:
        return MatchStatus.MISSING, "", ""
    for u in edu_units:
        if padded in u.norm:
            return MatchStatus.FOUND_EXACT, term, _evidence(u)

    req_level = degree_level(padded) or (2 if " degree " in padded else 0)
    if req_level == 0:
        toks = content_tokens(term)
        for u in edu_units:
            if toks and sum(f" {t} " in u.norm for t in toks) / len(toks) >= 0.6:
                return MatchStatus.FOUND_SYNONYM, "education keywords", _evidence(u)
        return MatchStatus.MISSING, "", ""

    best = max(edu_units, key=lambda u: degree_level(u.norm))
    res_level = degree_level(best.norm)
    if res_level >= req_level:
        if _field_ok(req_norm, edu_units):
            return MatchStatus.FOUND_SYNONYM, "degree level and field satisfied", _evidence(best)
        return MatchStatus.IMPLIED, "degree level satisfied, field differs", _evidence(best)
    return MatchStatus.MISSING, "", _evidence(best)


# --------------------------------------------------------------------------- #
# Matching                                                                    #
# --------------------------------------------------------------------------- #
def _suggest(kw: JDKeyword, status: MatchStatus) -> str:
    if status == MatchStatus.FOUND_EXACT:
        return ""
    if status == MatchStatus.FOUND_SYNONYM:
        return f"Matched via a related term. Use the JD's exact wording '{kw.term}' where it is accurate."
    if status == MatchStatus.IMPLIED:
        return f"Only implied. Make '{kw.term}' explicit in a bullet if it genuinely applies."
    if kw.importance == Importance.CRITICAL:
        return (f"Must-have '{kw.term}' was not found. Add it only if you genuinely have it "
                "(use the Gaps step); otherwise expect lower screening odds.")
    return f"'{kw.term}' was not found. Add it only if it is accurate."


_DEGREE_HINT_RE = re.compile(
    r"\b(bachelor'?s?|master'?s?|degree|ph\.?\s?d|diploma|b\.?\s?tech|m\.?\s?tech|"
    r"b\.?\s?e\.?(?=\s|/|$)|m\.?\s?e\.?(?=\s|/|$)|b\.?\s?sc|m\.?\s?sc|bca|mca|mba)\b",
    re.IGNORECASE,
)


def _looks_like_degree(term: str) -> bool:
    """True for education requirements even if the LLM mislabeled their category.

    Deliberately ignores 'MS' alone so tools like 'MS Excel' are not treated as degrees.
    """
    return bool(_DEGREE_HINT_RE.search(term))


def match_keyword(kw: JDKeyword, units: List[EvidenceUnit]) -> RequirementMatch:
    """Match one JD keyword against the resume units."""
    base = {"requirement": kw.term, "importance": kw.importance, "category": kw.category}

    if kw.category == "education" or _looks_like_degree(kw.term):
        status, matched_as, ev = _match_education(kw.term, units)
        return RequirementMatch(**base, status=status, matched_as=matched_as, evidence=ev,
                                suggestion=_suggest(kw, status))

    req_norm = normalize(kw.term)

    # 1) Exact phrase
    u = _find(units, req_norm)
    if u:
        return RequirementMatch(**base, status=MatchStatus.FOUND_EXACT, matched_as=kw.term,
                                evidence=_evidence(u), suggestion="")

    # 2) Alias / acronym expansion
    for form in surface_forms(kw.term):
        if form == req_norm:
            continue
        u = _find(units, form)
        if u:
            return RequirementMatch(**base, status=MatchStatus.FOUND_SYNONYM, matched_as=form,
                                    evidence=_evidence(u),
                                    suggestion=_suggest(kw, MatchStatus.FOUND_SYNONYM))

    # 3) All meaningful words of a multi-word requirement in one unit
    toks = content_tokens(kw.term)
    if len(toks) >= 2:
        for u in units:
            if all(f" {t} " in u.norm for t in toks):
                return RequirementMatch(**base, status=MatchStatus.FOUND_SYNONYM,
                                        matched_as="all words of the requirement",
                                        evidence=_evidence(u),
                                        suggestion=_suggest(kw, MatchStatus.FOUND_SYNONYM))

    return RequirementMatch(**base, status=MatchStatus.MISSING, matched_as="", evidence="",
                            suggestion=_suggest(kw, MatchStatus.MISSING))


def match_requirements(resume: ResumeData, job: JobData) -> List[RequirementMatch]:
    """Match every JD keyword. Falls back to the skill lists if keywords are empty."""
    keywords = list(job.keywords)
    if not keywords:
        keywords = ([JDKeyword(term=s, importance="critical") for s in job.must_have_skills]
                    + [JDKeyword(term=s, importance="nice") for s in job.nice_to_have_skills])
    units = build_units(resume)
    return [match_keyword(kw, units) for kw in keywords]


# --------------------------------------------------------------------------- #
# Years of experience & quote verification                                    #
# --------------------------------------------------------------------------- #
def years_of_experience(resume: ResumeData, today: Optional[date] = None) -> float:
    """Total years across experience entries, merging overlapping periods."""
    intervals: List[Tuple[int, int]] = []
    for e in resume.experience:
        start = parse_month_year(e.start_date, today=today)
        if not start:
            continue
        end = parse_month_year(e.end_date, is_end=True, today=today) if e.end_date else start
        if not end:
            end = start
        a = start[0] * 12 + start[1] - 1
        b = end[0] * 12 + end[1]          # inclusive of the end month
        if b > a:
            intervals.append((a, b))
    intervals.sort()
    total, cur_a, cur_b = 0, None, None
    for a, b in intervals:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a  # type: ignore[operator]
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a  # type: ignore[operator]
    return round(total / 12, 1)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def verify_quote(quote: str, source_text: str, threshold: float = 0.85) -> bool:
    """True if `quote` appears (near-)verbatim in `source_text`.

    Used to reject LLM 'evidence' that was paraphrased or invented.
    """
    q, s = _squash(quote), _squash(source_text)
    if len(q) < 8:
        return False
    if q in s:
        return True
    match = SequenceMatcher(None, s, q, autojunk=False).find_longest_match(0, len(s), 0, len(q))
    return match.size >= threshold * len(q)