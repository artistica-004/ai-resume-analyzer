"""Generate a JD-tailored resume as structured data, with an anti-fabrication guard.

Flow:
  1. LLM rewrites the resume (temperature 0.3) under strict rules.
  2. enforce_structure(): restore every immutable fact (contact, companies,
     titles, dates, education, certifications) from the ORIGINAL, drop any
     invented entries, restore entries the model dropped, filter skills.
  3. scrub_free_text(): any bullet that still contains an unsupported
     number or skill is replaced with its closest original bullet; summary
     sentences with problems are removed.
  4. enforce_length(): cap bullets and trim to the 1-page / 2-page budget.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple

from .aliases import canonical, normalize
from .llm import LLMClient
from .matcher import match_requirements
from .models import (
    JobData,
    MatchStatus,
    ResumeData,
    TailoredResumeResult,
    TailorLLMOutput,
    UserConfirmedSkill,
)
from .prompts import TAILOR_PROMPT
from .resume_structurer import parse_month_year
from .validator import allowed_skill_keys, skill_supported, source_text, text_issues

logger = logging.getLogger(__name__)

PLACEHOLDER_RE = re.compile(r"\[add metric[^\]]*\]", re.IGNORECASE)
TONE_RULES = {
    "concise": "Keep every bullet to ONE line (max ~20 words). Summary max 3 lines.",
    "standard": "Bullets 1-2 lines (max ~30 words). Summary 3-4 lines.",
    "detailed": "Bullets up to 2 full lines (max ~35 words). Summary 4 lines.",
}


def _key(s: str) -> str:
    return normalize(s or "")


def _best_original(text: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None
    return max(candidates, key=lambda c: SequenceMatcher(None, text.lower(), c.lower()).ratio())


def _date_sort_key(e) -> tuple:
    end = parse_month_year(e.end_date, is_end=True) or (0, 0)
    start = parse_month_year(e.start_date) or (0, 0)
    return (end, start)


# --------------------------------------------------------------------------- #
# Guards                                                                      #
# --------------------------------------------------------------------------- #
def enforce_structure(original: ResumeData, tailored: ResumeData,
                      confirmed: Sequence[UserConfirmedSkill]) -> Tuple[ResumeData, List[str], List[str]]:
    """Restore immutable facts from the original. Returns (resume, reverted, notes)."""
    out = tailored.model_copy(deep=True)
    reverted: List[str] = []
    notes: List[str] = []

    out.contact = original.contact.model_copy(deep=True)
    out.education = [e.model_copy(deep=True) for e in original.education]
    out.certifications = list(original.certifications)
    out.detected_headings = []
    out.other_sections = {}
    if original.other_sections:
        notes.append("Sections with non-standard headings were not included: "
                     + ", ".join(original.other_sections)
                     + ". Move important items into Experience/Projects if relevant.")

    # ---- Experience ----
    used = set()
    new_exp = []
    for e in out.experience:
        match = next((o for o in original.experience if id(o) not in used and o.company
                      and _key(o.company) == _key(e.company)), None)
        if match is None:
            match = next((o for o in original.experience if id(o) not in used and o.title
                          and _key(o.title) == _key(e.title)), None)
        if match is None:
            reverted.append(f"Removed an experience entry not in your resume: '{e.title} @ {e.company}'.")
            continue
        used.add(id(match))
        e.company, e.title, e.location = match.company, match.title, match.location
        e.start_date, e.end_date = match.start_date, match.end_date
        if not e.bullets:
            e.bullets = list(match.bullets)
        new_exp.append(e)
    for o in original.experience:
        if id(o) not in used:
            new_exp.append(o.model_copy(deep=True))
            reverted.append(f"Restored experience '{o.title} @ {o.company}' that the AI dropped.")
    new_exp.sort(key=_date_sort_key, reverse=True)
    out.experience = new_exp

    # ---- Projects ----
    src = source_text(original, confirmed)
    src_norm = f" {normalize(src)} "
    keys = allowed_skill_keys(original, confirmed)
    orig_proj = {_key(p.name): p for p in original.projects}
    new_proj, kept_names = [], set()
    for p in out.projects:
        o = orig_proj.get(_key(p.name))
        if o is None or _key(o.name) in kept_names:
            reverted.append(f"Removed a project not in your resume: '{p.name}'.")
            continue
        kept_names.add(_key(o.name))
        p.name, p.link, p.role = o.name, o.link, o.role
        p.start_date, p.end_date = o.start_date, o.end_date
        p.tech_stack = [t for t in p.tech_stack if skill_supported(t, keys, src_norm)] or list(o.tech_stack)
        if not p.bullets:
            p.bullets = list(o.bullets)
        new_proj.append(p)
    out.projects = new_proj
    dropped = [o.name for o in original.projects if _key(o.name) not in kept_names]
    if dropped:
        notes.append("Lower-relevance projects left out: " + ", ".join(dropped) + ".")

    # ---- Skills ----
    def _filter(items: List[str]) -> List[str]:
        kept, seen = [], set()
        for s in items:
            k = canonical(s)
            if k in seen:
                continue
            if skill_supported(s, keys, src_norm):
                seen.add(k)
                kept.append(s)
            else:
                reverted.append(f"Removed unsupported skill '{s}'.")
        return kept

    out.skills = _filter(out.skills) or list(original.skills)
    out.skill_groups = {g: v for g, v in ((g, _filter(v)) for g, v in out.skill_groups.items()) if v}
    have = {canonical(s) for s in out.skills}
    for c in confirmed:
        if canonical(c.skill) not in have:
            out.skills.append(c.skill)
    return out, reverted, notes


def scrub_free_text(original: ResumeData, resume: ResumeData, confirmed: Sequence[UserConfirmedSkill],
                    job: JobData) -> Tuple[ResumeData, List[str]]:
    """Replace any bullet/summary sentence that contains unsupported facts."""
    src = source_text(original, confirmed)
    reverted: List[str] = []

    orig_exp = {_key(e.company) or _key(e.title): e for e in original.experience}
    for e in resume.experience:
        o = orig_exp.get(_key(e.company) or _key(e.title))
        fixed = []
        for b in e.bullets:
            problems = text_issues(b, src, job)
            if problems:
                repl = _best_original(b, o.bullets if o else [])
                reverted.append(f"Reverted a bullet in '{e.company}' (unsupported: "
                                + ", ".join(v for _, v in problems) + ").")
                if repl and repl not in fixed:
                    fixed.append(repl)
            elif b not in fixed:
                fixed.append(b)
        e.bullets = fixed

    orig_proj = {_key(p.name): p for p in original.projects}
    for p in resume.projects:
        o = orig_proj.get(_key(p.name))
        fixed = []
        for b in p.bullets:
            problems = text_issues(b, src, job)
            if problems:
                repl = _best_original(b, o.bullets if o else [])
                reverted.append(f"Reverted a bullet in project '{p.name}' (unsupported: "
                                + ", ".join(v for _, v in problems) + ").")
                if repl and repl not in fixed:
                    fixed.append(repl)
            elif b not in fixed:
                fixed.append(b)
        p.bullets = fixed

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", resume.summary or "") if s.strip()]
    clean = []
    for s in sentences:
        problems = text_issues(s, src, job)
        if problems:
            reverted.append("Removed a summary sentence with unsupported facts ("
                            + ", ".join(v for _, v in problems) + ").")
        else:
            clean.append(s)
    resume.summary = " ".join(clean) if clean else original.summary
    return resume, reverted


def enforce_length(resume: ResumeData, one_page: bool, fresher: bool) -> Tuple[ResumeData, bool, List[str]]:
    """Cap bullets per entry and trim to the word budget for 1 or 2 pages."""
    notes: List[str] = []
    trimmed = False
    cap_exp = 5 if one_page else 7
    cap_proj = (4 if fresher else 3) if one_page else 4
    budget = 560 if one_page else 1050

    for e in resume.experience:
        if len(e.bullets) > cap_exp:
            e.bullets = e.bullets[:cap_exp]
            trimmed = True
    for p in resume.projects:
        if len(p.bullets) > cap_proj:
            p.bullets = p.bullets[:cap_proj]
            trimmed = True

    removed = 0
    while len(resume.full_text().split()) > budget:
        pool = [(0 if fresher else 1, len(x.bullets), x) for x in resume.experience if len(x.bullets) > 2]
        pool += [(1 if fresher else 0, len(x.bullets), x) for x in resume.projects if len(x.bullets) > 1]
        if not pool:
            break
        # Prefer trimming the less important kind first, then the entry with most bullets.
        pool.sort(key=lambda t: (t[0], -t[1]))
        pool[0][2].bullets.pop()  # bullets are already ordered most-relevant first
        removed += 1
        trimmed = True
    if removed:
        notes.append(f"Removed {removed} lower-relevance bullet(s) to fit "
                     f"{'1 page' if one_page else '2 pages'}.")
    if len(resume.full_text().split()) > budget:
        notes.append("The resume may still exceed the page budget; consider trimming further.")
    return resume, trimmed, notes


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #
def tailor_resume(original: ResumeData, job: JobData, confirmed: Sequence[UserConfirmedSkill],
                  client: LLMClient, tone: str = "standard", fresher: bool = False,
                  years: float = 0.0) -> TailoredResumeResult:
    confirmed = list(confirmed)
    one_page = years < 8
    before = match_requirements(original, job)
    have = [{"requirement": m.requirement, "resume_term": m.matched_as} for m in before
            if m.status in (MatchStatus.FOUND_EXACT, MatchStatus.FOUND_SYNONYM)]
    confirmed_keys = {canonical(c.skill) for c in confirmed}
    missing = [m.requirement for m in before if m.status == MatchStatus.MISSING
               and canonical(m.requirement) not in confirmed_keys]

    job_brief = {
        "job_title": job.job_title, "seniority": job.seniority,
        "must_have_skills": job.must_have_skills, "nice_to_have_skills": job.nice_to_have_skills,
        "responsibilities": job.responsibilities[:10], "domain": job.domain,
    }
    user_prompt = (
        f"ORIGINAL RESUME (JSON):\n{original.model_dump_json(exclude={'detected_headings'})}\n\n"
        f"TARGET JOB:\n{json.dumps(job_brief)}\n\n"
        f"REQUIREMENTS THE CANDIDATE HAS (use the JD wording when it is the same thing):\n{json.dumps(have)}\n\n"
        f"MISSING REQUIREMENTS — DO NOT ADD THESE:\n{json.dumps(missing)}\n\n"
        "USER-CONFIRMED SKILLS (may be added, using ONLY the user's description):\n"
        f"{json.dumps([c.model_dump() for c in confirmed])}\n\n"
        f"SETTINGS: page budget = {'1 page' if one_page else '2 pages max'}; "
        f"fresher mode = {fresher}; tone = {tone}. {TONE_RULES.get(tone, TONE_RULES['standard'])}"
    )
    raw = client.chat_json(TAILOR_PROMPT, user_prompt, TailorLLMOutput, temperature=0.3, max_tokens=4000)

    resume, reverted, notes = enforce_structure(original, raw.resume, confirmed)
    resume, reverted2 = scrub_free_text(original, resume, confirmed, job)
    resume, trimmed, trim_notes = enforce_length(resume, one_page, fresher)

    placeholders = []
    for e in resume.experience:
        placeholders += [f"{e.company}: {b}" for b in e.bullets if PLACEHOLDER_RE.search(b)]
    for p in resume.projects:
        placeholders += [f"{p.name}: {b}" for b in p.bullets if PLACEHOLDER_RE.search(b)]

    after = match_requirements(resume, job)
    found_before = {canonical(m.requirement) for m in before if m.status != MatchStatus.MISSING}
    keywords_added = [m.requirement for m in after
                      if m.status != MatchStatus.MISSING and canonical(m.requirement) not in found_before]
    skills_not_added = [m.requirement for m in after if m.status == MatchStatus.MISSING
                        and m.category in ("hard_skill", "tool", "certification")]

    if reverted or reverted2:
        logger.info("Tailor guard reverted %d item(s)", len(reverted) + len(reverted2))
    return TailoredResumeResult(
        resume=resume, change_log=raw.change_log, placeholders_to_fill=placeholders,
        skills_not_added=skills_not_added, keywords_added=keywords_added,
        content_trimmed=trimmed or any("left out" in n for n in notes),
        trim_notes=notes + trim_notes, reverted_items=reverted + reverted2,
    )