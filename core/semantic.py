"""LLM-based relevance judgment, with every quote verified in code.

The LLM answers three narrow questions:
  1. How relevant (0-10) is each experience/project entry to the job?
  2. Is each still-missing requirement implied by related experience?
  3. How well do the candidate's titles/seniority align with the role?

Anything it claims must be backed by a verbatim quote from the resume.
verify_assessment() checks every quote with matcher.verify_quote(); a
relevance rating without verified evidence is capped at 3/10, and an
'implied' claim without verified evidence is discarded.
"""

from __future__ import annotations

import json
import logging
from typing import List, Tuple

from .aliases import canonical
from .llm import LLMClient
from .matcher import verify_quote
from .models import (
    JobData,
    MatchStatus,
    RelevanceItem,
    RequirementMatch,
    ResumeData,
    SemanticAssessment,
)
from .prompts import SEMANTIC_MATCH_PROMPT

logger = logging.getLogger(__name__)

MAX_ENTRY_CHARS = 700
MAX_UNMATCHED = 20
UNVERIFIED_CAP = 3
_RANK = {"critical": 0, "important": 1, "nice": 2}


def build_entries(resume: ResumeData) -> List[Tuple[str, str]]:
    """(kind, text) for each experience then project entry, in resume order."""
    entries: List[Tuple[str, str]] = []
    for e in resume.experience:
        head = f"{e.title} @ {e.company} ({e.start_date} - {e.end_date})"
        entries.append(("EXPERIENCE", f"{head}: " + " ; ".join(e.bullets)))
    for p in resume.projects:
        tech = f" [Tech: {', '.join(p.tech_stack)}]" if p.tech_stack else ""
        entries.append(("PROJECT", f"{p.name}{tech}: " + " ; ".join(p.bullets)))
    return [(kind, text[:MAX_ENTRY_CHARS]) for kind, text in entries]


def unmatched_requirements(matches: List[RequirementMatch]) -> List[str]:
    """Missing skill-type requirements, most important first."""
    missing = [m for m in matches
               if m.status == MatchStatus.MISSING and m.category not in ("education", "certification")]
    missing.sort(key=lambda m: _RANK[m.importance.value])
    return [m.requirement for m in missing][:MAX_UNMATCHED]


def _job_summary(job: JobData) -> str:
    lines = [
        f"Title: {job.job_title or 'unknown'} (seniority: {job.seniority})",
        f"Min years: {job.min_years_experience if job.min_years_experience is not None else 'not stated'}",
        f"Must-have: {', '.join(job.must_have_skills[:15]) or 'n/a'}",
        f"Domain: {job.domain or 'n/a'}",
        "Responsibilities: " + "; ".join(job.responsibilities[:8]),
    ]
    return "\n".join(lines)


def verify_assessment(raw: SemanticAssessment, resume: ResumeData,
                      entries: List[Tuple[str, str]], unmatched: List[str]) -> SemanticAssessment:
    """Pure function: drop or cap every claim whose evidence is not in the resume."""
    source = resume.full_text()

    by_index: dict = {}
    for item in raw.relevance:
        if not 0 <= item.index < len(entries) or item.index in by_index:
            continue
        if item.evidence and not verify_quote(item.evidence, source):
            item = item.model_copy(update={
                "relevance": min(item.relevance, UNVERIFIED_CAP),
                "evidence": "",
                "reason": (item.reason + " [evidence not found in resume; capped]").strip(),
            })
        elif not item.evidence:
            item = item.model_copy(update={"relevance": min(item.relevance, UNVERIFIED_CAP)})
        by_index[item.index] = item
    for i in range(len(entries)):
        if i not in by_index:
            by_index[i] = RelevanceItem(index=i, relevance=0, reason="Not assessed by the model.")

    allowed = {canonical(u): u for u in unmatched}
    implied = []
    for imp in raw.implied:
        key = canonical(imp.requirement)
        if not imp.implied or key not in allowed:
            continue
        if not imp.evidence or not verify_quote(imp.evidence, source):
            logger.info("Discarded an 'implied' claim with unverifiable evidence")
            continue
        implied.append(imp.model_copy(update={"requirement": allowed[key]}))

    return SemanticAssessment(
        relevance=[by_index[i] for i in sorted(by_index)],
        implied=implied,
        title_alignment=raw.title_alignment,
        title_alignment_reason=raw.title_alignment_reason,
    )


def assess(resume: ResumeData, job: JobData, matches: List[RequirementMatch],
           client: LLMClient) -> SemanticAssessment:
    """Run the semantic LLM step and return a verified assessment."""
    entries = build_entries(resume)
    unmatched = unmatched_requirements(matches)
    if not entries and not unmatched:
        return SemanticAssessment()

    entry_block = "\n".join(f"[{i}] {kind} | {text}" for i, (kind, text) in enumerate(entries))
    user_prompt = (
        f"JOB:\n{_job_summary(job)}\n\n"
        f"ENTRIES:\n{entry_block or '(no experience or project entries)'}\n\n"
        f"UNMATCHED REQUIREMENTS:\n{json.dumps(unmatched)}"
    )
    raw = client.chat_json(SEMANTIC_MATCH_PROMPT, user_prompt, SemanticAssessment,
                           temperature=0.1, max_tokens=2500)
    return verify_assessment(raw, resume, entries, unmatched)


def apply_implied(matches: List[RequirementMatch],
                  assessment: SemanticAssessment) -> List[RequirementMatch]:
    """Upgrade 'missing' requirements to 'implied' where the evidence was verified."""
    implied = {canonical(i.requirement): i for i in assessment.implied}
    out: List[RequirementMatch] = []
    for m in matches:
        imp = implied.get(canonical(m.requirement))
        if m.status == MatchStatus.MISSING and imp is not None:
            m = m.model_copy(update={
                "status": MatchStatus.IMPLIED,
                "evidence": f'"{imp.evidence}" ({imp.reason})',
                "suggestion": f"Only implied. Make '{m.requirement}' explicit in a bullet if it genuinely applies.",
            })
        out.append(m)
    return out