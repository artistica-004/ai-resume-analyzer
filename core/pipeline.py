"""Orchestration: the only module app.py and scripts need to call.

parse resume -> parse JD -> match + gap analysis -> (user confirms skills)
-> tailor -> validate -> render DOCX/PDF -> re-analyze tailored -> compare
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .aliases import canonical
from .ats_checker import audit
from .bullets import review_all, rewrite_weakest
from .jd_analyzer import analyze_jd
from .llm import LLMClient
from .matcher import match_requirements, years_of_experience
from .models import (
    AnalysisReport,
    ComparisonReport,
    JobData,
    MatchStatus,
    ParsedResume,
    ParseQuality,
    ResumeData,
    TailoredResumeResult,
    UserConfirmedSkill,
    ValidationReport,
)
from .parser import ResumeParseError, parse_resume_bytes
from .renderer import render_docx, render_pdf
from .resume_structurer import structure_resume
from .scorer import compute_report
from .semantic import apply_implied, assess
from .tailor import tailor_resume
from .validator import validate

logger = logging.getLogger(__name__)


@dataclass
class PreparedInputs:
    resume: ResumeData
    job: JobData
    warnings: List[str] = field(default_factory=list)


@dataclass
class TailoredBundle:
    result: TailoredResumeResult
    validation: ValidationReport
    docx_bytes: bytes
    pdf_bytes: bytes


def get_client() -> LLMClient:
    return LLMClient()


def prepare_inputs(parsed: ParsedResume, jd_text: str, client: LLMClient) -> PreparedInputs:
    """Structure the resume and the JD (2 LLM calls)."""
    resume, w1 = structure_resume(parsed.raw_text, client)
    job, w2 = analyze_jd(jd_text, client)
    return PreparedInputs(resume=resume, job=job, warnings=w1 + w2)


def analyze(resume: ResumeData, job: JobData, client: Optional[LLMClient], quality: Optional[ParseQuality],
            raw_text: str, n_rewrite: int = 6, fresher: bool = False,
            warnings: Optional[List[str]] = None) -> AnalysisReport:
    """Full hybrid analysis of one resume against one JD."""
    matches = match_requirements(resume, job)
    if client is not None:
        semantic = assess(resume, job, matches, client)
        matches = apply_implied(matches, semantic)
    else:
        from .models import SemanticAssessment
        semantic = SemanticAssessment()
    years = years_of_experience(resume)
    ats = audit(resume, quality, raw_text, years)
    reviews = review_all(resume, job)
    if n_rewrite > 0 and client is not None:
        reviews = rewrite_weakest(reviews, n_rewrite, resume, job, client)
    return compute_report(resume=resume, job=job, matches=matches, semantic=semantic, ats=ats,
                          bullet_reviews=reviews, quality=quality, years=years,
                          warnings=list(warnings or []), fresher=fresher)


def generate_tailored(original: ResumeData, job: JobData, confirmed: Sequence[UserConfirmedSkill],
                      client: LLMClient, tone: str = "standard", fresher: bool = False) -> TailoredBundle:
    """Tailor, validate and render the new resume."""
    years = years_of_experience(original)
    result = tailor_resume(original, job, confirmed, client, tone=tone, fresher=fresher, years=years)
    validation = validate(original, result.resume, confirmed, job)
    return TailoredBundle(result=result, validation=validation,
                          docx_bytes=render_docx(result.resume, fresher),
                          pdf_bytes=render_pdf(result.resume, fresher))


def analyze_tailored(bundle: TailoredBundle, job: JobData, client: Optional[LLMClient],
                     fresher: bool = False) -> AnalysisReport:
    """Re-analyze the tailored resume by RE-PARSING its own PDF (proves it is ATS-readable)."""
    try:
        parsed = parse_resume_bytes(bundle.pdf_bytes, "tailored_resume.pdf")
        quality, raw = parsed.quality, parsed.raw_text
    except ResumeParseError:
        logger.warning("Could not re-parse the generated PDF; using structured text instead")
        quality, raw = None, bundle.result.resume.full_text()
    return analyze(bundle.result.resume, job, client, quality, raw, n_rewrite=0, fresher=fresher)


def compare(before: AnalysisReport, after: AnalysisReport) -> ComparisonReport:
    prev = {canonical(m.requirement): m for m in before.requirement_matches}
    newly = [m.requirement for m in after.requirement_matches
             if m.status != MatchStatus.MISSING
             and canonical(m.requirement) in prev
             and prev[canonical(m.requirement)].status == MatchStatus.MISSING]
    still = [m.requirement for m in after.requirement_matches if m.status == MatchStatus.MISSING]
    return ComparisonReport(before=before, after=after,
                            delta=round(after.overall_score - before.overall_score, 1),
                            newly_matched=newly, still_missing=still)