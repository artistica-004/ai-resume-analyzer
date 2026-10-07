"""All Pydantic models used across the pipeline.

Conventions:
- Scores shown to users are always 0-100.
- Models returned by the LLM are tolerant (defaults everywhere) so that a
  missing optional field never crashes the app.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
class ParseQuality(BaseModel):
    """Signals about how well the text extraction went."""

    method: str = ""                     # e.g. "pdfplumber", "python-docx"
    num_pages: int = 0
    char_count: int = 0
    word_count: int = 0
    is_scanned: bool = False             # image-only PDF, no text layer
    has_tables: bool = False
    has_images: bool = False
    multi_column_suspected: bool = False
    header_footer_text: bool = False     # important info found in header/footer
    warnings: List[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    """Output of parser.parse_resume(). Never None."""

    raw_text: str
    file_name: str
    file_type: Literal["pdf", "docx"]
    quality: ParseQuality


# --------------------------------------------------------------------------- #
# Resume structure                                                            #
# --------------------------------------------------------------------------- #
class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: List[str] = Field(default_factory=list)   # LinkedIn, GitHub, portfolio


class ExperienceItem(BaseModel):
    company: str = ""
    title: str = ""
    location: str = ""
    start_date: str = ""      # normalised "Mon YYYY" where possible
    end_date: str = ""        # "Present" allowed
    bullets: List[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str = ""
    role: str = ""
    tech_stack: List[str] = Field(default_factory=list)
    link: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: List[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str = ""
    degree: str = ""          # e.g. "B.Tech"
    field: str = ""           # e.g. "Computer Science"
    start_date: str = ""
    end_date: str = ""
    grade: str = ""           # e.g. "CGPA 8.4/10"


class ResumeData(BaseModel):
    """Structured resume. Used for both the original and the tailored version."""

    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    # Optional grouping for rendering, e.g. {"Languages": [...], "Tools": [...]}
    skill_groups: Dict[str, List[str]] = Field(default_factory=dict)
    experience: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    # Anything else (achievements, publications...) kept verbatim
    other_sections: Dict[str, List[str]] = Field(default_factory=dict)
    # Headings exactly as they appeared in the source (for the ATS audit)
    detected_headings: List[str] = Field(default_factory=list)

    def all_bullets(self) -> List[str]:
        """Every experience and project bullet, in order."""
        out: List[str] = []
        for e in self.experience:
            out.extend(e.bullets)
        for p in self.projects:
            out.extend(p.bullets)
        return out

    def full_text(self) -> str:
        """Flatten the structured resume back to plain text (for matching)."""
        parts: List[str] = [self.summary, ", ".join(self.skills)]
        for group, items in self.skill_groups.items():
            parts.append(f"{group}: {', '.join(items)}")
        for e in self.experience:
            parts.append(f"{e.title} {e.company} {e.start_date} {e.end_date}")
            parts.extend(e.bullets)
        for p in self.projects:
            parts.append(f"{p.name} {p.role} {', '.join(p.tech_stack)}")
            parts.extend(p.bullets)
        for ed in self.education:
            parts.append(f"{ed.degree} {ed.field} {ed.institution} {ed.grade}")
        parts.extend(self.certifications)
        for items in self.other_sections.values():
            parts.extend(items)
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Job description                                                             #
# --------------------------------------------------------------------------- #
class Importance(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    NICE = "nice"


class JDKeyword(BaseModel):
    term: str
    importance: Importance = Importance.IMPORTANT
    category: Literal[
        "hard_skill", "tool", "soft_skill", "domain", "education", "certification", "other"
    ] = "hard_skill"

    @field_validator("importance", mode="before")
    @classmethod
    def _coerce_importance(cls, v: object) -> object:
        """Accept 'Critical', 'must-have', 'nice to have'... from the LLM."""
        if isinstance(v, str):
            s = v.strip().lower()
            if s.startswith("crit") or "must" in s or s == "required":
                return "critical"
            if s.startswith("nice") or "prefer" in s or "bonus" in s:
                return "nice"
            return "important"
        return v

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, v: object) -> object:
        allowed = {"hard_skill", "tool", "soft_skill", "domain", "education", "certification", "other"}
        if isinstance(v, str):
            s = v.strip().lower().replace(" ", "_").replace("-", "_")
            return s if s in allowed else "other"
        return "other"


class JobData(BaseModel):
    job_title: str = ""
    seniority: Literal["intern", "entry", "mid", "senior", "lead", "unknown"] = "unknown"
    min_years_experience: Optional[float] = None
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    domain: str = ""
    education_requirements: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    keywords: List[JDKeyword] = Field(default_factory=list)

    @field_validator("seniority", mode="before")
    @classmethod
    def _coerce_seniority(cls, v: object) -> object:
        allowed = {"intern", "entry", "mid", "senior", "lead", "unknown"}
        if isinstance(v, str) and v.strip().lower() in allowed:
            return v.strip().lower()
        return "unknown"


# --------------------------------------------------------------------------- #
# Matching & semantic judgment                                                #
# --------------------------------------------------------------------------- #
class MatchStatus(str, Enum):
    FOUND_EXACT = "found_exact"
    FOUND_SYNONYM = "found_synonym"
    IMPLIED = "implied_by_related_experience"
    MISSING = "missing"


class RequirementMatch(BaseModel):
    requirement: str
    importance: Importance
    category: str = "hard_skill"
    status: MatchStatus
    matched_as: str = ""        # the resume term that matched (e.g. "JS")
    evidence: str = ""          # resume snippet that justifies the status
    suggestion: str = ""


class RelevanceItem(BaseModel):
    """LLM judgment of how relevant one experience/project is to the JD."""

    index: int                     # position in the combined list sent to the LLM
    relevance: int = Field(0, ge=0, le=10)
    evidence: str = ""             # must be a verbatim quote from the resume
    reason: str = ""


class ImpliedRequirement(BaseModel):
    """LLM claim that a 'missing' requirement is implied by related experience."""

    requirement: str
    implied: bool = False
    evidence: str = ""             # verbatim quote, verified in code
    reason: str = ""


class SemanticAssessment(BaseModel):
    relevance: List[RelevanceItem] = Field(default_factory=list)
    implied: List[ImpliedRequirement] = Field(default_factory=list)
    title_alignment: int = Field(0, ge=0, le=10)
    title_alignment_reason: str = ""


# --------------------------------------------------------------------------- #
# ATS audit                                                                   #
# --------------------------------------------------------------------------- #
class ATSIssue(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str                      # machine-readable, e.g. "MISSING_EMAIL"
    message: str
    evidence: str = ""


class ATSReport(BaseModel):
    issues: List[ATSIssue] = Field(default_factory=list)
    score_percent: float = 100.0   # 0-100


# --------------------------------------------------------------------------- #
# Bullet analysis                                                             #
# --------------------------------------------------------------------------- #
class BulletReview(BaseModel):
    section: Literal["experience", "project"]
    parent: str                    # company or project name
    original: str
    has_action_verb: bool = False
    has_tech: bool = False
    has_metric: bool = False
    jd_relevant: bool = False
    relevant_requirements: List[str] = Field(default_factory=list)
    length_ok: bool = True
    passive_voice: bool = False
    score: int = 0                 # 0-100, computed in code
    problems: List[str] = Field(default_factory=list)
    rewrite: str = ""
    rewrite_explanation: str = ""
    targets: List[str] = Field(default_factory=list)   # JD requirements targeted


class BulletRewrite(BaseModel):
    index: int
    rewrite: str
    explanation: str = ""
    targets: List[str] = Field(default_factory=list)


class BulletRewriteBatch(BaseModel):
    rewrites: List[BulletRewrite] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Scoring & report                                                            #
# --------------------------------------------------------------------------- #
class CategoryScore(BaseModel):
    name: str
    weight: float                  # max points for this category
    points: float                  # earned points (0..weight)
    percent: float                 # points / weight * 100
    reasons: List[str] = Field(default_factory=list)
    deductions: List[str] = Field(default_factory=list)   # each cites evidence


class FixSuggestion(BaseModel):
    title: str
    detail: str
    estimated_gain: float          # points out of 100
    category: str


class AnalysisReport(BaseModel):
    overall_score: float           # 0-100, rounded to 1 decimal
    category_scores: List[CategoryScore]
    requirement_matches: List[RequirementMatch]
    ats_report: ATSReport
    bullet_reviews: List[BulletReview]
    top_fixes: List[FixSuggestion]
    years_experience: float = 0.0
    confidence: Literal["low", "medium", "high"] = "medium"
    confidence_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an estimate produced by a hybrid rules + AI model. It is NOT the "
        "output of any real Applicant Tracking System, and real ATS products differ."
    )


# --------------------------------------------------------------------------- #
# Tailoring & validation                                                      #
# --------------------------------------------------------------------------- #
class UserConfirmedSkill(BaseModel):
    """A skill the user explicitly opted in to, with their own description."""

    skill: str
    description: str


class ChangeLogEntry(BaseModel):
    section: str
    change: str
    reason: str = ""
    jd_requirement: str = ""


class TailorLLMOutput(BaseModel):
    """Raw output of the tailoring prompt (before validation)."""

    resume: ResumeData
    change_log: List[ChangeLogEntry] = Field(default_factory=list)
    keywords_added: List[str] = Field(default_factory=list)


class TailoredResumeResult(BaseModel):
    resume: ResumeData
    change_log: List[ChangeLogEntry] = Field(default_factory=list)
    placeholders_to_fill: List[str] = Field(default_factory=list)
    skills_not_added: List[str] = Field(default_factory=list)
    keywords_added: List[str] = Field(default_factory=list)
    content_trimmed: bool = False
    trim_notes: List[str] = Field(default_factory=list)
    reverted_items: List[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    kind: Literal["company", "title", "date", "degree", "institution",
                  "skill", "number", "certification"]
    value: str
    location: str
    severity: Literal["error", "warning"] = "error"
    message: str = ""


class ValidationReport(BaseModel):
    passed: bool
    issues: List[ValidationIssue] = Field(default_factory=list)


class ComparisonReport(BaseModel):
    before: AnalysisReport
    after: AnalysisReport
    delta: float
    newly_matched: List[str] = Field(default_factory=list)
    still_missing: List[str] = Field(default_factory=list)