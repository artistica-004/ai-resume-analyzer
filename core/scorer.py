"""Transparent weighted scoring (0-100), computed entirely in code.

| Category                         | Weight |
|----------------------------------|--------|
| Hard skills / keyword coverage   | 35     |
| Experience & project relevance   | 25     |
| Role/title & seniority alignment | 10     |
| Education & certifications       | 10     |
| Impact / quantified achievements | 10     |
| ATS format / parse-ability       | 10     |

The LLM only contributes evidence-verified 0-10 judgments (relevance and
title alignment); all arithmetic happens here, so identical inputs always
produce identical scores.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .aliases import normalize
from .ats_checker import ERROR_PENALTY, WARNING_PENALTY, points_from_issues
from .matcher import content_tokens
from .models import (
    AnalysisReport,
    ATSReport,
    BulletReview,
    CategoryScore,
    FixSuggestion,
    Importance,
    JobData,
    MatchStatus,
    ParseQuality,
    RequirementMatch,
    ResumeData,
    SemanticAssessment,
)

WEIGHTS: Dict[str, float] = {
    "keywords": 35, "relevance": 25, "title": 10, "education": 10, "impact": 10, "ats": 10,
}
NAMES: Dict[str, str] = {
    "keywords": "Hard skills & keyword coverage",
    "relevance": "Experience & project relevance",
    "title": "Role/title & seniority alignment",
    "education": "Education & certifications",
    "impact": "Impact & quantified achievements",
    "ats": "ATS format & parse-ability",
}
IMPORTANCE_WEIGHT = {Importance.CRITICAL: 3, Importance.IMPORTANT: 2, Importance.NICE: 1}
STATUS_CREDIT = {MatchStatus.FOUND_EXACT: 1.0, MatchStatus.FOUND_SYNONYM: 0.9,
                 MatchStatus.IMPLIED: 0.5, MatchStatus.MISSING: 0.0}
_SENIORITY_YEARS = {"intern": 0, "entry": 0, "mid": 2, "senior": 5, "lead": 8}
_TITLE_NOISE = {normalize(w) for w in ["senior", "junior", "sr", "jr", "lead", "intern", "associate",
                                        "trainee", "entry", "level", "i", "ii", "iii", "staff"]}


def _cat(key: str, points: float, reasons: List[str], deductions: List[str]) -> CategoryScore:
    weight = WEIGHTS[key]
    points = max(0.0, min(weight, points))
    return CategoryScore(name=NAMES[key], weight=weight, points=round(points, 2),
                         percent=round(points / weight * 100, 1), reasons=reasons, deductions=deductions)


def _skill_matches(matches: List[RequirementMatch]) -> List[RequirementMatch]:
    return [m for m in matches if m.category not in ("education", "certification")]


# --------------------------------------------------------------------------- #
# Categories                                                                  #
# --------------------------------------------------------------------------- #
def score_keywords(matches: List[RequirementMatch]) -> CategoryScore:
    skill = _skill_matches(matches)
    if not skill:
        return _cat("keywords", WEIGHTS["keywords"] * 0.5,
                    ["The JD had no extractable skills; neutral 50% applied."], [])
    max_w = sum(IMPORTANCE_WEIGHT[m.importance] for m in skill)
    earned = sum(IMPORTANCE_WEIGHT[m.importance] * STATUS_CREDIT[m.status] for m in skill)
    points = WEIGHTS["keywords"] * earned / max_w

    reasons, deductions = [], []
    for imp in (Importance.CRITICAL, Importance.IMPORTANT, Importance.NICE):
        group = [m for m in skill if m.importance == imp]
        if group:
            found = sum(1 for m in group if m.status != MatchStatus.MISSING)
            reasons.append(f"{found}/{len(group)} {imp.value} requirements matched.")
    for m in skill:
        lost = WEIGHTS["keywords"] * IMPORTANCE_WEIGHT[m.importance] * (1 - STATUS_CREDIT[m.status]) / max_w
        if lost < 0.05:
            continue
        if m.status == MatchStatus.MISSING:
            deductions.append(f"Missing {m.importance.value} '{m.requirement}' (−{lost:.1f}).")
        elif m.status == MatchStatus.FOUND_SYNONYM:
            deductions.append(f"'{m.requirement}' matched only as '{m.matched_as}' (−{lost:.1f}).")
        elif m.status == MatchStatus.IMPLIED:
            deductions.append(f"'{m.requirement}' only implied by: {m.evidence[:90]} (−{lost:.1f}).")
    return _cat("keywords", points, reasons, deductions)


def score_relevance(resume: ResumeData, semantic: SemanticAssessment, fresher: bool) -> CategoryScore:
    labels = ([f"[Experience] {e.title} @ {e.company}" for e in resume.experience]
              + [f"[Project] {p.name}" for p in resume.projects])
    n_exp = len(resume.experience)
    if not labels or not semantic.relevance:
        return _cat("relevance", 0, [], ["No experience or project entries could be assessed (−25)."])

    project_base = 1.0 if (fresher or n_exp == 0) else 0.7
    total_w, total, best = 0.0, 0.0, 0
    reasons, deductions = [], []
    for item in semantic.relevance:
        if item.index >= len(labels):
            continue
        if item.index < n_exp:
            w = 1 / (1 + 0.25 * item.index)
        else:
            w = project_base / (1 + 0.25 * (item.index - n_exp))
        total_w += w
        total += w * item.relevance
        best = max(best, item.relevance)
        line = f"{labels[item.index]}: {item.relevance}/10"
        if item.evidence:
            line += f' — "{item.evidence[:90]}"'
        reasons.append(line)
        if item.relevance < 4:
            deductions.append(f"{labels[item.index]} rated {item.relevance}/10: {item.reason[:120]}")
    mean = total / total_w if total_w else 0
    pct = (0.5 * best + 0.5 * mean) / 10
    reasons.insert(0, f"Best entry {best}/10, recency-weighted average {mean:.1f}/10 (50/50 blend).")
    return _cat("relevance", WEIGHTS["relevance"] * pct, reasons, deductions)


def score_title(resume: ResumeData, job: JobData, semantic: SemanticAssessment, years: float) -> CategoryScore:
    reasons, deductions = [], []
    title_toks = [t for t in content_tokens(job.job_title) if t not in _TITLE_NOISE]
    cand = " ".join([e.title for e in resume.experience] + [p.role for p in resume.projects]
                    + [resume.summary])
    cand_norm = f" {normalize(cand)} "
    overlap = (sum(1 for t in title_toks if f" {t} " in cand_norm) / len(title_toks)) if title_toks else 0.5
    llm = semantic.title_alignment / 10
    title_pts = 6 * (0.5 * overlap + 0.5 * llm)
    reasons.append(f"Title word overlap {overlap:.0%}; AI title alignment {semantic.title_alignment}/10 "
                   f"→ {title_pts:.1f}/6.")
    if semantic.title_alignment_reason:
        reasons.append(semantic.title_alignment_reason)
    if overlap < 0.5 and job.job_title:
        deductions.append(f"Your titles/summary rarely mention '{job.job_title}'.")

    expected = job.min_years_experience
    if expected is None:
        expected = _SENIORITY_YEARS.get(job.seniority)
    if expected is None or expected <= 0:
        sen_pts = 4.0
        reasons.append("No minimum experience stated → full seniority points (4/4).")
    else:
        sen_pts = 4 * min(1.0, years / expected)
        if job.seniority in ("intern", "entry") and years == 0 and resume.projects:
            sen_pts = max(sen_pts, 2.0)
        reasons.append(f"{years:.1f} years found vs {expected:g} required → {sen_pts:.1f}/4.")
        if years < expected:
            deductions.append(f"Experience {years:.1f} yrs is below the {expected:g} yrs requested.")
    return _cat("title", title_pts + sen_pts, reasons, deductions)


def score_education(resume: ResumeData, matches: List[RequirementMatch]) -> CategoryScore:
    reasons, deductions = [], []
    edu = [m for m in matches if m.category == "education"]
    if edu:
        credit = sum(STATUS_CREDIT[m.status] for m in edu) / len(edu)
        edu_pts = 7 * credit
        for m in edu:
            reasons.append(f"'{m.requirement}': {m.status.value.replace('_', ' ')}"
                           + (f" — {m.evidence[:80]}" if m.evidence else ""))
            if m.status == MatchStatus.MISSING:
                deductions.append(f"Education requirement '{m.requirement}' not met.")
    else:
        edu_pts = 7.0 if resume.education else 3.5
        reasons.append("No explicit education requirement in the JD"
                       + ("; education is listed." if resume.education else "; but no education listed."))
        if not resume.education:
            deductions.append("No education section found.")

    certs = [m for m in matches if m.category == "certification"]
    if certs:
        cert_pts = 3 * sum(STATUS_CREDIT[m.status] for m in certs) / len(certs)
        for m in certs:
            if m.status == MatchStatus.MISSING:
                deductions.append(f"Certification '{m.requirement}' not found.")
    else:
        cert_pts = 3.0
        reasons.append("No certification required → full certification points.")
    return _cat("education", edu_pts + cert_pts, reasons, deductions)


def score_impact(reviews: List[BulletReview]) -> CategoryScore:
    if not reviews:
        return _cat("impact", 0, [], ["No bullets found to assess impact (−10)."])
    n = len(reviews)
    metric = sum(r.has_metric for r in reviews) / n
    verb = sum(r.has_action_verb for r in reviews) / n
    passive = sum(r.passive_voice for r in reviews) / n
    points = 10 * (0.5 * metric + 0.3 * verb + 0.2 * (1 - passive))
    reasons = [f"{metric:.0%} of bullets have a metric (50% weight).",
               f"{verb:.0%} start with a strong action verb (30% weight).",
               f"{1 - passive:.0%} use active voice (20% weight)."]
    deductions = []
    no_metric = [r.original for r in reviews if not r.has_metric]
    if no_metric:
        deductions.append(f"{len(no_metric)} bullet(s) lack a measurable result, e.g. \"{no_metric[0][:80]}\".")
    weak = [r.original for r in reviews if not r.has_action_verb]
    if weak:
        deductions.append(f"{len(weak)} bullet(s) start weakly, e.g. \"{weak[0][:80]}\".")
    return _cat("impact", points, reasons, deductions)


def score_ats(ats: ATSReport) -> CategoryScore:
    points = points_from_issues(ats.issues)
    deductions = []
    for i in ats.issues:
        if i.severity == "error":
            deductions.append(f"{i.message} (−{ERROR_PENALTY:g})")
        elif i.severity == "warning":
            deductions.append(f"{i.message} (−{WARNING_PENALTY:g})")
    reasons = [f"{len(ats.issues)} format checks flagged "
               f"({sum(i.severity == 'error' for i in ats.issues)} errors, "
               f"{sum(i.severity == 'warning' for i in ats.issues)} warnings)."]
    return _cat("ats", points, reasons, deductions)


# --------------------------------------------------------------------------- #
# Fixes & confidence                                                          #
# --------------------------------------------------------------------------- #
def build_top_fixes(matches: List[RequirementMatch], cats: Dict[str, CategoryScore],
                    reviews: List[BulletReview], ats: ATSReport, job: JobData, k: int = 5) -> List[FixSuggestion]:
    fixes: List[FixSuggestion] = []
    skill = _skill_matches(matches)
    max_w = sum(IMPORTANCE_WEIGHT[m.importance] for m in skill) or 1

    for m in skill:
        if m.status == MatchStatus.MISSING and m.importance != Importance.NICE:
            gain = WEIGHTS["keywords"] * IMPORTANCE_WEIGHT[m.importance] * 0.9 / max_w
            fixes.append(FixSuggestion(
                title=f"Add '{m.requirement}' (only if you genuinely have it)",
                detail=f"{m.importance.value.title()} requirement not found. Confirm it in the Gaps step and "
                       "describe where you used it.",
                estimated_gain=gain, category=NAMES["keywords"]))
    syn = [m for m in skill if m.status in (MatchStatus.FOUND_SYNONYM, MatchStatus.IMPLIED)]
    if syn:
        gain = sum(WEIGHTS["keywords"] * IMPORTANCE_WEIGHT[m.importance] * (1 - STATUS_CREDIT[m.status])
                   for m in syn) / max_w
        names = ", ".join(m.requirement for m in syn[:4])
        fixes.append(FixSuggestion(
            title=f"Use the JD's exact wording for {len(syn)} term(s)",
            detail=f"e.g. {names}. Rephrase existing content with the JD's terms where accurate.",
            estimated_gain=gain, category=NAMES["keywords"]))

    if reviews:
        n = len(reviews)
        metric = sum(r.has_metric for r in reviews) / n
        if metric < 0.6:
            fixes.append(FixSuggestion(
                title="Add measurable results to your bullets",
                detail=f"Only {metric:.0%} of bullets contain a number. Add real metrics (%, users, time saved).",
                estimated_gain=10 * 0.5 * (min(1.0, metric + 0.4) - metric), category=NAMES["impact"]))
        verb = sum(r.has_action_verb for r in reviews) / n
        if verb < 0.8:
            fixes.append(FixSuggestion(
                title="Start every bullet with a strong action verb",
                detail="Replace 'Responsible for / Worked on' with Built, Designed, Automated, Reduced…",
                estimated_gain=10 * 0.3 * (1 - verb), category=NAMES["impact"]))

    for issue in ats.issues:
        if issue.severity in ("error", "warning"):
            fixes.append(FixSuggestion(
                title=f"Fix format: {issue.message}",
                detail=issue.evidence or "See the ATS audit for details.",
                estimated_gain=ERROR_PENALTY if issue.severity == "error" else WARNING_PENALTY,
                category=NAMES["ats"]))

    rel = cats["relevance"]
    if rel.percent < 60:
        fixes.append(FixSuggestion(
            title="Lead with your most JD-relevant work",
            detail="Move bullets that match the JD's responsibilities to the top of each role and trim unrelated ones.",
            estimated_gain=WEIGHTS["relevance"] * (0.6 - rel.percent / 100) * 0.5, category=NAMES["relevance"]))

    title = cats["title"]
    if title.percent < 70 and job.job_title:
        fixes.append(FixSuggestion(
            title=f"Mention the target role '{job.job_title}' in your summary",
            detail="State the role you are targeting and your closest matching experience.",
            estimated_gain=min(2.0, WEIGHTS["title"] - title.points), category=NAMES["title"]))

    fixes.sort(key=lambda f: -f.estimated_gain)
    for f in fixes:
        f.estimated_gain = round(f.estimated_gain, 1)
    return [f for f in fixes if f.estimated_gain > 0][:k]


def assess_confidence(resume: ResumeData, job: JobData, semantic: SemanticAssessment,
                      quality: Optional[ParseQuality], warnings: List[str]) -> tuple:
    level, reasons = 2, []
    if quality:
        if quality.multi_column_suspected or quality.has_tables or quality.header_footer_text:
            level = min(level, 1)
            reasons.append("Layout (columns/tables/headers) may have affected text extraction.")
        if quality.char_count < 1200:
            level = min(level, 1)
            reasons.append("Relatively little text was extracted.")
    if not resume.experience and not resume.projects:
        level = 0
        reasons.append("No experience or projects were detected.")
    if len(job.keywords) < 5:
        level = 0
        reasons.append("The JD had very few clear requirements.")
    weak = sum(1 for r in semantic.relevance if "capped" in r.reason or "Not assessed" in r.reason)
    if semantic.relevance and weak > len(semantic.relevance) / 2:
        level = min(level, 1)
        reasons.append("Many AI relevance judgments lacked verifiable evidence.")
    if any("first" in w and "characters" in w for w in warnings):
        level = min(level, 1)
        reasons.append("Input was truncated.")
    if level == 2:
        reasons.append("Text extracted cleanly and the JD had clear requirements.")
    return ["low", "medium", "high"][level], reasons


def compute_report(resume: ResumeData, job: JobData, matches: List[RequirementMatch],
                   semantic: SemanticAssessment, ats: ATSReport, bullet_reviews: List[BulletReview],
                   quality: Optional[ParseQuality], years: float, warnings: List[str],
                   fresher: bool = False) -> AnalysisReport:
    cats = {
        "keywords": score_keywords(matches),
        "relevance": score_relevance(resume, semantic, fresher),
        "title": score_title(resume, job, semantic, years),
        "education": score_education(resume, matches),
        "impact": score_impact(bullet_reviews),
        "ats": score_ats(ats),
    }
    overall = round(sum(c.points for c in cats.values()), 1)
    confidence, conf_reasons = assess_confidence(resume, job, semantic, quality, warnings)
    all_warnings = list(dict.fromkeys(warnings + (quality.warnings if quality else [])))
    return AnalysisReport(
        overall_score=overall,
        category_scores=list(cats.values()),
        requirement_matches=matches,
        ats_report=ats,
        bullet_reviews=bullet_reviews,
        top_fixes=build_top_fixes(matches, cats, bullet_reviews, ats, job),
        years_experience=years,
        confidence=confidence,
        confidence_reasons=conf_reasons,
        warnings=all_warnings,
    )