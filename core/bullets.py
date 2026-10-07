"""Bullet-level analysis.

1. review_all(): scores every experience/project bullet with fixed rules
   (action verb, tech, metric, JD relevance, length, passive voice).
2. rewrite_weakest(): sends the weakest N bullets to the LLM for rewriting,
   then REJECTS any rewrite that introduces a number or tech term not
   present in the resume.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Tuple

from .aliases import ALIASES, canonical, is_ambiguous_short_form, normalize, surface_forms
from .llm import LLMClient
from .models import BulletReview, BulletRewriteBatch, JobData, ResumeData
from .prompts import BULLET_REWRITE_PROMPT
from .validator import unsupported_numbers, unsupported_tech_terms

logger = logging.getLogger(__name__)

ACTION_VERBS = {normalize(v) for v in [
    "accelerated", "achieved", "analyzed", "architected", "automated", "built", "championed",
    "collaborated", "configured", "conducted", "consolidated", "created", "cut", "debugged",
    "decreased", "delivered", "deployed", "designed", "developed", "drove", "enabled", "engineered",
    "enhanced", "established", "evaluated", "expanded", "extracted", "facilitated", "fine-tuned",
    "generated", "grew", "guided", "identified", "implemented", "improved", "increased", "integrated",
    "introduced", "launched", "led", "maintained", "managed", "mentored", "migrated", "modeled",
    "monitored", "optimized", "orchestrated", "organized", "owned", "performed", "pioneered",
    "planned", "presented", "processed", "produced", "programmed", "prototyped", "published",
    "redesigned", "reduced", "refactored", "researched", "resolved", "restructured", "revamped",
    "scaled", "secured", "shipped", "simplified", "spearheaded", "streamlined", "strengthened",
    "structured", "supported", "tested", "trained", "transformed", "troubleshot", "upgraded",
    "validated", "visualized", "wrote", "coordinated", "authored", "benchmarked", "cleaned",
    "containerized", "crafted", "curated", "documented", "formulated", "investigated",
    "leveraged", "predicted", "quantified", "standardized", "taught", "translated", "won",
]}
WEAK_STARTS = ("responsible for", "worked on", "helped", "assisted", "involved in", "participated",
               "tasked with", "duties included", "duties", "worked as", "handled")
PASSIVE_RE = re.compile(r"\b(was|were|been|being|is|are)\s+\w+(ed|en)\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PLACEHOLDER_RE = re.compile(r"\[add metric[^\]]*\]", re.IGNORECASE)
_METRIC_RE = re.compile(
    r"\d+(?:\.\d+)?\s?(?:%|x\b|\+|k\b|m\b|ms\b|sec|seconds?|minutes?|hours?|hrs?|days?|weeks?|"
    r"users?|customers?|clients?|requests?|records?|rows?|students?|members?|people|projects?|"
    r"apis?|endpoints?|models?|lakh|crore|million|billion|downloads?|accuracy)"
    r"|[$₹€£]\s?\d|\b\d{2,}\b",
    re.IGNORECASE,
)
_SOFT_KEYS = {"communication", "teamwork", "problem solving", "agile", "user interface",
              "user experience", "application programming interface"}
TECH_FORMS = sorted({normalize(f) for key, forms in ALIASES.items() if key not in _SOFT_KEYS
                     for f in forms + [key] if not is_ambiguous_short_form(normalize(f))},
                    key=len, reverse=True)


def has_metric(text: str) -> bool:
    t = _PLACEHOLDER_RE.sub(" ", text)
    t = _YEAR_RE.sub(" ", t)
    return bool(_METRIC_RE.search(t))


def requirement_forms(job: JobData) -> List[Tuple[str, List[str]]]:
    """(JD term, its non-ambiguous normalized forms) for skill-like keywords."""
    out = []
    for kw in job.keywords:
        if kw.category in ("soft_skill", "education"):
            continue
        forms = [f for f in surface_forms(kw.term) if not is_ambiguous_short_form(f)]
        if forms:
            out.append((kw.term, forms))
    return out


def review_bullet(text: str, section: str, parent: str,
                  req_forms: List[Tuple[str, List[str]]]) -> BulletReview:
    """Score one bullet 0-100 with transparent rules."""
    norm = f" {normalize(text)} "
    words = text.split()
    first = normalize(words[0]) if words else ""
    low = text.lower().strip()

    weak = low.startswith(WEAK_STARTS)
    action = first in ACTION_VERBS and not weak
    tech = any(f" {f} " in norm for f in TECH_FORMS)
    relevant = [term for term, forms in req_forms if any(f" {f} " in norm for f in forms)]
    metric = has_metric(text)
    length_ok = 8 <= len(words) <= 35
    passive = bool(PASSIVE_RE.search(low)) or weak

    score = 25 * action + 20 * (tech or bool(relevant)) + 25 * metric + 20 * bool(relevant) + 10 * length_ok
    score -= 10 * passive
    score = max(0, min(100, score))

    problems = []
    if not action:
        problems.append("Does not start with a strong action verb")
    if not (tech or relevant):
        problems.append("No tool/technology mentioned")
    if not metric:
        problems.append("No measurable result")
    if not relevant:
        problems.append("Not tied to any JD requirement")
    if not length_ok:
        problems.append("Too short" if len(words) < 8 else "Too long (over ~2 lines)")
    if passive:
        problems.append("Passive or weak phrasing")

    return BulletReview(
        section=section,  # type: ignore[arg-type]
        parent=parent, original=text, has_action_verb=action, has_tech=tech or bool(relevant),
        has_metric=metric, jd_relevant=bool(relevant), relevant_requirements=relevant,
        length_ok=length_ok, passive_voice=passive, score=score, problems=problems,
    )


def review_all(resume: ResumeData, job: JobData) -> List[BulletReview]:
    forms = requirement_forms(job)
    reviews: List[BulletReview] = []
    for e in resume.experience:
        parent = f"{e.title} @ {e.company}".strip(" @")
        reviews.extend(review_bullet(b, "experience", parent, forms) for b in e.bullets)
    for p in resume.projects:
        reviews.extend(review_bullet(b, "project", p.name, forms) for b in p.bullets)
    return reviews


def rewrite_weakest(reviews: List[BulletReview], n: int, resume: ResumeData, job: JobData,
                    client: Optional[LLMClient]) -> List[BulletReview]:
    """Rewrite the n lowest-scoring bullets (score < 85) and verify each rewrite."""
    if n <= 0 or not reviews or client is None:
        return reviews
    order = sorted(range(len(reviews)), key=lambda i: (reviews[i].score, i))[:n]
    targets = [i for i in order if reviews[i].score < 85]
    if not targets:
        return reviews

    reqs = [kw.term for kw in job.keywords if kw.category != "soft_skill"][:20]
    allowed = list(dict.fromkeys(
        resume.skills + [t for p in resume.projects for t in p.tech_stack]
        + [s for g in resume.skill_groups.values() for s in g]
    ))
    lines = [f"[{k}] ({reviews[i].section}: {reviews[i].parent}) {reviews[i].original}"
             for k, i in enumerate(targets)]
    user_prompt = (
        "BULLETS:\n" + "\n".join(lines)
        + f"\n\nJD REQUIREMENTS: {json.dumps(reqs)}"
        + f"\n\nALLOWED FACTS (skills/tools the candidate has): {json.dumps(allowed)}"
    )
    batch = client.chat_json(BULLET_REWRITE_PROMPT, user_prompt, BulletRewriteBatch,
                             temperature=0.3, max_tokens=2000)

    source = resume.full_text()
    req_keys = {canonical(r): r for r in reqs}
    out = [r.model_copy(deep=True) for r in reviews]
    for rw in batch.rewrites:
        if not 0 <= rw.index < len(targets) or not rw.rewrite.strip():
            continue
        i = targets[rw.index]
        new_numbers = unsupported_numbers(rw.rewrite, out[i].original)
        new_tech = unsupported_tech_terms(rw.rewrite, source)
        if new_numbers or new_tech:
            added = ", ".join(new_numbers + new_tech)
            out[i].rewrite_explanation = f"Rewrite discarded: it introduced unsupported facts ({added})."
            logger.info("Discarded a bullet rewrite with unsupported facts")
            continue
        out[i].rewrite = rw.rewrite.strip()
        out[i].rewrite_explanation = rw.explanation
        out[i].targets = [req_keys[canonical(t)] for t in rw.targets if canonical(t) in req_keys]
    return out