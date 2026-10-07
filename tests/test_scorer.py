"""Scoring is deterministic, bounded and explainable (no LLM: client=None)."""

from core.pipeline import analyze
from core.scorer import WEIGHTS


def _report(resume, job):
    return analyze(resume, job, None, None, resume.full_text(), n_rewrite=0)


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_same_input_same_score(resume, job):
    assert _report(resume, job).model_dump() == _report(resume, job).model_dump()


def test_score_is_sum_of_categories(resume, job):
    rep = _report(resume, job)
    assert 0 <= rep.overall_score <= 100
    assert rep.overall_score == round(sum(c.points for c in rep.category_scores), 1)
    assert sum(c.weight for c in rep.category_scores) == 100
    assert len(rep.top_fixes) <= 5


def test_every_missing_requirement_is_explained(resume, job):
    rep = _report(resume, job)
    kw = next(c for c in rep.category_scores if c.name.startswith("Hard skills"))
    assert any("Docker" in d for d in kw.deductions)


def test_adding_a_real_skill_raises_score(resume, job):
    before = _report(resume, job).overall_score
    improved = resume.model_copy(deep=True)
    improved.skills.append("Docker")
    assert _report(improved, job).overall_score > before


def test_missing_email_lowers_ats(resume, job):
    ats = lambda r: next(c for c in r.category_scores if c.name.startswith("ATS")).points  # noqa: E731
    broken = resume.model_copy(deep=True)
    broken.contact.email = ""
    assert ats(_report(broken, job)) < ats(_report(resume, job))