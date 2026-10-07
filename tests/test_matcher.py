"""Deterministic matcher: exact / synonym / missing, evidence, education, years."""

from datetime import date

from core.matcher import build_units, match_keyword, match_requirements, verify_quote, years_of_experience
from core.models import ExperienceItem, JDKeyword, MatchStatus, ResumeData


def _status(matches, term):
    return next(m for m in matches if m.requirement == term)


def test_exact_match_with_evidence(resume, job):
    m = _status(match_requirements(resume, job), "Python")
    assert m.status == MatchStatus.FOUND_EXACT
    assert "churn prediction" in m.evidence  # bullet evidence preferred over skills list


def test_synonym_matches(resume, job):
    matches = match_requirements(resume, job)
    js = _status(matches, "JavaScript")
    rest = _status(matches, "RESTful APIs")
    assert js.status == MatchStatus.FOUND_SYNONYM and js.matched_as == "js"
    assert rest.status == MatchStatus.FOUND_SYNONYM


def test_missing(resume, job):
    m = _status(match_requirements(resume, job), "Docker")
    assert m.status == MatchStatus.MISSING
    assert m.evidence == ""
    assert "Docker" in m.suggestion


def test_education_degree_level(resume, job):
    m = _status(match_requirements(resume, job), "Bachelor's degree in Computer Science")
    assert m.status in (MatchStatus.FOUND_EXACT, MatchStatus.FOUND_SYNONYM)


def test_ambiguous_short_alias_needs_skill_context():
    r = ResumeData(experience=[ExperienceItem(company="X", title="Y",
                                              bullets=["Helped the team go to market faster"])])
    m = match_keyword(JDKeyword(term="Go", importance="critical"), build_units(r))
    assert m.status == MatchStatus.MISSING


def test_years_of_experience_simple(resume):
    assert years_of_experience(resume, today=date(2026, 10, 1)) == 0.5


def test_years_of_experience_merges_overlaps():
    r = ResumeData(experience=[
        ExperienceItem(company="A", title="Dev", start_date="Jan 2020", end_date="Dec 2020"),
        ExperienceItem(company="B", title="Dev", start_date="Jun 2020", end_date="Jun 2021"),
    ])
    assert years_of_experience(r) == 1.5


def test_verify_quote(resume):
    text = resume.full_text()
    assert verify_quote("churn prediction model in Python", text)
    assert not verify_quote("Led a team of 20 engineers at Google", text)


def test_matcher_is_deterministic(resume, job):
    a = [m.model_dump() for m in match_requirements(resume, job)]
    b = [m.model_dump() for m in match_requirements(resume, job)]
    assert a == b


def test_mislabeled_degree_requirement_still_matches():
    from core.models import EducationItem
    r = ResumeData(education=[EducationItem(institution="PIT", degree="B.E.", field="Computer Engineering")])
    kw = JDKeyword(term="B.Tech / B.E. in Computer Science or equivalent", importance="important",
                   category="hard_skill")  # wrong category on purpose
    m = match_keyword(kw, build_units(r))
    assert m.status != MatchStatus.MISSING


def test_ms_excel_is_not_a_degree():
    r = ResumeData(skills=["MS Excel"])
    m = match_keyword(JDKeyword(term="MS Excel", importance="nice", category="tool"), build_units(r))
    assert m.status == MatchStatus.FOUND_EXACT