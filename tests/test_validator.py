"""Anti-fabrication validator catches injected fakes and allows legitimate content."""

from core.models import UserConfirmedSkill
from core.validator import validate


def _kinds(report):
    return {(i.kind, i.value.lower()) for i in report.issues}


def test_identical_copy_passes(resume, job):
    report = validate(resume, resume.model_copy(deep=True), job=job)
    assert report.passed, report.issues


def test_fake_company_caught(resume):
    t = resume.model_copy(deep=True)
    t.experience[0].company = "Google"
    assert ("company", "google") in _kinds(validate(resume, t))


def test_fake_date_caught(resume):
    t = resume.model_copy(deep=True)
    t.experience[0].start_date = "Jan 2023"
    assert any(k == "date" for k, _ in _kinds(validate(resume, t)))


def test_fake_skill_in_bullet_caught(resume, job):
    t = resume.model_copy(deep=True)
    t.experience[0].bullets[0] = "Deployed churn models on Kubernetes and Docker"
    kinds = _kinds(validate(resume, t, job=job))
    assert ("skill", "kubernetes") in kinds
    assert any(k == "skill" and "docker" in v for k, v in kinds)


def test_fake_skill_in_skills_list_caught(resume):
    t = resume.model_copy(deep=True)
    t.skills.append("TensorFlow")
    assert ("skill", "tensorflow") in _kinds(validate(resume, t))


def test_fake_number_caught(resume):
    t = resume.model_copy(deep=True)
    t.experience[0].bullets[0] = "Built a churn prediction model reaching 95% accuracy"
    assert ("number", "95") in _kinds(validate(resume, t))


def test_confirmed_skill_is_allowed(resume, job):
    t = resume.model_copy(deep=True)
    t.skills.append("Docker")
    t.projects[0].bullets.append("Containerized the parser API using Docker")
    confirmed = [UserConfirmedSkill(skill="Docker",
                                    description="Containerized my parser API using Docker for a college project.")]
    report = validate(resume, t, confirmed=confirmed, job=job)
    assert report.passed, report.issues


def test_metric_placeholder_is_not_a_fabricated_number(resume):
    t = resume.model_copy(deep=True)
    t.experience[0].bullets[2] = ("Cleaned raw datasets for model training "
                                  "[add metric: e.g. % improvement / users / time saved]")
    assert validate(resume, t).passed