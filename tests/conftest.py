"""Shared pytest fixtures. Tests never call the Groq API."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.models import (  # noqa: E402
    ContactInfo,
    EducationItem,
    ExperienceItem,
    JDKeyword,
    JobData,
    ProjectItem,
    ResumeData,
)


@pytest.fixture
def resume() -> ResumeData:
    return ResumeData(
        contact=ContactInfo(name="Asha Rao", email="asha@example.com", phone="+91 9876543210",
                            location="Vijayawada", links=["https://github.com/asha"]),
        summary="Final-year CS student building ML and backend projects.",
        skills=["Python", "JS", "REST APIs", "SQL", "scikit-learn", "Git"],
        experience=[ExperienceItem(
            company="DataNest Labs", title="ML Intern", start_date="Jan 2025", end_date="Jun 2025",
            bullets=[
                "Built a churn prediction model in Python with scikit-learn reaching 87% accuracy",
                "Developed REST APIs in Flask to serve model predictions to 3 internal teams",
                "Responsible for cleaning data",
            ])],
        projects=[ProjectItem(name="Resume Parser", tech_stack=["Python", "Streamlit"],
                              bullets=["Created a Streamlit app that parses resumes into sections"])],
        education=[EducationItem(institution="ABC Institute of Technology", degree="B.Tech",
                                 field="Computer Science", start_date="2021", end_date="2025",
                                 grade="CGPA 8.6/10")],
        certifications=["AWS Certified Cloud Practitioner"],
    )


@pytest.fixture
def job() -> JobData:
    return JobData(
        job_title="Machine Learning Engineer", seniority="entry", min_years_experience=0,
        must_have_skills=["Python", "Machine Learning", "RESTful APIs"],
        keywords=[
            JDKeyword(term="Python", importance="critical"),
            JDKeyword(term="Machine Learning", importance="critical"),
            JDKeyword(term="RESTful APIs", importance="critical"),
            JDKeyword(term="JavaScript", importance="important"),
            JDKeyword(term="Docker", importance="important", category="tool"),
            JDKeyword(term="Bachelor's degree in Computer Science", importance="important",
                      category="education"),
        ],
    )