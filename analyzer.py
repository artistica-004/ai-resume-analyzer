import os
import json
from groq import Groq
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ResumeAnalysis(BaseModel):
    match_score: int
    matching_keywords: List[str]
    missing_keywords: List[str]
    strengths: List[str]
    weaknesses: List[str]
    improved_bullets: List[str]


def clean_json_response(raw: str) -> str:
    """Strip markdown fences if model wraps response in ```json ... ```"""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def analyze_resume(resume_text: str, job_description: str) -> ResumeAnalysis:
    prompt = f"""
You are an expert ATS (Applicant Tracking System) and career coach.

Analyze the resume against the job description and return ONLY valid JSON.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return JSON with exactly these fields:
- match_score: integer 0-100 (how well resume matches JD)
- matching_keywords: list of keywords present in both resume and JD
- missing_keywords: list of important keywords in JD but missing from resume
- strengths: list of 3-4 strong points in the resume
- weaknesses: list of 3-4 areas that need improvement
- improved_bullets: take 3 weak bullet points from resume and rewrite them stronger and ATS-friendly

IMPORTANT: Return ONLY the raw JSON object. Do NOT use ```json or ``` or any markdown formatting. Start your response with {{ and end with }}.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    print("RAW RESPONSE:", repr(raw))  # helpful for debugging, remove later

    raw = clean_json_response(raw)

    if not raw:
        raise ValueError("Model returned an empty response. Please try again.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse model response as JSON.\nError: {e}\nRaw output:\n{raw}")

    return ResumeAnalysis(**data)

