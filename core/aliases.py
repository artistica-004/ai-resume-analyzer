"""Synonym / acronym dictionary and text normalisation.

Design:
- ALIASES maps a canonical key -> list of surface forms (including the key).
- normalize() lowercases, protects tech tokens like "c++" / ".net",
  strips punctuation and applies a tiny rule-based stemmer
  (no NLTK/spaCy downloads needed on Hugging Face Spaces).
- canonical() maps any surface form to its canonical key.

To extend: add entries to ALIASES. Keys should be the most common name.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Set

ALIASES: Dict[str, List[str]] = {
    # ---------- languages ----------
    "javascript": ["javascript", "js", "ecmascript", "es6", "java script"],
    "typescript": ["typescript", "ts"],
    "python": ["python", "python3", "py"],
    "c++": ["c++", "cpp", "cplusplus"],
    "c#": ["c#", "csharp", "c sharp"],
    "golang": ["golang", "go lang", "go"],
    "sql": ["sql", "structured query language"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    # ---------- AI / data ----------
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision", "cv"],
    "large language models": ["large language models", "large language model", "llm", "llms"],
    "generative ai": ["generative ai", "genai", "gen ai"],
    "retrieval augmented generation": ["retrieval augmented generation", "rag"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "data analysis": ["data analysis", "data analytics"],
    "data visualization": ["data visualization", "data visualisation", "dataviz"],
    "exploratory data analysis": ["exploratory data analysis", "eda"],
    "prompt engineering": ["prompt engineering"],
    "hugging face": ["hugging face", "huggingface", "hf transformers"],
    # ---------- web / backend ----------
    "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "nodejs", "node"],
    "express.js": ["express.js", "expressjs", "express"],
    "next.js": ["next.js", "nextjs"],
    "vue.js": ["vue.js", "vuejs", "vue"],
    "angular": ["angular", "angularjs"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi", "fast api"],
    "spring boot": ["spring boot", "springboot"],
    "restful apis": ["restful apis", "restful api", "rest apis", "rest api", "rest", "restful"],
    "graphql": ["graphql"],
    "microservices": ["microservices", "microservice", "micro services"],
    "object oriented programming": ["object oriented programming", "oop", "oops", "object-oriented"],
    "data structures and algorithms": ["data structures and algorithms", "dsa", "data structures"],
    # ---------- databases ----------
    "postgresql": ["postgresql", "postgres", "psql"],
    "mysql": ["mysql"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "nosql": ["nosql", "no sql"],
    # ---------- cloud / devops ----------
    "amazon web services": ["amazon web services", "aws"],
    "google cloud platform": ["google cloud platform", "gcp", "google cloud"],
    "microsoft azure": ["microsoft azure", "azure"],
    "kubernetes": ["kubernetes", "k8s"],
    "docker": ["docker", "containerization", "containers"],
    "ci/cd": ["ci/cd", "cicd", "ci cd", "continuous integration", "continuous deployment",
              "continuous delivery"],
    "github actions": ["github actions"],
    "git": ["git", "github", "gitlab", "version control"],
    "linux": ["linux", "unix"],
    "terraform": ["terraform", "iac", "infrastructure as code"],
    # ---------- practices / soft ----------
    "agile": ["agile", "scrum", "kanban"],
    "unit testing": ["unit testing", "unit tests", "pytest", "junit", "jest"],
    "test driven development": ["test driven development", "tdd"],
    "communication": ["communication", "communication skills"],
    "teamwork": ["teamwork", "collaboration", "team player"],
    "problem solving": ["problem solving", "problem-solving"],
    "user interface": ["user interface", "ui"],
    "user experience": ["user experience", "ux"],
    "application programming interface": ["application programming interface", "api", "apis"],
}

# Tokens whose punctuation is meaningful and must survive normalisation.
_PROTECTED = {
    "c++": " cplusplus ",
    "c#": " csharp ",
    ".net": " dotnet ",
    "ci/cd": " cicd ",
    "node.js": " nodejs ",
    "react.js": " reactjs ",
    "next.js": " nextjs ",
    "vue.js": " vuejs ",
    "express.js": " expressjs ",
}

# Words the stemmer must never touch (they would collapse wrongly).
_NO_STEM: Set[str] = {
    "kubernetes", "pandas", "aws", "js", "ts", "css", "sass", "less", "devops",
    "analytics", "graphics", "statistics", "mathematics", "physics", "jenkins",
    "redis", "postgres", "express", "apis", "llms", "ms", "access", "process",
}


def _stem(word: str) -> str:
    """Tiny conservative suffix stripper (good enough for skill matching)."""
    if word in _NO_STEM or len(word) <= 3 or not word.isalpha():
        return word
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "s" and word.endswith("ss"):
                return word
            return word[: -len(suffix)] + repl
    return word


@lru_cache(maxsize=4096)
def normalize(text: str) -> str:
    """Lowercase, protect tech tokens, strip punctuation, stem each word."""
    t = f" {text.lower()} "
    for raw, safe in _PROTECTED.items():
        t = t.replace(raw, safe)
    t = re.sub(r"[^a-z0-9+#\s]", " ", t)
    words = [_stem(w) for w in t.split()]
    return " ".join(words)


def _build_index() -> Dict[str, str]:
    """normalized surface form -> canonical key."""
    index: Dict[str, str] = {}
    for canonical_key, forms in ALIASES.items():
        for form in forms + [canonical_key]:
            index[normalize(form)] = canonical_key
    return index


_INDEX: Dict[str, str] = _build_index()


def canonical(term: str) -> str:
    """Return the canonical key for a term, or its normalized form if unknown."""
    n = normalize(term)
    return _INDEX.get(n, n)


def surface_forms(term: str) -> List[str]:
    """All normalized surface forms that count as this term (incl. itself)."""
    key = canonical(term)
    forms = {normalize(term)}
    if key in ALIASES:
        forms.update(normalize(f) for f in ALIASES[key])
        forms.add(normalize(key))
    return sorted(forms, key=len, reverse=True)


def is_ambiguous_short_form(form_normalized: str) -> bool:
    """Short aliases like 'go', 'ts', 'cv', 'ai' need stricter context to count."""
    return len(form_normalized) <= 2 or form_normalized in {"go", "node", "rest", "express", "vue"}