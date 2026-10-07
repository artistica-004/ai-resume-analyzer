"""Alias / acronym normalisation."""

from core.aliases import canonical, normalize, surface_forms


def test_acronyms_map_to_canonical():
    assert canonical("k8s") == "kubernetes"
    assert canonical("JS") == "javascript"
    assert canonical("ML") == "machine learning"
    assert canonical("NLP") == "natural language processing"
    assert canonical("LLMs") == "large language models"


def test_variants_share_a_key():
    assert canonical("REST APIs") == canonical("RESTful API") == "restful apis"
    assert canonical("CI/CD") == canonical("continuous integration")
    assert canonical("Postgres") == canonical("PostgreSQL")


def test_protected_punctuation():
    assert canonical("C++") == "c++"
    assert canonical("C#") == "c#"
    assert canonical("Node.js") == "node.js"
    assert canonical("C++") != canonical("C#")


def test_normalize_is_case_and_punctuation_insensitive():
    assert normalize("Machine-Learning!") == normalize("machine learning")
    assert normalize("Kubernetes") == "kubernetes"  # protected from stemming


def test_unknown_terms_fall_back_to_normalized_form():
    assert canonical("Snowflake Cortex") == normalize("Snowflake Cortex")


def test_surface_forms_include_expansion():
    assert normalize("machine learning") in surface_forms("ML")