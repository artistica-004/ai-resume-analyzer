"""AI Resume Analyzer v2: JD-Tailored Resume Generator + Deep Analyzer.

Streamlit UI only. All business logic lives in core/.
Steps: 1 Input → 2 Analysis → 3 Gaps → 4 Tailored resume → 5 Before vs After → 6 Download
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, List, Optional, Tuple, TypeVar

import pandas as pd
import streamlit as st

from core.jd_analyzer import JDError, analyze_jd
from core.llm import DEFAULT_MODEL, LLMError
from core.models import (
    AnalysisReport,
    ComparisonReport,
    JobData,
    MatchStatus,
    ParsedResume,
    ParseQuality,
    ResumeData,
    TailoredResumeResult,
    UserConfirmedSkill,
    ValidationReport,
)
from core.parser import MAX_FILE_MB, ResumeParseError, parse_resume_bytes
from core.pipeline import TailoredBundle, analyze, analyze_tailored, compare, generate_tailored
from core.renderer import render_report_pdf
from core.resume_structurer import structure_resume
from core.tailor import PLACEHOLDER_RE

# Logs never contain resume or JD text: core modules only log counts and types.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

st.set_page_config(page_title="AI Resume Analyzer & Tailor", page_icon="📄", layout="wide")

MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
STEPS = ["1. Input", "2. Analysis", "3. Gaps", "4. Tailored resume", "5. Before vs After", "6. Download"]
STATUS_LABEL = {
    MatchStatus.FOUND_EXACT: "✅ Exact",
    MatchStatus.FOUND_SYNONYM: "🟢 Synonym",
    MatchStatus.IMPLIED: "🟡 Implied",
    MatchStatus.MISSING: "❌ Missing",
}
SEVERITY_ICON = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Cached pipeline calls (re-runs with identical inputs never re-call the API) #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_parse(data: bytes, name: str) -> ParsedResume:
    return parse_resume_bytes(data, name)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_structure(raw_text: str, model: str) -> Tuple[ResumeData, List[str]]:
    from core.pipeline import get_client
    return structure_resume(raw_text, get_client())


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_jd(jd_text: str, model: str) -> Tuple[JobData, List[str]]:
    from core.pipeline import get_client
    return analyze_jd(jd_text, get_client())


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_analysis(resume_json: str, job_json: str, quality_json: str, raw_text: str,
                    n_rewrite: int, fresher: bool, warnings: Tuple[str, ...], model: str) -> AnalysisReport:
    from core.pipeline import get_client
    quality = ParseQuality.model_validate_json(quality_json) if quality_json else None
    return analyze(ResumeData.model_validate_json(resume_json), JobData.model_validate_json(job_json),
                   get_client(), quality, raw_text, n_rewrite=n_rewrite, fresher=fresher,
                   warnings=list(warnings))


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_tailor(resume_json: str, job_json: str, confirmed_json: Tuple[str, ...], tone: str,
                  fresher: bool, model: str) -> TailoredBundle:
    from core.pipeline import get_client
    confirmed = [UserConfirmedSkill.model_validate_json(c) for c in confirmed_json]
    return generate_tailored(ResumeData.model_validate_json(resume_json), JobData.model_validate_json(job_json),
                             confirmed, get_client(), tone=tone, fresher=fresher)


@st.cache_data(show_spinner=False, max_entries=16, ttl=3600)
def cached_after(tailored_json: str, pdf_bytes: bytes, job_json: str, fresher: bool, model: str) -> AnalysisReport:
    from core.pipeline import get_client
    bundle = TailoredBundle(result=TailoredResumeResult(resume=ResumeData.model_validate_json(tailored_json)),
                            validation=ValidationReport(passed=True), docx_bytes=b"", pdf_bytes=pdf_bytes)
    return analyze_tailored(bundle, JobData.model_validate_json(job_json), get_client(), fresher=fresher)


def run_step(label: str, fn: Callable[..., T], *args) -> Optional[T]:
    """Run one pipeline step with a spinner and friendly errors (never a stack trace)."""
    try:
        with st.spinner(label):
            return fn(*args)
    except (ResumeParseError, JDError, LLMError) as exc:
        st.error(str(exc))
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error during: %s", label)
        st.error("Something unexpected went wrong. Please try again; if it persists, try a different file.")
    return None


# --------------------------------------------------------------------------- #
# Session state & navigation                                                  #
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {"max_step": 1, "nav": STEPS[0], "parsed": None, "resume": None, "job": None,
                "prep_warnings": [], "report_before": None, "confirmed": [], "bundle": None,
                "report_after": None, "comparison": None, "settings": {}}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def go_to(step: int) -> None:
    """Request navigation to a step. Applied at the start of the next run,
    because a widget's key (the sidebar radio 'nav') can't be changed after
    the widget has been drawn in the current run."""
    st.session_state.max_step = max(st.session_state.max_step, step)
    st.session_state.pending_nav = STEPS[step - 1]
    st.rerun()


def reset_downstream(from_step: int) -> None:
    """Clear results that depend on earlier steps."""
    if from_step <= 2:
        for k in ("report_before",):
            st.session_state[k] = None
    if from_step <= 3:
        st.session_state.confirmed = []
    if from_step <= 4:
        for k in ("bundle", "report_after", "comparison"):
            st.session_state[k] = None
    st.session_state.max_step = min(st.session_state.max_step, from_step)


# --------------------------------------------------------------------------- #
# Display helpers                                                             #
# --------------------------------------------------------------------------- #
def gauge(score: float, label: str) -> None:
    color = "#16a34a" if score >= 75 else "#d97706" if score >= 50 else "#dc2626"
    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;">
  <div style="width:150px;height:150px;border-radius:50%;
       background:conic-gradient({color} {score * 3.6:.1f}deg,#e5e7eb 0deg);
       display:flex;align-items:center;justify-content:center;">
    <div style="width:112px;height:112px;border-radius:50%;background:#ffffff;display:flex;
         flex-direction:column;align-items:center;justify-content:center;">
      <span style="font-size:2rem;font-weight:700;color:{color};">{score:.0f}</span>
      <span style="font-size:.8rem;color:#6b7280;">/ 100</span>
    </div>
  </div>
  <div style="margin-top:.4rem;font-weight:600;">{label}</div>
</div>""", unsafe_allow_html=True)


def _md_escape(text: str) -> str:
    return text.replace("$", "\\$").replace("*", "\\*")


def highlight(text: str, terms: List[str]) -> str:
    """Bold added keywords and colour metric placeholders for markdown display."""
    text = _md_escape(text)
    text = PLACEHOLDER_RE.sub(lambda m: ":orange-background[⚠ " + m.group(0).strip("[]") + "]", text)
    for term in sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True):
        text = re.sub(rf"(?<![\w*])({re.escape(term)})(?![\w*])", r"**\1**", text, flags=re.IGNORECASE)
    return text


def resume_markdown(r: ResumeData, terms: Optional[List[str]] = None) -> str:
    terms = terms or []
    c = r.contact
    lines = [f"### {_md_escape(c.name or 'Name not found')}",
             _md_escape(" | ".join(x for x in [c.email, c.phone, c.location, *c.links] if x))]
    if r.summary:
        lines += ["#### Summary", highlight(r.summary, terms)]
    if r.skills or r.skill_groups:
        lines.append("#### Skills")
        if r.skill_groups:
            lines += [f"- **{_md_escape(g)}:** {highlight(', '.join(v), terms)}" for g, v in r.skill_groups.items()]
        else:
            lines.append(highlight(", ".join(r.skills), terms))
    if r.experience:
        lines.append("#### Experience")
        for e in r.experience:
            lines.append(f"**{_md_escape(e.title)}** | {_md_escape(e.company)} | {e.start_date} - {e.end_date}")
            lines += [f"- {highlight(b, terms)}" for b in e.bullets]
    if r.projects:
        lines.append("#### Projects")
        for p in r.projects:
            tech = f" | Tech: {', '.join(p.tech_stack)}" if p.tech_stack else ""
            lines.append(f"**{_md_escape(p.name)}**{_md_escape(tech)}")
            lines += [f"- {highlight(b, terms)}" for b in p.bullets]
    if r.education:
        lines.append("#### Education")
        lines += [f"- {_md_escape(' '.join(x for x in [ed.degree, ed.field, '|', ed.institution, ed.end_date, ed.grade] if x))}"
                  for ed in r.education]
    if r.certifications:
        lines.append("#### Certifications")
        lines += [f"- {_md_escape(x)}" for x in r.certifications]
    return "\n\n".join(lines)


def show_report(report: AnalysisReport, title: str) -> None:
    """Full analysis display: gauge, breakdown, requirements, ATS, fixes, bullets."""
    st.subheader(title)
    c1, c2 = st.columns([1, 2])
    with c1:
        gauge(report.overall_score, "Estimated match score")
    with c2:
        icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}[report.confidence]
        st.markdown(f"**Confidence:** {icon} {report.confidence.title()}")
        for reason in report.confidence_reasons:
            st.caption(f"• {reason}")
        st.caption(f"Years of experience detected from dates: {report.years_experience:.1f}")
        for w in report.warnings:
            st.warning(w)
        st.caption(f"⚠️ {report.disclaimer}")

    st.markdown("#### Category breakdown")
    df = pd.DataFrame({"Category": [c.name for c in report.category_scores],
                       "Score %": [c.percent for c in report.category_scores]}).set_index("Category")
    st.bar_chart(df, height=260)
    for c in report.category_scores:
        with st.expander(f"{c.name}: {c.points:.1f} / {c.weight:g}"):
            for r in c.reasons:
                st.markdown(f"- {_md_escape(r)}")
            if c.deductions:
                st.markdown("**Deductions (with evidence):**")
                for d in c.deductions:
                    st.markdown(f"- {_md_escape(d)}")

    st.markdown("#### Top 5 fixes that raise your score the most")
    if not report.top_fixes:
        st.success("No major fixes found.")
    for i, f in enumerate(report.top_fixes, 1):
        st.markdown(f"**{i}. {_md_escape(f.title)}** — est. **+{f.estimated_gain:.1f}** pts  \n"
                    f"<span style='color:#6b7280'>{_md_escape(f.detail)}</span>", unsafe_allow_html=True)

    st.markdown("#### Requirement-by-requirement match")
    req_df = pd.DataFrame([{
        "Requirement": m.requirement, "Importance": m.importance.value, "Status": STATUS_LABEL[m.status],
        "Matched as": m.matched_as, "Evidence from resume": m.evidence, "Suggestion": m.suggestion,
    } for m in report.requirement_matches])
    st.dataframe(req_df, use_container_width=True, hide_index=True)

    st.markdown("#### ATS format audit")
    if not report.ats_report.issues:
        st.success("No ATS format issues detected.")
    for issue in report.ats_report.issues:
        ev = f" — `{issue.evidence}`" if issue.evidence else ""
        st.markdown(f"{SEVERITY_ICON[issue.severity]} {_md_escape(issue.message)}{ev}")

    if report.bullet_reviews:
        st.markdown("#### Bullet-level review")
        bdf = pd.DataFrame([{
            "Score": b.score, "Bullet": b.original, "Where": b.parent,
            "Verb": "✓" if b.has_action_verb else "✗", "Tech": "✓" if b.has_tech else "✗",
            "Metric": "✓" if b.has_metric else "✗", "JD-relevant": "✓" if b.jd_relevant else "✗",
            "Problems": "; ".join(b.problems),
        } for b in sorted(report.bullet_reviews, key=lambda x: x.score)])
        st.dataframe(bdf, use_container_width=True, hide_index=True)
        rewrites = [b for b in report.bullet_reviews if b.rewrite or b.rewrite_explanation]
        if rewrites:
            with st.expander(f"✍️ Suggested rewrites for the weakest {len(rewrites)} bullet(s)", expanded=True):
                for b in rewrites:
                    st.markdown(f"**Before ({b.score}/100):** {_md_escape(b.original)}")
                    if b.rewrite:
                        st.markdown(f"**After:** {highlight(b.rewrite, [])}")
                    st.caption(b.rewrite_explanation + (f" Targets: {', '.join(b.targets)}" if b.targets else ""))
                    st.divider()


# --------------------------------------------------------------------------- #
# Steps                                                                       #
# --------------------------------------------------------------------------- #
def step_input() -> None:
    st.header("Step 1 — Upload your resume and the job description")
    col1, col2 = st.columns(2)
    with col1:
        upload = st.file_uploader(f"Resume (PDF or DOCX, max {MAX_FILE_MB} MB)", type=["pdf", "docx"])
    with col2:
        jd_text = st.text_area("Job description", height=260, placeholder="Paste the full job description here…")

    with st.expander("Optional settings", expanded=False):
        o1, o2, o3 = st.columns(3)
        target_title = o1.text_input("Target job title (overrides the JD's title)")
        stated_years = o2.number_input("Your years of experience (0 = auto-detect)", 0.0, 40.0, 0.0, 0.5)
        tone = o3.selectbox("Tone", ["standard", "concise", "detailed"])
        fresher = st.toggle("Fresher / student mode (give projects & education more weight)",
                            value=bool(stated_years and stated_years < 1))

    if st.button("🔍 Analyze my resume", type="primary", use_container_width=True):
        if upload is None:
            st.warning("Please upload your resume.")
            return
        if len(jd_text.strip()) < 150:
            st.warning("Please paste the full job description (at least 150 characters).")
            return
        reset_downstream(2)
        st.session_state.settings = {"tone": tone, "fresher": fresher, "target_title": target_title.strip(),
                                     "stated_years": stated_years}

        parsed = run_step("Reading your resume…", cached_parse, upload.getvalue(), upload.name)
        if parsed is None:
            return
        structured = run_step("Understanding resume sections (AI)…", cached_structure, parsed.raw_text, MODEL)
        if structured is None:
            return
        jd = run_step("Analyzing the job description (AI)…", cached_jd, jd_text, MODEL)
        if jd is None:
            return
        resume, w1 = structured
        job, w2 = jd
        if target_title.strip():
            job = job.model_copy(update={"job_title": target_title.strip()})

        n_rewrite = st.session_state.get("n_rewrite", 6)
        report = run_step("Scoring and finding evidence (AI + rules)…", cached_analysis,
                          resume.model_dump_json(), job.model_dump_json(), parsed.quality.model_dump_json(),
                          parsed.raw_text, n_rewrite, fresher, tuple(w1 + w2), MODEL)
        if report is None:
            return
        if stated_years and abs(stated_years - report.years_experience) >= 2:
            report.warnings.append(f"You entered {stated_years:g} years, but the dates on your resume add up to "
                                   f"{report.years_experience:.1f}. The page budget uses the resume dates.")
        st.session_state.update(parsed=parsed, resume=resume, job=job, prep_warnings=w1 + w2,
                                report_before=report)
        go_to(2)


def step_analysis() -> None:
    report: AnalysisReport = st.session_state.report_before
    job: JobData = st.session_state.job
    st.header("Step 2 — Analysis of your original resume")
    st.caption(f"Target role: **{job.job_title or 'not stated'}** · seniority: {job.seniority} · "
               f"min. experience: {job.min_years_experience if job.min_years_experience is not None else 'not stated'}")
    show_report(report, "Original resume vs JD")
    if st.button("Next: review skill gaps →", type="primary"):
        go_to(3)


def step_gaps() -> None:
    report: AnalysisReport = st.session_state.report_before
    st.header("Step 3 — Gaps you may want to address")
    st.info("The generator **never invents** skills. Tick a skill only if you genuinely have experience with it, "
            "and describe that experience in your own words. Unticked skills stay out of your resume.")

    gaps = [m for m in report.requirement_matches if m.status == MatchStatus.MISSING
            and m.category in ("hard_skill", "tool", "certification", "domain")]
    rank = {"critical": 0, "important": 1, "nice": 2}
    gaps.sort(key=lambda m: rank[m.importance.value])
    if not gaps:
        st.success("No missing skills, so there's nothing to confirm.")

    confirmed: List[UserConfirmedSkill] = []
    problems: List[str] = []
    for i, m in enumerate(gaps):
        badge = {"critical": "🔴 must-have", "important": "🟠 important", "nice": "⚪ nice-to-have"}[m.importance.value]
        checked = st.checkbox(f"I genuinely have experience with **{m.requirement}** ({badge})", key=f"gap_chk_{i}")
        desc = st.text_area("Describe where and how you used it (1-2 sentences, no made-up numbers)",
                            key=f"gap_desc_{i}", disabled=not checked, height=70,
                            placeholder="e.g. Used Docker to containerize my Flask API for a college project.")
        if checked:
            if len(desc.strip()) < 15:
                problems.append(m.requirement)
            else:
                confirmed.append(UserConfirmedSkill(skill=m.requirement, description=desc.strip()))
        st.divider()

    tone = st.session_state.settings.get("tone", "standard")
    fresher = st.session_state.settings.get("fresher", False)
    if st.button("✨ Generate my tailored resume", type="primary", use_container_width=True):
        if problems:
            st.warning("Please add a short description (15+ characters) for: " + ", ".join(problems))
            return
        reset_downstream(4)
        resume: ResumeData = st.session_state.resume
        job: JobData = st.session_state.job
        bundle = run_step("Tailoring your resume (AI) and checking every fact…", cached_tailor,
                          resume.model_dump_json(), job.model_dump_json(),
                          tuple(c.model_dump_json() for c in confirmed), tone, fresher, MODEL)
        if bundle is None:
            return
        after = run_step("Re-analyzing the tailored resume…", cached_after,
                         bundle.result.resume.model_dump_json(), bundle.pdf_bytes, job.model_dump_json(),
                         fresher, MODEL)
        if after is None:
            return
        st.session_state.update(confirmed=confirmed, bundle=bundle, report_after=after,
                                comparison=compare(report, after))
        go_to(4)


def step_tailored() -> None:
    bundle: TailoredBundle = st.session_state.bundle
    result = bundle.result
    st.header("Step 4 — Your tailored resume")

    if bundle.validation.passed:
        st.success("✅ No fabricated facts detected. Every company, title, date, degree, skill and number "
                   "traces back to your original resume or your confirmed skills.")
    else:
        st.error("⚠️ The validator found items that are not supported by your original resume. "
                 "Review them before using this resume:")
        for i in bundle.validation.issues:
            st.markdown(f"- **{i.kind}** `{i.value}` in *{i.location}*: {i.message}")

    if result.placeholders_to_fill:
        st.warning(f"✏️ {len(result.placeholders_to_fill)} bullet(s) contain a **metric placeholder**. "
                   "Replace each `[add metric…]` with a real number, or delete it, before sending.")
        for p in result.placeholders_to_fill:
            st.markdown(f"- {highlight(p, [])}")
    if result.content_trimmed or result.trim_notes:
        for note in result.trim_notes:
            st.info(f"✂️ {note}")
    if result.reverted_items:
        with st.expander(f"🛡️ Anti-fabrication guard reverted {len(result.reverted_items)} item(s)"):
            for r in result.reverted_items:
                st.markdown(f"- {_md_escape(r)}")

    if result.keywords_added:
        st.markdown("**JD keywords now covered:** " + " ".join(f"`{k}`" for k in result.keywords_added))

    left, right = st.columns(2)
    with left:
        st.markdown("##### Original")
        with st.container(border=True, height=650):
            st.markdown(resume_markdown(st.session_state.resume))
    with right:
        st.markdown("##### Tailored (added keywords in **bold**, placeholders highlighted)")
        with st.container(border=True, height=650):
            st.markdown(resume_markdown(result.resume, result.keywords_added))

    st.markdown("#### Change log")
    if result.change_log:
        st.dataframe(pd.DataFrame([c.model_dump() for c in result.change_log]),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No change log returned.")

    if result.skills_not_added:
        st.markdown("#### Gaps still not on your resume")
        st.caption("These were left out on purpose because they are not supported by your resume. "
                   "Learn them or add real evidence, then re-run.")
        st.markdown(", ".join(f"`{s}`" for s in result.skills_not_added))

    if st.button("Next: compare scores →", type="primary"):
        go_to(5)


def step_compare() -> None:
    cmp: ComparisonReport = st.session_state.comparison
    st.header("Step 5 — Before vs After")
    c1, c2, c3 = st.columns(3)
    with c1:
        gauge(cmp.before.overall_score, "Original")
    with c2:
        gauge(cmp.after.overall_score, "Tailored")
    with c3:
        st.metric("Change", f"{cmp.after.overall_score:.1f}", f"{cmp.delta:+.1f} pts")
        st.caption("Both scores use the same rubric and code. The tailored score was computed by "
                   "re-parsing the generated PDF, as an ATS would.")

    df = pd.DataFrame({
        "Category": [c.name for c in cmp.before.category_scores],
        "Original %": [c.percent for c in cmp.before.category_scores],
        "Tailored %": [c.percent for c in cmp.after.category_scores],
    }).set_index("Category")
    st.bar_chart(df, height=300, stack=False)

    a, b = st.columns(2)
    with a:
        st.markdown("#### ✅ Moved from missing → matched")
        st.markdown("\n".join(f"- {r}" for r in cmp.newly_matched) or "_None_")
    with b:
        st.markdown("#### ❌ Still missing")
        st.markdown("\n".join(f"- {r}" for r in cmp.still_missing) or "_None_")

    with st.expander("Full analysis of the tailored resume"):
        show_report(cmp.after, "Tailored resume vs JD")
    if st.button("Next: download →", type="primary"):
        go_to(6)


def step_download() -> None:
    bundle: TailoredBundle = st.session_state.bundle
    cmp: ComparisonReport = st.session_state.comparison
    job: JobData = st.session_state.job
    name = (st.session_state.resume.contact.name or "resume").replace(" ", "_")
    st.header("Step 6 — Download")
    if bundle.result.placeholders_to_fill:
        st.warning("Remember to replace the `[add metric…]` placeholders before you send the resume.")

    c1, c2 = st.columns(2)
    c1.download_button("📄 Tailored resume (DOCX)", bundle.docx_bytes, f"{name}_tailored.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       use_container_width=True)
    c2.download_button("📕 Tailored resume (PDF)", bundle.pdf_bytes, f"{name}_tailored.pdf",
                       "application/pdf", use_container_width=True)

    c3, c4 = st.columns(2)
    c3.download_button("🧾 Analysis report (JSON)", cmp.model_dump_json(indent=2), "analysis_report.json",
                       "application/json", use_container_width=True)
    report_pdf = run_step("Building report PDF…", render_report_pdf, cmp.before, cmp.after, job.job_title)
    if report_pdf:
        c4.download_button("📊 Analysis report (PDF)", report_pdf, "analysis_report.pdf", "application/pdf",
                           use_container_width=True)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    with st.sidebar:
        st.title("📄 Resume Analyzer")
        st.radio("Steps", STEPS[: st.session_state.max_step], key="nav")
        st.divider()
        st.slider("Weakest bullets to rewrite", 1, 15, 6, key="n_rewrite",
                  help="Applies the next time you click Analyze.")
        st.caption(f"Model: `{MODEL}`")
        if st.button("🔄 Start over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.divider()
        st.caption("🔒 **Privacy:** your resume and JD are processed in memory only. They are never written "
                   "to disk, never logged, and are sent only to the Groq API for analysis. Results are kept "
                   "in temporary in-memory cache for up to 1 hour so re-runs don't re-call the API.")
        st.caption("Scores are estimates from a hybrid rules + AI model, **not** a real ATS result.")


def main() -> None:
    init_state()
    # Apply navigation requested by go_to() BEFORE the sidebar radio is created
    pending = st.session_state.pop("pending_nav", None)
    if pending:
        st.session_state.nav = pending
    sidebar()
    st.markdown("## JD-Tailored Resume Generator + Deep Analyzer")
    step = STEPS.index(st.session_state.nav) + 1 if st.session_state.nav in STEPS else 1

    # Guard: never render a step whose data is missing (e.g. after Start over)
    required = {2: "report_before", 3: "report_before", 4: "bundle", 5: "comparison", 6: "comparison"}
    if step in required and st.session_state.get(required[step]) is None:
        step = 1

    {1: step_input, 2: step_analysis, 3: step_gaps, 4: step_tailored,
     5: step_compare, 6: step_download}[step]()


if __name__ == "__main__":
    main()