import streamlit as st
from parser import parse_resume
from analyzer import analyze_resume

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown("Upload your resume + paste a job description → get a match score, missing skills, and improved bullet points.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader("PDF or DOCX", type=["pdf", "docx"])

with col2:
    st.subheader("Paste Job Description")
    job_desc = st.text_area("Job Description", height=300, placeholder="Paste the full job posting here...")

if st.button("🔍 Analyze Resume", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload a resume.")
    elif not job_desc.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analyzing your resume..."):
            resume_text = parse_resume(uploaded_file)
            result = analyze_resume(resume_text, job_desc)

        # Match Score
        st.markdown("---")
        score = result.match_score
        color = "green" if score >= 70 else "orange" if score >= 50 else "red"
        st.markdown(f"## Match Score: :{color}[{score}/100]")
        st.progress(score / 100)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("✅ Matching Keywords")
            for kw in result.matching_keywords:
                st.markdown(f"- `{kw}`")

            st.subheader("💪 Strengths")
            for s in result.strengths:
                st.markdown(f"- {s}")

        with col4:
            st.subheader("❌ Missing Keywords")
            for kw in result.missing_keywords:
                st.markdown(f"- `{kw}`")

            st.subheader("⚠️ Weaknesses")
            for w in result.weaknesses:
                st.markdown(f"- {w}")

        st.subheader("✍️ Improved Bullet Points")
        st.markdown("*Here are your weak bullets rewritten to be stronger and ATS-friendly:*")
        for i, bullet in enumerate(result.improved_bullets, 1):
            st.success(f"{i}. {bullet}")