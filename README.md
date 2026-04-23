# 📄 AI Resume Analyzer

An AI-powered resume analyzer that compares your resume against a job description and gives you a match score, missing keywords, and rewrites your weak bullet points to be stronger and ATS-friendly.

## 🔴 Live Demo
👉 [Try it on Hugging Face](https://huggingface.co/spaces/artistica-004/ai-resume-analyzer)

---

## 🚀 Features

- 📊 **Match Score** — Get a score out of 100 showing how well your resume matches the job description
- 🔑 **Missing Keywords** — See exactly which important keywords from the JD are missing in your resume
- ✅ **Matching Keywords** — Know what skills you already have that align with the role
- 💪 **Strengths & Weaknesses** — Get honest feedback on what's working and what's not
- ✍️ **Improved Bullet Points** — AI rewrites your weak bullets to be stronger and ATS-optimized

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Groq API (Llama-3.3-70b) | LLM for analysis and rewriting |
| Pydantic | Structured JSON output validation |
| PyPDF2 | PDF text extraction |
| python-docx | DOCX text extraction |
| Streamlit | Frontend UI |
| Hugging Face Spaces | Deployment |

---

## 📁 Project Structure
ai-resume-analyzer/
├── app.py          # Streamlit UI
├── analyzer.py     # LLM logic + Pydantic models
├── parser.py       # PDF/DOCX text extraction
├── requirements.txt
├── .env            # API key (never push this)
└── README.md

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/artistica-004/ai-resume-analyzer.git
cd ai-resume-analyzer
```

**2. Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the root folder:

GROQ_API_KEY=your_groq_api_key_here

Get your free API key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

---

## 📌 How It Works

1. User uploads their resume (PDF or DOCX)
2. User pastes the target job description
3. App extracts text from the resume using PyPDF2 / python-docx
4. Resume + JD are sent to Groq LLM with a structured prompt
5. Pydantic validates the JSON response into 6 fields
6. Results are displayed in a clean Streamlit UI

---

## 🧠 Key Concepts Demonstrated

- **Structured LLM output** using Pydantic for reliable JSON extraction
- **Prompt engineering** for multi-field structured data extraction
- **PDF/DOCX parsing** in Python
- **Streamlit** file upload and dynamic UI rendering
- **Secure API key management** via environment variables

---

## 🌐 Deployment

Deployed on **Hugging Face Spaces** using Streamlit SDK.
API key stored securely as a repository secret.

---

## 👩‍💻 Author

**Shivani Chaudhary**
[GitHub](https://github.com/artistica-004)

---

## 📜 License
MIT License

