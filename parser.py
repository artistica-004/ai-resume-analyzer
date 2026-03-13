import PyPDF2
import docx
import io

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def parse_resume(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        return None