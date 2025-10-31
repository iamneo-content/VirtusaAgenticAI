import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from PDF file"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        text = f"[Error extracting text: {e}]"
    return text.strip()