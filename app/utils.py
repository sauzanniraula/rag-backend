# Helper functions (Chunking, Text extraction)

import pdfplumber
from io import BytesIO
from typing import List

# Extraction Logic
def extract_text(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return file_bytes.decode("utf-8")

# Chunking Strategy 1: Fixed Size
def fixed_chunking(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size - overlap)]

# Chunking Strategy 2: Recursive (Paragraph based)
def recursive_chunking(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 10]