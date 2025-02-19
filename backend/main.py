import os
import shutil
import pdfplumber
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import logging

# ================================
# FastAPI & Logging Setup
# ================================
app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ================================
# PDF Extraction
# ================================
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF via pdfplumber.
    Returns the extracted text as a string.
    """
    logger.debug(f"Extracting text from PDF: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    logger.debug(f"Extracted text length: {len(text)}")
    return text

# ================================
# FastAPI Routes
# ================================
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Saves the PDF to disk.
    """
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "message": "File uploaded successfully"}

@app.post("/analyze/")
async def analyze_document(file: UploadFile = File(...)):
    """
    1. Save PDF
    2. Extract text
    3. Send text to arbitrary REST API endpoint
    """
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text_from_pdf(file_path)
        response = requests.post("http://example.com/api/endpoint", json={"text": text})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.exception("Analysis failed.")
        return {"error": str(e)}
