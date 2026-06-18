import os
import pandas as pd

from pypdf import PdfReader
from docx import Document

UPLOAD_FOLDER = "data/uploads"

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv"
}


def ensure_upload_folder():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def is_valid_file(filename):
    extension = os.path.splitext(filename)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def save_uploaded_file(uploaded_file):
    ensure_upload_folder()

    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


# ----------------------------
# TEXT EXTRACTION FUNCTIONS
# ----------------------------

def extract_pdf_text(file_path):

    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx_text(file_path):

    document = Document(file_path)

    paragraphs = []

    for para in document.paragraphs:
        paragraphs.append(para.text)

    return "\n".join(paragraphs)


def extract_txt_text(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        return f.read()


def extract_csv_text(file_path):

    df = pd.read_csv(file_path)

    return df.to_string(index=False)


def extract_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    elif extension == ".docx":
        return extract_docx_text(file_path)

    elif extension == ".txt":
        return extract_txt_text(file_path)

    elif extension == ".csv":
        return extract_csv_text(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )