import os
import json
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
RAW_DIR = "E:\\Main_pro\\NLPv3\\data\\raw"
OUTPUT_DIR = "E:\\Main_pro\\NLPv3\\data\\output"

def extract_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def process_pdf(pdf_path: str) -> list:
    raw_text = extract_text(pdf_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    doc_name = os.path.basename(pdf_path)
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "doc_name": doc_name,
            "chunk_id": i,
            "text": chunk
        })
    return result


def process_all_pdfs(raw_dir: str = RAW_DIR, output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    all_chunks = []
    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(raw_dir, pdf_file)
        chunks = process_pdf(pdf_path)
        all_chunks.extend(chunks)

    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    process_all_pdfs()
