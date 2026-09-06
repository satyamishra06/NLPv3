import os
import json
import pdfplumber
import docx
from bs4 import BeautifulSoup
import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
RAW_DIR = "E:\\Main_pro\\NLPv3\\data\\raw"
OUTPUT_DIR = "E:\\Main_pro\\NLPv3\\data\\output"
SUPPORTED_FORMATS = (".pdf", ".docx", ".txt", ".html", ".md")


def extract_text_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text
def extract_text_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def extract_text_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n", strip=True)
def extract_text_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text_pdf(file_path)
    elif ext == ".docx":
        return extract_text_docx(file_path)
    elif ext == ".txt":
        return extract_text_txt(file_path)
    elif ext == ".html":
        return extract_text_html(file_path)
    elif ext == ".md":
        return extract_text_md(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


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


def process_document(file_path: str) -> list:
    raw_text = extract_text(file_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    doc_name = os.path.basename(file_path)
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "doc_name": doc_name,
            "chunk_id": i,
            "text": chunk
        })
    return result


def process_all_documents(raw_dir: str = RAW_DIR, output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    all_chunks = []
    all_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(SUPPORTED_FORMATS)]

    if not all_files:
        print(f"No supported files found in {raw_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return

    for file_name in all_files:
        file_path = os.path.join(raw_dir, file_name)
        print(f"Processing: {file_name}")
        try:
            chunks = process_document(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  -> Error: {e}")

    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    process_all_documents()
