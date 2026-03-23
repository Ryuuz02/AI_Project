from pypdf import PdfReader
from pathlib import Path
import re
import pickle

def build_chunks(
    text: str,
    min_para_len=300,
    chunk_size=500,
    overlap=100
):
    if not text.strip():
        return []
    # -------------------------
    # Step 1: Clean PDF noise
    # -------------------------
    text = text.replace('\r', '')

    text = re.sub(r'\n{2,}', '<<<PARA>>>', text)
    text = re.sub(r'\n\s*', ' ', text)
    text = text.replace('<<<PARA>>>', '\n\n')
    text = re.sub(r'[ \t]+', ' ', text).strip()

    # -------------------------
    # Step 2: Paragraph handling
    # -------------------------
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(raw_paragraphs) > 1:
        paragraphs = raw_paragraphs
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)

        paragraphs = []
        current = ""

        for sentence in sentences:
            if not current:
                current = sentence
                continue

            if len(sentence) > 0 and len(current) > min_para_len and sentence[0].isupper():
                paragraphs.append(current.strip())
                current = sentence
            else:
                current += " " + sentence

        if current:
            paragraphs.append(current.strip())

    # -------------------------
    # Step 3: Split large paragraphs (WITH overlap)
    # -------------------------
    split_paragraphs = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            split_paragraphs.append(para)
            continue

        start = 0
        while start < len(para):
            end = start + chunk_size
            chunk = para[start:end]
            split_paragraphs.append(chunk.strip())

            start += chunk_size - overlap

    # -------------------------
    # Step 4: Final chunking (WITH overlap)
    # -------------------------
    chunks = []
    current_chunk = ""

    for para in split_paragraphs:
        if not current_chunk:
            current_chunk = para
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += " " + para
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + " " + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)

    return "\n".join(text)

def load_document(file_path: str) -> dict:
    path = Path(file_path)

    if path.suffix == ".txt":
        content = read_txt(file_path)
        links = []
    elif path.suffix == ".pdf":
        content = read_pdf(file_path)
        links = extract_links_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return {
        "filename": path.name,
        "content": content,
        "length": len(content),
        "links": links
    }

def extract_links_from_pdf(file_path: str):
    reader = PdfReader(file_path)
    links = []

    for page_num, page in enumerate(reader.pages):
        if "/Annots" in page:
            for annot in page["/Annots"]:
                obj = annot.get_object()

                if obj.get("/Subtype") == "/Link":
                    if "/A" in obj and "/URI" in obj["/A"]:
                        links.append({
                            "page": page_num,
                            "url": obj["/A"]["/URI"]
                        })

    return links

def retrieve_chunks(file_path: str, output_path="chunks.pkl", meta_path="metadata.pkl"):
    doc = load_document(file_path)

    if not doc["content"].strip():
        raise ValueError("Document content is empty")

    filename = doc["filename"]

    # Load metadata
    if Path(meta_path).exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = set()

    if filename in metadata:
        print(f"{filename} already processed. Skipping.")
        return []

    # Build chunks
    new_chunks = build_chunks(doc["content"])

    # Load existing chunks
    if Path(output_path).exists():
        with open(output_path, "rb") as f:
            existing_chunks = pickle.load(f)
    else:
        existing_chunks = []

    # Append
    all_chunks = existing_chunks + new_chunks

    with open(output_path, "wb") as f:
        pickle.dump(all_chunks, f)

    # Update metadata
    metadata.add(filename)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    return new_chunks

    return chunks