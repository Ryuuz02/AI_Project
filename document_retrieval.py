from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import get_connection

def build_chunks(text: str, chunk_size=400, overlap=100):
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators = [
        "\n\n",      # paragraphs
        "\n",        # lines (important for lyrics)
        r"(?<=\.) ", # sentences
        " "
]
    )

    return splitter.split_text(text)

def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path: str):
    reader = PdfReader(file_path)
    pages = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages.append((page_num, page_text))

    return pages
def load_document(file_path: str) -> dict:
    path = Path(file_path)

    if path.suffix == ".txt":
        content = read_txt(file_path)
        pages = [(0, content)]  # treat as single page
        links = []

    elif path.suffix == ".pdf":
        pages = read_pdf(file_path)
        content = "\n".join([p[1] for p in pages])  # optional
        links = extract_links_from_pdf(file_path)

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return {
        "filename": path.name,
        "pages": pages,   # 🔥 NEW
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

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM documents WHERE filename = ?", (filename,))
    if cursor.fetchone():
        print(f"{filename} already processed. Skipping.")
        conn.close()
        return []

    # Build chunks
    chunks = []

    for page_num, page_text in doc["pages"]:
        page_chunks = build_chunks(page_text)

        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "source": doc["filename"],
                "page": page_num,   # 🔥 NEW
                "index": i          # index within page
            })

    # Save document
    cursor.execute(
        "INSERT INTO documents (filename) VALUES (?)",
        (filename,)
    )

    # Save chunks
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (text, source, page, chunk_index)
            VALUES (?, ?, ?, ?)
        """, (
            chunk["text"],
            chunk["source"],
            chunk["page"],
            chunk["index"]
        ))

    conn.commit()
    conn.close()

    return chunks

# Use this to recreate chunks.pkl and metadata.pkl from the PDFs and TXTs in the project
# retrieve_chunks("PDFs\cover_letter_guide.pdf")