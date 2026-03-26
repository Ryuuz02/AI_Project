# Import statements
from pypdf import PdfReader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db import get_connection

# Function to split text into chunks with specified size and overlap, using a hierarchy of separators
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
        " "])
    return splitter.split_text(text)

# Function that reads .txt files
def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Function that reads .pdf files and extracts text from each page, returning a list of (page_number, text) tuples
def read_pdf(file_path: str):
    reader = PdfReader(file_path)
    pages = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages.append((page_num, page_text))

    return pages

# Function to load a document (either .txt or .pdf), extract its content, and return a structured dictionary with filename, content, length, and links (for PDFs)
def load_document(file_path: str) -> dict:
    path = Path(file_path)

    # If it's a .txt file
    if path.suffix == ".txt":
        content = read_txt(file_path)
        pages = [(0, content)]
        links = []

    # If it's a .pdf file
    elif path.suffix == ".pdf":
        pages = read_pdf(file_path)
        content = "\n".join([p[1] for p in pages])
        links = extract_links_from_pdf(file_path)

    # If it's an unsupported file type
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Return the information
    return {
        "filename": path.name,
        "pages": pages,
        "content": content,
        "length": len(content),
        "links": links
    }

# Function to extract hyperlinks from a PDF file, returning a list of dictionaries with page number and URL
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

# main function to load a document, split it into chunks, and store those chunks in the database, while avoiding duplicates based on filename
def retrieve_chunks(file_path: str):
    doc = load_document(file_path)

    if not doc["content"].strip():
        raise ValueError("Document content is empty")

    # Get the file name and Initialize DB connection
    filename = doc["filename"]
    conn = get_connection()
    cursor = conn.cursor()

    # if it's already in the DB
    cursor.execute("SELECT 1 FROM documents WHERE filename = ?", (filename,))
    if cursor.fetchone():
        # Skip it and tell the user
        print(f"{filename} already processed. Skipping.")
        conn.close()
        return []
    
    chunks = []

    # For each page
    for page_num, page_text in doc["pages"]:
        # Uses the text to create chunks
        page_chunks = build_chunks(page_text)

        # For each chunk
        for i, chunk in enumerate(page_chunks):
            # Creates a dictionary with the chunk text, source filename, page number, and chunk index, and appends it to the list of chunks
            chunks.append({
                "text": chunk,
                "source": doc["filename"],
                "page": page_num, 
                "index": i          
            })

    # Add the filename to the documents stored
    cursor.execute(
        "INSERT INTO documents (filename) VALUES (?)",
        (filename,)
    )

    # For each chunk
    for chunk in chunks:
        # Add it to the chunk table in the database
        cursor.execute("""
            INSERT INTO chunks (text, source, page, chunk_index)
            VALUES (?, ?, ?, ?)
        """, (
            chunk["text"],
            chunk["source"],
            chunk["page"],
            chunk["index"]
        ))

    # Commit changes and close the database connection
    conn.commit()
    conn.close()

    return chunks