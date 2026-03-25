from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from query_engine import answer_query, load_system
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from document_retrieval import retrieve_chunks
import os
from db import get_connection, init_db



app = FastAPI()
init_db()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
system = load_system()
chat_history = []


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.get("/documents")
def list_documents():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT filename FROM documents")
    docs = [row[0] for row in cursor.fetchall()]

    conn.close()

    return {"documents": docs}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process into chunks
    retrieve_chunks(file_path)

    return {"status": f"{file.filename} uploaded and processed"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    global chat_history

    user_query = request.query

    # Add user message
    chat_history.append({"role": "user", "content": user_query})

    system = load_system()

    answer = answer_query(user_query, system, chat_history)

    # Add assistant response
    chat_history.append({"role": "assistant", "content": answer})

    return {"answer": answer}

@app.get("/app", response_class=HTMLResponse)
def app_ui(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})