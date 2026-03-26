# Import statements
from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from query_engine import answer_query, load_system
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from document_retrieval import retrieve_chunks
import os
from db import get_connection, init_db

# Initialize the FastAPI app, database, static files, templates, system components, and an empty chat history
app = FastAPI()
init_db()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
system = load_system()
chat_history = []

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models for request and response bodies
class QueryRequest(BaseModel):
    query: str
class QueryResponse(BaseModel):
    answer: str

# At the root endpoint, return a simple status message to confirm the API is running
@app.get("/")
def root():
    return {"status": "RAG API is running"}

# At the /documents endpoint, return a list of all processed document filenames from the database
@app.get("/documents")
def list_documents():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM documents")
    docs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return {"documents": docs}

# At the /upload endpoint, accept a file upload, save it to the uploads directory, process it with the retrieve_chunks function, and return a status message 
# confirming the upload and processing
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    retrieve_chunks(file_path)
    return {"status": f"{file.filename} uploaded and processed"}

# At the /query endpoint, accept a query in the request body, append it to the chat history, load the system components, 
# call the answer_query function with the user query, system components, and chat history, append the answer to the chat history, 
# and return the answer in the response body
@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    # Use global chat_history to maintain the conversation context across multiple queries
    global chat_history
    user_query = request.query
    chat_history.append({"role": "user", "content": user_query})
    system = load_system()
    answer = answer_query(user_query, system, chat_history)
    chat_history.append({"role": "assistant", "content": answer})
    return {"answer": answer}

# At the /app endpoint, return the app.html template to serve the frontend user interface
@app.get("/app", response_class=HTMLResponse)
def app_ui(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})