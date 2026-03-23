from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain_ollama import OllamaLLM


# -------------------------
# Load everything ONCE
# -------------------------
def load_system():
    # Load chunks
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # LLM
    llm = OllamaLLM(model="llama3.1:8b")
    # llm = OllamaLLM(model="qwen35-uncensored")

    return {
        "chunks": chunks,
        "model": model,
        "index": index,
        "llm": llm
    }


# -------------------------
# Retrieval
# -------------------------
def query_faiss(query_text, system, top_k=3):
    model = system["model"]
    index = system["index"]
    chunks = system["chunks"]

    query_vec = model.encode([query_text], convert_to_numpy=True)

    distances, indices = index.search(query_vec, top_k)

    return [chunks[i] for i in indices[0]]


# -------------------------
# LLM
# -------------------------
def generate_answer(query, retrieved_chunks, system):
    llm = system["llm"]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a strict assistant.

- Only use the provided context
- Do not guess
- If unsure, say "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    return llm.invoke(prompt)


# -------------------------
# Main query function
# -------------------------
def answer_query(query: str, system):
    retrieved = query_faiss(query, system, top_k=5)
    answer = generate_answer(query, retrieved, system)

    return answer

system = load_system()

result = answer_query(
    "What projects did Jacob work on?",
    system
)

print(result)