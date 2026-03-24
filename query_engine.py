from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
from langchain_ollama import OllamaLLM



# -------------------------
# Load everything ONCE
# -------------------------
def load_system():
    # Load chunks
    with open("chunks.pkl", "rb") as f:
        chunk_objects = pickle.load(f)
        texts = [c["text"] for c in chunk_objects]

    # Embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Create embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # LLM
    llm = OllamaLLM(model="llama3.1:8b")
    # llm = OllamaLLM(model="qwen35-uncensored")

    return {
        "chunks": chunk_objects,
        "texts": texts,
        "model": model,
        "index": index,
        "llm": llm,
        "reranker": reranker
    }


# -------------------------
# Retrieval
# -------------------------
def query_faiss(query_text, system, top_k=3):
    model = system["model"]
    index = system["index"]
    chunk_objects = system["chunks"]

    query_vec = model.encode([query_text], convert_to_numpy=True)

    distances, indices = index.search(query_vec, top_k)

    return [chunk_objects[i] for i in indices[0]]

def rerank_chunks(query, chunks, system, top_k=5):
    reranker = system["reranker"]

    pairs = [(query, c["text"]) for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    return [c for c, _ in ranked[:top_k]]

def expand_with_neighbors(selected_chunks, all_chunks, window=1):
    expanded = []
    seen = set()

    for chunk in selected_chunks:
        idx = chunk["index"]
        source = chunk["source"]
        page = chunk["page"]

        for i in range(idx - window, idx + window + 1):
            if i < 0:
                continue

            # find matching chunk (same source + index)
            for c in all_chunks:
                if c["source"] == source and c["page"] == page and c["index"] == i:
                    key = (c["source"], c["index"])
                    if key not in seen:
                        seen.add(key)
                        expanded.append(c)

    return expanded

def sort_chunks(chunks):
    return sorted(chunks, key=lambda c: (c["source"], c["index"]))
# -------------------------
# LLM
# -------------------------
def format_chat_history(chat_history, max_messages=6):
    recent = chat_history[-max_messages:]

    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")

    return "\n".join(formatted)

def generate_answer(query, retrieved_chunks, system, chat_history):
    llm = system["llm"]

    context = "\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""
You are a strict assistant.

- Use conversation history for context
- Only use the provided context for factual answers
- Do not guess
- If unsure, say "I don't know"

Conversation:
{chat_history}

Context:
{context}

Question:
{query}

Answer:
"""
    sources = list(set([c["source"] for c in retrieved_chunks]))
    answer = llm.invoke(prompt)
    return f"{answer}\n\nSources:\n" + "\n".join(sources)


# -------------------------
# Main query function
# -------------------------
def answer_query(query: str, system, chat_history):
    history_text = format_chat_history(chat_history)

    rewritten_query = rewrite_query(query, chat_history, system)
    retrieval_query = f"{rewritten_query}\n{query}"
        # Step 1: High recall
    candidates = query_faiss(retrieval_query, system, top_k=20)

    # Step 2: Rerank
    top_chunks = rerank_chunks(retrieval_query, candidates, system, top_k=5)

    # Step 3: Add neighbors
    expanded_chunks = expand_with_neighbors(top_chunks, system["chunks"], window=1)

    # Step 4: Sort for coherence
    final_chunks = sort_chunks(expanded_chunks)


    answer = generate_answer(query, final_chunks, system, history_text)

    return answer

def rewrite_query(query, chat_history, system):
    llm = system["llm"]

    history_text = format_chat_history(chat_history)

    prompt = f"""
Rewrite the user's question into a standalone question.

STRICT RULES:
- Preserve the exact intent (e.g., before, after, compare, etc.)
- Resolve references like "that line" using conversation history
- Do NOT generalize or simplify the question
- Do NOT answer the question
- ONLY include the question in your response, nothing else

Conversation:
{history_text}

User question:
{query}

Rewritten question:
"""
    rewritten_query = llm.invoke(prompt).strip()
    return rewritten_query