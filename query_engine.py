# Import statements
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from db import get_connection
from langchain_ollama import OllamaLLM

# Initialize system components
def load_system():
    # Load chunks from DB
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text, source, page, chunk_index FROM chunks")
    rows = cursor.fetchall()

    # Convert to list of dicts
    chunk_objects = [
        {
            "text": r[0],
            "source": r[1],
            "page": r[2],
            "index": r[3]
        }
        for r in rows
    ]

    # Close DB and extract texts
    conn.close()
    texts = [c["text"] for c in chunk_objects]
    
    # Initialize embedder and reranker
    model = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Build FAISS index by encoding the texts
    if len(texts) > 0:
        embeddings = model.encode(texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    else:
        index = None
    
    # Initialize LLM
    llm = OllamaLLM(model="llama3.1:8b")
    # llm = OllamaLLM(model="qwen35-uncensored")

    # Return components, or default values if nothing in database
    if len(texts) == 0:
        return {
            "chunks": [],
            "texts": [],
            "model": model,
            "index": None,
            "llm": llm,
            "reranker": reranker
        }
    return {
        "chunks": chunk_objects,
        "texts": texts,
        "model": model,
        "index": index,
        "llm": llm,
        "reranker": reranker
    }

# Function to retrieve relevant chunks from the FAISS index based on a query
def query_faiss(query_text, system, top_k=3):
    # Failsafe if DB is empty
    if system["index"] is None:
        return []
    
    model = system["model"]
    index = system["index"]
    chunk_objects = system["chunks"]

    # Encode the given query to use for searching
    query_vec = model.encode([query_text], convert_to_numpy=True)

    # Search index for nearest neighbors
    distances, indices = index.search(query_vec, top_k)

    # return corresponding chunk objects
    return [chunk_objects[i] for i in indices[0]]

# Function to rerank retrieved chunks using the CrossEncoder based on relevance to the query
def rerank_chunks(query, chunks, system, top_k=5):
    # Initialize reranker and prepare pairs for scoring
    reranker = system["reranker"]
    pairs = [(query, c["text"]) for c in chunks]

    # Get relevance scores for each chunk and sort by score
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    # Return the chunks corresponding to the top K scores
    return [c for c, _ in ranked[:top_k]]

# Function to expand the selected chunks with their neighbors (previous and next chunks from the same source and page)
def expand_with_neighbors(selected_chunks, all_chunks, window=1):
    expanded = []
    seen = set()

    # For each chunk
    for chunk in selected_chunks:
        # Get its index, source, and page
        idx = chunk["index"]
        source = chunk["source"]
        page = chunk["page"]

        # for a window around that index
        for i in range(idx - window, idx + window + 1):
            # Skip if index is out of bounds
            if i < 0:
                continue

            # Find the chunk in all_chunks that matches this source, page, and index
            for c in all_chunks:
                if c["source"] == source and c["page"] == page and c["index"] == i:
                    key = (c["source"], c["page"], c["index"])
                    # Add it to the lists if we haven't seen this source+page+index before
                    if key not in seen:
                        seen.add(key)
                        expanded.append(c)

    # Return the expanded list of chunks, which includes the original selected chunks plus their neighbors
    return expanded

# Function to sort chunks by source, then page, then index within page
def sort_chunks(chunks):
    return sorted(chunks, key=lambda c: (c["source"], c["page"], c["index"]))

# Function to format chat history for LLM input
def format_chat_history(chat_history, max_messages=6):
    # Only look at last max_messages # of messages
    recent = chat_history[-max_messages:]
    formatted = []
    # For each message
    for msg in recent:
        # Tag it as user or assistant and include the content
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")

    # Join all messages into a single string with line breaks
    return "\n".join(formatted)

def generate_answer(query, retrieved_chunks, system, chat_history):
    # Initialize LLM and prepare context
    llm = system["llm"]
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    # Use the context, chat history, and query to generate an answer
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
    # Extract unique sources from the retrieved chunks
    sources = list(set([c["source"] for c in retrieved_chunks]))
    # Get the answer from the LLM 
    answer = llm.invoke(prompt)
    # Append the sources at the end and return
    return f"{answer}\n\nSources:\n" + "\n".join(sources)

# Main function to answer a user query
def answer_query(query: str, system, chat_history):
    # Formats the chat history, rewrites the query, uses the rewritten query to retrieve relevant chunks, reranks those chunks, expands them with neighbors, 
    # sorts them, and then generates an answer using the LLM.
    history_text = format_chat_history(chat_history)
    rewritten_query = rewrite_query(query, chat_history, system)
    retrieval_query = f"{rewritten_query}\n{query}"
    candidates = query_faiss(retrieval_query, system, top_k=20)
    top_chunks = rerank_chunks(retrieval_query, candidates, system, top_k=5)
    expanded_chunks = expand_with_neighbors(top_chunks, system["chunks"], window=1)
    final_chunks = sort_chunks(expanded_chunks)
    answer = generate_answer(query, final_chunks, system, history_text)
    return answer

# Function to rewrite the user's query into a standalone question using the LLM, based on the conversation history
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