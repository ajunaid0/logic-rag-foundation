import os
import json
import numpy as np
import ollama
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from rag_setup import pull_ollama_model

# -----------------------------
# LOAD CHUNKS
# -----------------------------
def load_chunks(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {file_path} | {e}")
        return []

# -----------------------------
# GET QUERY EMBEDDING
# -----------------------------
def get_query_embedding(model_name, query):
    response = ollama.embeddings(
        model=model_name,
        prompt=query
    )
    return response["embedding"]

# -----------------------------
# BASE RETRIEVAL (Cosine Similarity)
# -----------------------------
def base_retrieval(query_embedding, embeddings, chunks, top_n):
    query_vec = np.array(query_embedding).reshape(1, -1)
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "source": chunk.get("metadata", {}).get("source"),
            "path": chunk.get("metadata", {}).get("path")
        })
    return results

# -----------------------------
# FAISS RETRIEVAL (Index Search)
# -----------------------------
def faiss_retrieval(query_embedding, index, chunks, top_n):
    query_vec = np.array(query_embedding).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)
    
    distances, indices = index.search(query_vec, top_n)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx != -1 and idx < len(chunks):
            chunk = chunks[idx]
            results.append({
                "score": float(distances[0][i]),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text"),
                "source": chunk.get("metadata", {}).get("source"),
                "path": chunk.get("metadata", {}).get("path")
            })
    return results

# -----------------------------
# MAIN RETRIEVAL PIPELINE
# -----------------------------
def retrieve_chunks(
    embedding_model_name="nomic-embed-text",
    query="",
    base_path="/content/",
    top_n=5,
    method="base"
):
    # Setup Paths
    vs_dir = os.path.join(base_path, "vector_store")
    BASE_META = os.path.join(vs_dir, "base_metadata.json")
    FAISS_META = os.path.join(vs_dir, "faiss_metadata.json")
    BASE_EMB = os.path.join(vs_dir, "embeddings.npy")
    FAISS_IDX = os.path.join(vs_dir, "faiss_index.faiss")

    # Log Initialization
    #print(f"[INFO] Query: '{query[:50]}...' | Method: {method.upper()} | Model: {embedding_model_name}")

    pull_ollama_model(embedding_model_name)
    q_emb = get_query_embedding(embedding_model_name, query)

    # 1. BASE MODE
    if method == "base":
        chunks = load_chunks(BASE_META)
        embeddings = np.load(BASE_EMB)
        results = base_retrieval(q_emb, embeddings, chunks, top_n)
        print(f"[SUCCESS] Base Search Complete | Top Match Score: {results[0]['score']:.4f}")
        return results

    # 2. FAISS MODE
    elif method == "faiss":
        if not os.path.exists(FAISS_IDX):
            print(f"[ERROR] Index not found at {FAISS_IDX}")
            return []
        
        chunks = load_chunks(FAISS_META)
        index = faiss.read_index(FAISS_IDX)
        results = faiss_retrieval(q_emb, index, chunks, top_n)
        print(f"[SUCCESS] FAISS Search Complete | Top Match Distance: {results[0]['score']:.4f}")
        return results

    else:
        raise ValueError("[ERROR] Method must be 'base' or 'faiss'")