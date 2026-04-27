import json
import os
import time
import numpy as np
from tqdm import tqdm
import faiss
import ollama

# Assuming this exists in your local environment
from rag_setup import pull_ollama_model

# -----------------------------
# LOAD CHUNKS
# -----------------------------
def load_chunks_from_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            print(f"[INFO] Source: {file_path} | Chunks Loaded: {len(data)}")
            return data
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return []

# -----------------------------
# EMBEDDING CALL
# -----------------------------
def get_embedding(client, model_name, text):
    response = client.embeddings(
        model=model_name,
        prompt=text
    )
    return response["embedding"]

# -----------------------------
# SMART SPLIT (CHAR-AWARE)
# -----------------------------
def split_chunk(text, max_chars=800):
    if len(text) <= max_chars:
        return [text]

    parts = []
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if not para: continue

        if len(para) <= max_chars:
            parts.append(para)
            continue

        lines = para.split("\n")
        current = ""

        for line in lines:
            if len(current) + len(line) + 1 <= max_chars:
                current += (" " + line) if current else line
            else:
                if current:
                    parts.append(current.strip())
                if len(line) > max_chars:
                    for i in range(0, len(line), max_chars):
                        parts.append(line[i:i + max_chars])
                    current = ""
                else:
                    current = line
        if current:
            parts.append(current.strip())
    return parts

# -----------------------------
# EMBED WITH RECOVERY
# -----------------------------
def embed_chunk_with_recovery(client, model_name, chunk):
    text = chunk["text"]
    cid = chunk.get('chunk_id', 'N/A')

    # Try full chunk first
    for attempt in range(3):
        try:
            return get_embedding(client, model_name, text)
        except Exception:
            if attempt < 2:
                print(f"[RETRY] Chunk: {cid} | Attempt: {attempt+1}/3")
                time.sleep(1)

    # Split and Mean-Pool logic
    print(f"[SPLIT] Chunk: {cid} | Length: {len(text)} chars | Action: Recursive Mean-Pooling")
    parts = split_chunk(text)
    part_embeddings = []

    for part in parts:
        for attempt in range(3):
            try:
                emb = get_embedding(client, model_name, part)
                part_embeddings.append(emb)
                break
            except Exception:
                time.sleep(1)

    if not part_embeddings:
        print(f"[FAILED] Chunk: {cid} | Status: Dropped after split attempts")
        return None

    return np.mean(np.array(part_embeddings), axis=0).tolist()

# -----------------------------
# MAIN EMBEDDING LOOP
# -----------------------------
def generate_embeddings_only(all_chunks, model_name, batch_size):
    client = ollama.Client(timeout=120)
    embeddings = []
    valid_chunks = []
    total_chunks = len(all_chunks)

    progress = tqdm(total=total_chunks, desc="Processing Vectors")

    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        for chunk in batch:
            embedding = embed_chunk_with_recovery(client, model_name, chunk)
            if embedding is not None:
                embeddings.append(embedding)
                valid_chunks.append(chunk)
            progress.update(1)

    progress.close()

    if not embeddings:
        raise ValueError("[ERROR] Pipeline generated zero embeddings.")

    return valid_chunks, np.array(embeddings).astype("float32")

# -----------------------------
# STORAGE
# -----------------------------
def store_base_embeddings(chunks, embeddings, emb_path, meta_path):
    np.save(emb_path, embeddings)
    with open(meta_path, "w") as file:
        json.dump(chunks, file)
    print(f"[STORAGE] Base Assets: {os.path.basename(emb_path)} saved")

def store_faiss_embeddings(chunks, embeddings, index_path, meta_path):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as file:
        json.dump(chunks, file)
    print(f"[STORAGE] FAISS Index: {os.path.basename(index_path)} saved")
    return index

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def generate_embeddings(
    embedding_model_name="nomic-embed-text",
    base_path="/content/",
    batch_size=100,
    method="base"
):
    # Setup Paths
    vs_dir = os.path.join(base_path, "vector_store")
    os.makedirs(vs_dir, exist_ok=True)
    
    chunks_path = os.path.join(vs_dir, "all_chunks.json")
    base_emb_path = os.path.join(vs_dir, "embeddings.npy")
    base_meta_path = os.path.join(vs_dir, "base_metadata.json")
    faiss_index_path = os.path.join(vs_dir, "faiss_index.faiss")
    faiss_meta_path = os.path.join(vs_dir, "faiss_metadata.json")

    all_chunks = load_chunks_from_json(chunks_path)
    if not all_chunks:
        return

    # Initialization Log
    print(f"[INFO] Model: {embedding_model_name} | Method: {method.upper()} | Batch: {batch_size}")

    pull_ollama_model(embedding_model_name)

    chunks, embeddings = generate_embeddings_only(
        all_chunks,
        embedding_model_name,
        batch_size
    )

    # Success Log
    print(f"[SUCCESS] Final Count: {len(chunks)} | Vector Dim: {embeddings.shape[1]}")

    if method in ["base"]:
        store_base_embeddings(chunks, embeddings, base_emb_path, base_meta_path)

    if method in ["faiss"]:
        store_faiss_embeddings(chunks, embeddings, faiss_index_path, faiss_meta_path)

    else:
        raise ValueError("[ERROR] Method must be 'base' or 'faiss'")
    
    return chunks, embeddings