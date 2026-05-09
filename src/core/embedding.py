import json
import os
import time
import numpy as np
from tqdm import tqdm
import faiss
import ollama
import spacy

# ==========================================================
# SPACY MODEL
#==========================================================

nlp = spacy.load("en_core_web_sm")

# -----------------------------
# HELPERS (Original Logic)
# -----------------------------

def load_chunks_from_json(file_path):
    """Original Logic: Loads the initial chunk file created by ingestion."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            print(f"[INFO] Source: {file_path} | Chunks Loaded: {len(data)}")
            return data
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return []

def get_embedding(client, model_name, text):
    """Original Logic: Direct call to Ollama embedding API."""
    response = client.embeddings(model=model_name, prompt=text)
    return response["embedding"]

def split_chunk(text, max_chars=800):
    """Original Logic: Smart split to ensure text fits embedding limits."""
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
                if current: parts.append(current.strip())
                if len(line) > max_chars:
                    for i in range(0, len(line), max_chars):
                        parts.append(line[i:i + max_chars])
                    current = ""
                else:
                    current = line
        if current: parts.append(current.strip())
    return parts

# -----------------------------
# RECOVERY LOGIC (Original Logic)
# -----------------------------

def get_sentences_spacy(text):
    """Returns a list of sentence strings from text using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def embed_chunk_with_recovery(client, model_name, chunk):
    text = chunk["text"]
    cid = chunk.get('chunk_id', 'N/A')

    # Try full chunk first
    for attempt in range(3):
        try:
            return get_embedding(client, model_name, text)
        except Exception:
            time.sleep(1)

    # RECOVERY MODE: Use Sentence-Aware Splitting instead of character splitting
    print(f"\n[SPLIT] Chunk: {cid} | Length: {len(text)} | Recursive Mean-Pooling")
    
    # Use spaCy to get sentences instead of raw split
    sentences = get_sentences_spacy(text)
    
    parts = []
    current_part = ""
    # Re-pack sentences into smaller 800-char sub-chunks
    for sent in sentences:
        if len(current_part) + len(sent) < 800:
            current_part += " " + sent
        else:
            if current_part: parts.append(current_part.strip())
            current_part = sent
    if current_part: parts.append(current_part.strip())

    part_embeddings = []
    part_weights = []

    for part in parts:
        for attempt in range(3):
            try:
                emb = get_embedding(client, model_name, part)
                part_embeddings.append(emb)
                part_weights.append(len(part)) # Store length for weighting
                break
            except Exception:
                time.sleep(1)

    if not part_embeddings:
        return None

    # Weighted Mean Pooling: Ensures longer segments have more "vote" in the vector
    embs = np.array(part_embeddings)
    weights = np.array(part_weights).reshape(-1, 1)
    weighted_mean = np.sum(embs * weights, axis=0) / np.sum(weights)
    
    return weighted_mean.tolist()

# -----------------------------
# PIPELINE & STORAGE
# -----------------------------

def generate_embeddings_only(all_chunks, model_name, batch_size):
    """Original Logic: The main loop for processing chunks into vectors."""
    client = ollama.Client(timeout=120)
    embeddings = []
    valid_chunks = []
    
    progress = tqdm(total=len(all_chunks), desc="Processing Vectors")
    for i in range(0, len(all_chunks), batch_size):
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

def store_assets(chunks, embeddings, config):
    """Original Logic: Saves data to either Numpy or FAISS Index (IP)."""
    vs_dir = config.VECTOR_STORE_DIR
    method=config.RETRIEVAL_METHOD
    
    if method == "base":
        emb_path = os.path.join(vs_dir, "embeddings.npy")
        meta_path = os.path.join(vs_dir, "base_metadata.json")
        np.save(emb_path, embeddings)
        with open(meta_path, "w") as f:
            json.dump(chunks, f)
        print(f"[STORAGE] Base Assets Saved")

    elif method == "faiss":
        index_path = os.path.join(vs_dir, "faiss_index.faiss")
        meta_path = os.path.join(vs_dir, "faiss_metadata.json")
        faiss.normalize_L2(embeddings)
        # Using IndexFlatIP as requested for Cosine-equivalent logic
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, index_path)
        with open(meta_path, "w") as f:
            json.dump(chunks, f)
        print(f"[STORAGE] FAISS Assets Saved")

# -----------------------------
# ENTRY POINT
# -----------------------------

def generate_embeddings(config):
    """Original Logic: Orchestrates the entire embedding workflow."""
    vs_dir = config.VECTOR_STORE_DIR
    chunks_path = os.path.join(vs_dir, "all_chunks.json")

    all_chunks = load_chunks_from_json(chunks_path)
    if not all_chunks: return

    # Embedding logic
    valid_chunks, embeddings_array = generate_embeddings_only(
        all_chunks, config.EMBEDDING_MODEL, config.EMBEDDING_BATCH_SIZE
    )

    # Storage logic
    store_assets(valid_chunks, embeddings_array, config)
    
    return valid_chunks, embeddings_array