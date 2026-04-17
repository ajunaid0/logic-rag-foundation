import json
import os
import time
import numpy as np
from tqdm import tqdm
import faiss
import ollama

from rag_setup import pull_ollama_model


# -----------------------------
# LOAD CHUNKS
# -----------------------------
def load_chunks_from_json(file_path):
    all_chunks = []
    try:
        with open(file_path, 'r') as f:
            all_chunks = json.load(f)

        print(f"\nLoaded {len(all_chunks)} chunks")

    except Exception as e:
        print(f"\nError loading chunks: {e}")

    return all_chunks


# -----------------------------
# BASE EMBEDDING PIPELINE
# -----------------------------
def embed_chunks_base(all_chunks, output_file_path, metadata_path, model_name, batch_size):

    client = ollama.Client(timeout=120)

    valid_chunks = []
    valid_embeddings = []

    total_chunks = len(all_chunks)

    for i in tqdm(range(0, total_chunks, batch_size)):

        batch = all_chunks[i:i + batch_size]

        print(f"\nProcessing batch {i} to {i + len(batch) - 1}")

        for chunk in batch:

            success = False

            for attempt in range(3):
                try:
                    res = client.embeddings(model=model_name, prompt=chunk['text'])
                    embedding = res['embedding']

                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)

                    success = True
                    break

                except Exception as e:
                    print(f"\nRetry {attempt+1} failed: {e}")
                    time.sleep(1)

            if not success:
                print(f"\nDROPPED chunk {chunk['chunk_id']}")

    embeddings_array = np.array(valid_embeddings).astype('float32')

    np.save(output_file_path, embeddings_array)

    with open(metadata_path, "w") as f:
        json.dump(valid_chunks, f)

    print(f"\nBase embeddings saved: {embeddings_array.shape}")

    return valid_chunks, embeddings_array


# -----------------------------
# FAISS PIPELINE
# -----------------------------
def embed_chunks_faiss(all_chunks, index_path, metadata_path, model_name, batch_size):

    client = ollama.Client(timeout=120)

    valid_chunks = []
    valid_embeddings = []

    total_chunks = len(all_chunks)

    dim = None

    for i in tqdm(range(0, total_chunks, batch_size)):

        batch = all_chunks[i:i + batch_size]

        print(f"\nProcessing batch {i} to {i + len(batch) - 1}")

        for chunk in batch:

            success = False

            for attempt in range(3):
                try:
                    res = client.embeddings(model=model_name, prompt=chunk['text'])
                    embedding = res['embedding']

                    if dim is None:
                        dim = len(embedding)

                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)

                    success = True
                    break

                except Exception as e:
                    print(f"\nRetry {attempt+1} failed: {e}")
                    time.sleep(1)

            if not success:
                print(f"\nDROPPED chunk {chunk['chunk_id']}")

    if len(valid_embeddings) == 0:
        raise ValueError("\nNo embeddings generated")

    embeddings_array = np.array(valid_embeddings).astype('float32')

    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    faiss.write_index(index, index_path)

    with open(metadata_path, "w") as f:
        json.dump(valid_chunks, f)

    print(f"\nFAISS index saved with {index.ntotal} vectors")

    return valid_chunks, index


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_embeddings(base_path, batch_size=100, method='base'):

    CHUNKS_PATH = os.path.join(base_path, 'vector_store', 'all_chunks.json')

    BASE_EMB_PATH = os.path.join(base_path, 'vector_store', 'embeddings.npy')
    BASE_META_PATH = os.path.join(base_path, 'vector_store', 'base_metadata.json')

    FAISS_INDEX_PATH = os.path.join(base_path, 'vector_store', 'faiss_index.faiss')
    FAISS_META_PATH = os.path.join(base_path, 'vector_store', 'faiss_metadata.json')

    MODEL_NAME = 'nomic-embed-text'

    all_chunks = load_chunks_from_json(CHUNKS_PATH)

    if not all_chunks:
        print("\nNo chunks found")
        return

    pull_ollama_model(MODEL_NAME)

    if method == 'base':

        print("----------------------------------")
        print(f'\n[INFO] Embedding Model: {MODEL_NAME} | Storage Method: Numpy Embeddings")')
        print("----------------------------------")

        return embed_chunks_base(
            all_chunks,
            BASE_EMB_PATH,
            BASE_META_PATH,
            MODEL_NAME,
            batch_size
        )

    elif method == 'faiss':

        print("----------------------------------")
        print(f'\n[INFO] Embedding Model: {MODEL_NAME} | Storage Method: FAISS Indices")')
        print("----------------------------------")

        return embed_chunks_faiss(
            all_chunks,
            FAISS_INDEX_PATH,
            FAISS_META_PATH,
            MODEL_NAME,
            batch_size
        )

    else:
        raise ValueError("\nmethod must be 'base' or 'faiss'")