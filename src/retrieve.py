import os
import json
import numpy as np
import ollama
import faiss


from sklearn.metrics.pairwise import cosine_similarity
from rag_setup import pull_ollama_model


# -----------------------------
# BASE RETRIEVAL
# -----------------------------
def base_retrieval(query_embedding_response, embeddings_path, chunks_path, top_n):

    query_embedding = np.array(query_embedding_response['embedding']).reshape(1, -1)

    if not os.path.exists(embeddings_path) or not os.path.exists(chunks_path):
        print("\nError: Embeddings or metadata file not found.")
        return []

    stored_embeddings = np.load(embeddings_path)

    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_scores = similarities[top_indices]

    with open(chunks_path, 'r') as f:
        all_chunks = json.load(f)

    retrieved_chunks = []

    #print(f"\n--- Top {top_n} Retrieved Documents ---")

    for i, idx in enumerate(top_indices):
        if idx < len(all_chunks):

            chunk = all_chunks[idx]
            score = float(top_scores[i])

            '''

            print(
                f"\nMatch {i+1} (Score: {score:.4f}) | "
                f"Chunk ID: {chunk.get('chunk_id')} | "
                f"Source: {chunk.get('source')} | "
                f"Text: {chunk.get('text')}"
            )
            '''

            retrieved_chunks.append({
                'score': score,
                **chunk
            })

        else:
            print(f"\nWarning: Index {idx} out of bounds")

    return retrieved_chunks


# -----------------------------
# FAISS RETRIEVAL
# -----------------------------
def faiss_retrieval(query_embedding_response, index, chunks_path, top_n):

    query_embedding = np.array(query_embedding_response['embedding']).astype('float32').reshape(1, -1)

    distances, indices = index.search(query_embedding, top_n)

    with open(chunks_path, 'r') as f:
        chunks = json.load(f)

    retrieved_chunks = []

    #print(f"\n--- Top {top_n} Retrieved Documents ---")

    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):

            chunk = chunks[idx]

            result = {
                'score': float(distances[0][i]),
                **chunk
            }

            retrieved_chunks.append(result)

    # Sort by distance (lower = better for L2)
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x['score'])
    '''
    for i, chunk in enumerate(sorted_chunks):
        print(
            f"\nMatch {i+1} (Score: {chunk['score']}) | "
            f"Chunk ID: {chunk['chunk_id']} | "
            f"Source: {chunk['source']} | "
            f"Text: {chunk['text']}"
        )
    '''
    
    return sorted_chunks


# -----------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------
def retrieve_chunks(query, base_path, top_n=15, method='base'):

    EMBEDDING_MODEL_NAME = 'nomic-embed-text'

    BASE_EMBEDDINGS_PATH = os.path.join(base_path, 'vector_store', 'embeddings.npy')
    BASE_METADATA_PATH = os.path.join(base_path, 'vector_store', 'base_metadata.json')

    FAISS_INDEX_PATH = os.path.join(base_path, 'vector_store', 'faiss_index.faiss')
    FAISS_METADATA_PATH = os.path.join(base_path, 'vector_store', 'faiss_metadata.json')

    # Load FAISS index only if needed
    FAISS_INDEX = None
    if method == 'faiss':
        if not os.path.exists(FAISS_INDEX_PATH):
            print("\nError: FAISS index file not found.")
            return []
        FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)

    pull_ollama_model(EMBEDDING_MODEL_NAME)

    #print(f"\nGenerating embedding for query: \n\n'{query}'")
    query_embedding_response = ollama.embeddings(
        model=EMBEDDING_MODEL_NAME,
        prompt=query
    )

    if method == 'base':
        print(f'[INFO] Retrieval method: BASE (COSINE SIMILARITY) | top_k: {top_n} | embedding_model: {EMBEDDING_MODEL_NAME}')
        return base_retrieval(
            query_embedding_response,
            BASE_EMBEDDINGS_PATH,
            BASE_METADATA_PATH,
            top_n
        )

    elif method == 'faiss':
        print(f'[INFO] Retrieval method: FAISS | top_k: {top_n} | embedding_model: {EMBEDDING_MODEL_NAME}')
        return faiss_retrieval(
            query_embedding_response,
            FAISS_INDEX,
            FAISS_METADATA_PATH,
            top_n
        )
    else:
        raise ValueError("\nmethod must be 'base' or 'faiss'")