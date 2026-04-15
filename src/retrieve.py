import os
import json
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from ollama_setup import pull_ollama_model


def retrieval(user_query, embedding_model_name, embeddings_path, chunks_path, top_n):
    import ollama

    # 1: Generate embeddings for the user query
    print(f"\nGenerating embeddings for query: '{user_query}'...")
    query_embedding_response = ollama.embeddings(model=embedding_model_name, prompt=user_query)
    query_embedding = np.array(query_embedding_response['embedding']).reshape(1, -1)

    # Check if files exist
    if not os.path.exists(embeddings_path) or not os.path.exists(chunks_path):
        print("Error: Embeddings or Chunks file not found.")
        return []

    # 2: Load stored embeddings
    stored_embeddings = np.load(embeddings_path)

    # 3: Find cosine similarity
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

    # 4 & 5: Get top N indices and scores
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_scores = similarities[top_indices]

    # 6: Load corresponding text chunks
    with open(chunks_path, 'r') as f:
        all_chunks = json.load(f)

    # 7: Collect full meta-information
    retrieved_chunks = []
    print(f"\n--- Top {top_n} Retrieved Documents ---")

    for i, idx in enumerate(top_indices):
        if idx < len(all_chunks):
            # Get the original chunk dictionary (contains id, source, path, char_start, text)
            full_chunk_data = all_chunks[idx]
            score = float(top_scores[i])

            print(f"\nMatch {i+1} (Score: {score:.4f}) | Source: {full_chunk_data.get('source')} | Chunk ID: {full_chunk_data.get('chunk_id')} ")

            # Create a combined dictionary: Score + all original metadata
            chunk_result = {
                'score': score,
                **full_chunk_data  # This "unpacks" id, text, source, path, char_start
            }

            retrieved_chunks.append(chunk_result)
        else:
            print(f"Warning: Index {idx} out of bounds.")

    print("----------------------------------")
    return retrieved_chunks

def get_top_k_chunks(query, base_path, top_n=15):
    """Main function to set up Ollama and perform retrieval with dynamic Top K."""
    print(f"Starting Retrieval process for Top {top_n} matches...")

    # Define parameters using dynamic paths
    EMBEDDING_MODEL_NAME = 'nomic-embed-text'
    EMBEDDINGS_PATH = os.path.join(base_path, 'vector_store', 'embeddings.npy')
    CHUNKS_PATH = os.path.join(base_path, 'vector_store', 'metadata.json')

    pull_ollama_model(EMBEDDING_MODEL_NAME)
    retrieved_chunks = retrieval(query, EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH, CHUNKS_PATH, top_n)
    print("Process complete.")
    return retrieved_chunks

