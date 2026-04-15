import json
import os
import numpy as np
from tqdm import tqdm

from ollama_setup import pull_ollama_model

def load_chunks_from_json(file_path):
    """Loads chunk data from a JSON file and handles potential errors."""
    all_chunks = []
    try:
        with open(file_path, 'r') as f:
            all_chunks = json.load(f)
        print(f"Successfully loaded {len(all_chunks)} chunks from {file_path}")
        print("First 5 chunks:")
        for i, chunk in enumerate(all_chunks[:5]):
            print(f"Chunk {i+1}: {chunk}")
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found. Please ensure your Google Drive is mounted and the path is correct.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file at {file_path}. Please check if the file contains valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return all_chunks

def process_and_embed_chunks(all_chunks, output_file_path, model_name='nomic-embed-text', batch_size=500):
    """Generates and saves embeddings for chunks in batches."""
    import ollama # Imported here as requested
    start_index = 0
    existing_embeddings_array = None

    if os.path.exists(output_file_path):
        try:
            existing_embeddings_array = np.load(output_file_path)
            start_index = existing_embeddings_array.shape[0]
            print(f"Found existing embeddings file at {output_file_path}.")
            print(f"Resuming from chunk index {start_index}. {start_index} embeddings already processed and saved.")
        except Exception as e:
            print(f"Error loading existing embeddings file: {e}. Starting from the beginning.")
            existing_embeddings_array = None
            start_index = 0
    else:
        print("No existing embeddings file found. Starting from the beginning.")

    total_chunks = len(all_chunks)
    print(f"Total chunks in dataset: {total_chunks}")
    print(f"Chunks remaining to process: {total_chunks - start_index}")

    # Loop through chunks in batches, starting from start_index
    for i in tqdm(range(start_index, total_chunks, batch_size), desc="Generating embeddings in batches"):
        client = ollama.Client(timeout=60)

        batch_start = i
        batch_end = min(i + batch_size, total_chunks)
        current_batch_chunks_to_process = all_chunks[batch_start:batch_end]

        batch_embeddings_list_for_numpy = []
        processed_count_in_batch = 0

        print(f"Processing chunks from index {batch_start} to {batch_end-1}...")
        for chunk_idx_in_batch, chunk in enumerate(current_batch_chunks_to_process):
            actual_chunk_index = batch_start + chunk_idx_in_batch
            try:
                embeddings_response = client.embeddings(model=model_name, prompt=chunk['text'])
                embedding_vector = embeddings_response['embedding']
                batch_embeddings_list_for_numpy.append(embedding_vector)
                all_chunks[actual_chunk_index]['embedding'] = embedding_vector
                processed_count_in_batch += 1
            except Exception as e:
                print(f"Error processing chunk {chunk['chunk_id']} (actual index {actual_chunk_index}): {e}")
                print("Skipping this chunk's embedding for numpy file and all_chunks for this run.")

        if batch_embeddings_list_for_numpy:
            current_batch_embeddings_array = np.array(batch_embeddings_list_for_numpy)
            if existing_embeddings_array is None:
                np.save(output_file_path, current_batch_embeddings_array)
                existing_embeddings_array = current_batch_embeddings_array
            else:
                existing_embeddings_array = np.concatenate((existing_embeddings_array, current_batch_embeddings_array))
                np.save(output_file_path, existing_embeddings_array)

            print(f"Batch {batch_start}-{batch_end-1} processed. {processed_count_in_batch} embeddings generated and saved. Current total in file: {existing_embeddings_array.shape[0]}")
        else:
            print(f"No embeddings generated for batch {batch_start} to {batch_end-1}. Likely due to errors. Continuing.")

    print(f"All available chunks processing attempt complete.")
    if existing_embeddings_array is not None:
        print(f"Final shape of embeddings array saved to {output_file_path}: {existing_embeddings_array.shape}")
    else:
        print("No embeddings were successfully saved to the numpy file.")

    print("\nFirst chunk with embedding (if available):")
    print({
        k: v for k, v in all_chunks[0].items() if k != 'embedding' or (k == 'embedding' and 'embedding' in all_chunks[0] and len(v) < 100)
    })
    return all_chunks, existing_embeddings_array

def generate_embeddings(base_path,batch_size=500):
    """
    Main function to generate embeddings using a relative base_path.
    """
    # Define file paths dynamically using base_path
    CHUNKS_FILE_PATH = os.path.join(base_path, 'vector_store', 'metadata.json')
    EMBEDDINGS_OUTPUT_PATH = os.path.join(base_path, 'vector_store', 'embeddings.npy')
    EMBEDDING_MODEL_NAME = 'nomic-embed-text'
    BATCH_SIZE = batch_size

    # Execute the workflow
    # Note: For Colab execution, 'google.colab.drive.mount' might be needed.
    # It's omitted here as the request is for a .py file where it's not applicable.

    all_chunks_data = load_chunks_from_json(CHUNKS_FILE_PATH)

    if all_chunks_data: 
        pull_ollama_model(EMBEDDING_MODEL_NAME)
        all_chunks_with_embeddings, final_embeddings = process_and_embed_chunks(
            all_chunks_data, EMBEDDINGS_OUTPUT_PATH, EMBEDDING_MODEL_NAME, BATCH_SIZE
        )
        print("Embedding process completed.")
    else:
        print("Failed to load chunks, skipping embedding process.")