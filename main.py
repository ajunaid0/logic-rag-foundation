import sys
import os

BASE_DIR = '/content/drive/MyDrive/GEN AI Roadmap/logic-rag-foundation'

sys.path.append(os.path.join(BASE_DIR, 'src'))

from rag_setup import util_files
util_files()

from ingest import ingestion_pipeline
from embed import generate_embeddings
from generate import generate_answer
from retrieve import retrieve_chunks


def ask_logic_question(query, method='faiss'):
    print('=== Retrieval ===')
    chunks = retrieve_chunks(query, BASE_DIR, top_n=15, method=method)
    print("✔ Completed")
    print("----------------------------------")
    print('=== Generation ===')
    answer = generate_answer(query, chunks)
    print('✔ Completed')
    print("----------------------------------")
    print(f"\nQ: {query}\nA: {answer}\n{'-'*30}")
    return answer,chunks


def main(query, FIRST_RUN=False, method='faiss'):

    if FIRST_RUN:
        print("Starting document ingestion and processing...")
        chunks = ingestion_pipeline(
            base_path=BASE_DIR,
            chunk_size=800,
            overlap=200
        )
        print("----------------------------------")
        print(f"Document processing complete with {len(chunks)} chunks ingested")

        print("Starting Chunk Embedding...")
        generate_embeddings(BASE_DIR, batch_size=100, method=method)
        print("----------------------------------")
        print("Embedding Process Completed")

    return ask_logic_question(query, method=method)


if __name__ == "__main__":
    query = input("\nEnter your question: ")
    print(main(query))