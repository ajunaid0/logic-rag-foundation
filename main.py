import sys
import os

# Adding src to path so imports work locally
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ollama_setup import ollama_utils
from ingest import ingestion_pipeline
from embed import generate_embeddings
from generate import generate_answer
from retrieve import get_top_k_chunks

def ask_logic_question(query):
    chunks = get_top_k_chunks(query,BASE_DIR,top_n=15)
    answer = generate_answer(query, chunks)
    return answer

if __name__ == "__main__":
    
    BASE_DIR = '/content/drive/MyDrive/GEN AI Roadmap/logic-rag-foundation'
    # Change this to True if you are running for the first time
    FIRST_RUN = False 
    
    ollama_utils()
    if FIRST_RUN:
        ingestion_pipeline(
          base_path=BASE_DIR, 
          chunk_size=800, 
          overlap=200
          )
        generate_embeddings(BASE_DIR,
        batch_size=500)
        
    user_query = input("Enter your logic question: ")
    ask_logic_question(user_query)
