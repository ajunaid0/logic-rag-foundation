import os
#from typing import TypeVarTuple

# --- PATHS ---
BASE_DIR = '/content/drive/MyDrive/GEN AI Roadmap/logic-rag-foundation'
DATA_DIR = os.path.join(BASE_DIR, 'data/logic_history_corpus')
VECTOR_STORE_DIR = os.path.join(BASE_DIR, 'vector_store/overlap')
LOG_DIR = os.path.join(BASE_DIR, 'logs/iteration 4')
GOLD_TEST_PATH = '/content/drive/MyDrive/GEN AI Roadmap/logic-rag-foundation/data/gold_test.xlsx'

# --- WORKFLOW SWITCHES ---
RUN_INGESTION = False
RUN_GENERATION = True
RUN_AUDIT = True

# --- MODELS ---
EMBEDDING_MODEL = 'mxbai-embed-large:335m'
GENERATION_MODEL = 'llama3:8b-instruct-q4_K_M'
RETRIEVAL_EVAL_MODEL = 'gemma3:12b'
JUDGE_MODEL = 'gemma4:e2b'
RERANK_MODEL_NAME = 'mixedbread-ai/mxbai-rerank-base-v2'
#RERANK_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2' 

# --- INGESTION SETTINGS ---
CHUNK_SIZE = 800
ITERATION = "4"  # Change this for each experimental run
CHUNK_MODE = "overlap" # Options: "clean" or "overlap"
EMBEDDING_BATCH_SIZE=100

# --- RETRIEVAL SETTINGS ---
RETRIEVAL_METHOD = "faiss"     # "base" or "faiss"
TOP_K_FINAL = 3      # How many chunks go to the LLM
TOP_N_INITIAL = 100  # How many chunks to retrieve before reranking
SIMILARITY_THRESHOLD=0.5
