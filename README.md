# Logic-RAG Foundation: 19th-Century Formal Logic Engine

An automated Retrieval-Augmented Generation (RAG) pipeline designed to navigate and synthesize classical logic axioms. This tool allows researchers to query a curated corpus of formal logic using local Large Language Models (LLMs).

## 📂 Project Structure
- **`data/`**: Source corpus of .txt files (Boole, Jevons, Mill, etc.).
- **`src/`**: Modular logic for ingestion, embedding, and retrieval.
- **`vector_store/`**: Local storage for NumPy embeddings and chunk metadata.
- **`main.py`**: The entry point for user queries.

## 🛠️ Technical Stack
- **Model:** `llama3:8b-instruct-q4_K_M` (via Ollama)
- **Embeddings:** `nomic-embed-text` (via Ollama)
- **Similarity:** Cosine Similarity (Transitioning to FAISS)
- **Environment:** Python 3.12 / Google Colab

## 🚀 How to Run
1. Ensure Ollama is running and models are pulled.
2. Initialize the pipeline:
   ```python
   from main import main
   main("What are the three parts of a categorical syllogism?")

