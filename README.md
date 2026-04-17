
# Logic-RAG Foundation: 19th-Century Formal Logic Engine

A Retrieval-Augmented Generation (RAG) pipeline for querying and synthesizing 19th-century formal logic texts. The system retrieves relevant passages from a curated corpus and generates grounded responses using a local LLM.

The pipeline includes an automated setup module that installs required dependencies and initializes components at runtime.

## 📂 Project Structure
- **`data/`**: Source corpus (currently supports `.txt` files only; e.g., Boole, Jevons, Mill)
- **`src/`**: Modules for ingestion, embedding, retrieval, and setup
- **`vector_store/`**: Storage for embeddings, chunk metadata, and FAISS indices
- **`main.py`**: Entry point for user queries
- **`rag_setup.py`**: Handles automatic environment setup and dependency installation

## 🛠️ Technical Stack
- **Model:** `llama3:8b-instruct-q4_K_M` (via Ollama)
- **Embeddings:** `nomic-embed-text`
- **Similarity:** Cosine similarity / FAISS
- **Environment:** Python 3.12 / Google Colab

## ⚙️ Requirements
- Ollama installed and running
- Required models available:
  - `llama3:8b-instruct-q4_K_M`
  - `nomic-embed-text`

*(Python dependencies are installed automatically via `rag_setup.py`)*

## 🚀 Usage
Run a query directly:

```python
from main import main

query = "What is the 'Middle Term,' and why is its distribution critical for a valid syllogism?"
main(query)
```

## ⚠️ Limitations
- Currently supports `.txt` files only (no PDF, DOCX, or HTML ingestion)

## 📌 Notes
- No manual dependency installation required
- Designed for small-to-medium curated corpora
- Optimized for interpretability and structured reasoning
