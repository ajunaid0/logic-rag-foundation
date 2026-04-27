
	# Logic-RAG Foundation: 19th-Century Formal Logic Engine

	A Retrieval-Augmented Generation (RAG) pipeline for querying and evaluating 19th-century formal logic texts.  
	The system retrieves relevant passages from a curated corpus and generates grounded responses using a local LLM, with built-in evaluation for retrieval and generation quality.

	---

	## 📂 Project Structure

	- **data/**: Source corpus (Boole, Jevons, Mill, etc.)
	- **src/**: Core modules (ingestion, embedding, retrieval, generation, evaluation)
	- **vector_store/**: FAISS index, embeddings, and chunk metadata
	- **main.py**: Unified entry point (gen / eval / ingest modes)
	- **rag_setup.py**: Automated environment setup (dependencies + Ollama)
	- **evaluation_logger.py**: Logs evaluation outputs and metrics

	---

	## ⚙️ Pipeline Modes

	### 1. Ingestion Mode
	Builds the vector database from raw text corpus.

	```python
	main(mode="ingest")
	```

	---

	### 2. Generation Mode (RAG QA)

	Runs retrieval + LLM generation.

	```python
	main(
	    query="What is the Middle Term in syllogism?",
	    mode="gen"
	)
	```

	---

	### 3. Evaluation Mode (Batch Pipeline)

	Runs full evaluation over a dataset:

	- Retrieval evaluation (Precision / Recall)
	- Generation evaluation (ROUGE-L)
	- Handles unanswerable questions (ROUGE = N/A)

	```python
	main(
	    queries=queries_dict,
	    mode="eval"
	)
	```

	Each query must follow this format:

	```python
	{
	    "query": "...",
	    "answer": "...",
	    "query_type": "Single-passage factual / Cross-concept / Unanswerable questions"
	}
	```

	---

	## 🧠 Evaluation Metrics

	### Retrieval Metrics
	- Precision@K
	- Recall (YES / NO verdict from evaluator)

	### Generation Metrics
	- ROUGE-L F1 score
	- Automatically skipped for unanswerable questions (marked as `N/A`)

	---

	## 🛠️ Technical Stack

	- **LLM:** `llama3:8b-instruct-q4_K_M` (via Ollama)
	- **Evaluation Model:** `gemma3:12b`
	- **Embeddings:** `mxbai-embed-large:335m`
	- **Vector Store:** FAISS
	- **Similarity:** Cosine similarity
	- **Runtime:** Python 3.12 / Google Colab / Local Linux

	---

	## 📦 System Features

	- Modular RAG pipeline (ingest → retrieve → generate → evaluate)
	- Batch evaluation framework with structured logging
	- Supports unanswerable question detection
	- Automatic ROUGE evaluation skipping for invalid cases
	- Chunk-level retrieval inspection with relevance scoring

	---

	## 🚀 Usage

	### Install + Setup (automatic)
	```python
	from rag_setup import run_system_setup
	run_system_setup()
	```

	### Run a single query
	```python
	from main import main

	main(
	    query="What is the Law of Contradiction?",
	    mode="gen"
	)
	```

	### Run full evaluation
	```python
	main(
	    queries=queries,
	    mode="eval"
	)
	```

	---

	## 📊 Output Example (Eval Mode)

	- Precision@K per query
	- Recall verdict (YES / NO)
	- ROUGE-L score per answer
	- Aggregated summary logs
	- ROUGE breakdown by question type
	- Low-scoring query detection

	---

	## ⚠️ Limitations

	- Currently supports `.txt` corpus only
	- Requires Ollama runtime for LLM inference
	- Evaluation depends on deterministic LLM grading (not human-verified)

	---

	## 📌 Notes

	- Fully local RAG pipeline (no external APIs required)
	- Designed for structured reasoning over classical logic texts
	- Optimized for experimentation and evaluation research
  