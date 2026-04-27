import sys
import os
import pandas as pd
from tabulate import tabulate
import textwrap

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
BASE_DIR = '/content/drive/MyDrive/GEN AI Roadmap/logic-rag-foundation'
sys.path.append(os.path.join(BASE_DIR, 'src'))

# -----------------------------
# SETUP
# -----------------------------
from rag_setup import run_system_setup
run_system_setup()

from ingest import ingestion_pipeline
from embed import generate_embeddings
from generate import generate_answer
from retrieve import retrieve_chunks
from retrieval_evaluator import ret_evaluator
from generator_evaluator import gen_evaluator
from evaluation_logger import log_results


# -----------------------------
# DISPLAY HELPERS
# -----------------------------
def format_chunks_for_display(chunks, title=None):
    if not chunks:
        print(f"[WARN] No chunks found for {title}")
        return

    if title:
        print(f"\n>>> {title}")

    display_data = []

    for idx, c in enumerate(chunks):
        verdict = c.get('relevance', '-')

        display_data.append({
            "Chunk": idx + 1,
            "Source": textwrap.fill(
                c.get('source') or c.get('metadata', {}).get('source', 'N/A'),
                width=20
            ),
            "Score": round(c.get('score', 0), 4),
            "Verdict": verdict,
            "Text": textwrap.fill(c.get('text', 'N/A'), width=150)
        })

    df = pd.DataFrame(display_data)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))


# -----------------------------
# CORE PIPELINES
# -----------------------------
def run_retrieval(query, embedding_model, top_k, method):
    print(f"[PIPELINE] Retrieval | {method.upper()} | {embedding_model}")

    return retrieve_chunks(
        embedding_model,
        query,
        BASE_DIR,
        top_n=top_k,
        method=method
    )


def run_generation(query, chunks, model):
    print(f"[PIPELINE] Generation | {model}")

    if not chunks:
        print("[ERROR] No context found.")
        return None

    return generate_answer(query, chunks, model)


# -----------------------------
# GEN MODE
# -----------------------------
def run_gen_mode(query, embedding_model, generation_model, top_k, method):
    if not query:
        print("[ERROR] Query required for gen mode")
        return
    chunks = run_retrieval(query, embedding_model, top_k, method)
    answer = run_generation(query, chunks, generation_model)

    print("\n" + "-" * 80)
    print(f"QUERY: {query}")
    print(f"ANSWER: {textwrap.fill(str(answer), width=100)}")
    print("-" * 80)


# -----------------------------
# INGEST MODE
# -----------------------------
def run_ingest_mode(chunk_size, method):
    print("\n==============================")
    print("STARTING DATA INGESTION")
    print("==============================")

    ingestion_pipeline(base_path=BASE_DIR, chunk_size=chunk_size)

    generate_embeddings(
        embedding_model_name='mxbai-embed-large:335m',
        base_path=BASE_DIR,
        batch_size=100,
        method=method
    )

    print("[SUCCESS] Ingestion complete.\n")


# -----------------------------
# EVAL MODE
# -----------------------------
def run_eval_mode(queries, embedding_model, evaluation_model, generation_model, top_k, method):

    if not queries:
        print("[ERROR] No queries provided.")
        return

    print(f"\n[INFO] Running evaluation on {len(queries)} queries using {evaluation_model}")

    for item in queries:
        q = item.get('query')
        a = item.get('answer')
        qt = item.get('query_type')

        chunks = run_retrieval(q, embedding_model, top_k, method)

        precise_chunks, recall_ans = ret_evaluator(
            evaluation_model, q, a, chunks, recall_mode = "unanswerable" not in qt.lower()
        )

        total = sum(1 for c in precise_chunks if c.get('relevance') == 'RELEVANT')
        precision = total / len(precise_chunks) if precise_chunks else None

        model_answer = run_generation(q, precise_chunks, generation_model)

        rouge_score = (
            gen_evaluator(a, model_answer)
            if model_answer and qt != "Unanswerable questions"
            else "N/A"
        )

        item['ret_precision'] = precision
        item['ret_recall'] = 1 if recall_ans == "YES" else 0
        item['rouge_score'] = rouge_score

    return queries


# -----------------------------
# MAIN ROUTER
# -----------------------------
def main(query=None, queries=None, mode='gen',
         top_k=5, chunk_size=800, method='faiss'):

    embedding_model = 'mxbai-embed-large:335m'
    generation_model = 'llama3:8b-instruct-q4_K_M'
    evaluation_model = 'gemma3:12b'

    if mode == 'ingest':
        run_ingest_mode(chunk_size, method)

    elif mode == 'gen':
        run_gen_mode(query, embedding_model, generation_model, top_k, method)

    elif mode == 'eval':
        evaluated_queries= run_eval_mode(
            queries,
            embedding_model,
            evaluation_model,
            generation_model,
            top_k,
            method
        )
            
        log_results(evaluated_queries)

    else:
        print("[ERROR] Invalid mode")


# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":
    user_query = input("\nEnter question: ")

    main(
        query=user_query,
        mode="gen",
        top_k=5,
        chunk_size=800,
        method='faiss'
    )