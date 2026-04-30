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
from logger import log_results


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
    format_chunks_for_display(chunks)
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
def run_eval_mode(queries, embedding_model, evaluation_model, generation_model, judge_model, top_k, method):
    if not queries:
        print("[ERROR] No queries provided.")
        return

    print(f"\n{'='*30}\nSTARTING EVALUATION: {len(queries)} Queries\n{'='*30}")

    for idx, item in enumerate(queries):
        q = item.get('query')
        a = item.get('answer')
        qt = item.get('query_type', 'Unknown')
        is_unanswerable = "unanswerable" in qt.lower()

        # --- STAGE 1: RETRIEVAL ---
        print(f"\n[Q{idx+1}] Processing: {q[:70]}...")
        chunks = run_retrieval(q, embedding_model, top_k, method)
        
        # --- STAGE 2: PRECISION & RECALL ---
        print(f"  > Evaluating Retrieval Quality...")
        precise_chunks, recall_ans = ret_evaluator(
            evaluation_model, q, a, chunks, 
            recall_mode=(not is_unanswerable)
        )
        
        # Print the grid to terminal so you can see it NOW
        format_chunks_for_display(precise_chunks, title=f"Retrieval Trace for Q{idx+1}")
        print(f"  > Recall Verdict: {recall_ans}")

        # --- STAGE 3: GENERATION ---
        print(f"  > Generating Answer...")
        model_answer = run_generation(q, precise_chunks, generation_model)
        print(f"  > Model Answer: {str(model_answer)[:100]}...")

        # --- STAGE 4: FAITHFULNESS & ROUGE ---
        print(f"  > Running Final Audit...")
        faithfulness_verdict = "N/A"
        rouge_score = "N/A"
        raw_judge_output=' '
        
        if model_answer and not is_unanswerable:
            rouge_score, raw_judge_output, faithfulness_verdict = gen_evaluator(q, a, model_answer, precise_chunks, judge_model)
        elif is_unanswerable:
            faithfulness_verdict = "CORRECTLY REJECTED" if "could not find" in model_answer.lower() else "FAILED REJECTION"

        # --- STAGE 5: PRESERVE FOR EXCEL ---
        context_blocks = []
        for c_idx, c in enumerate(precise_chunks):
            block = (f"ID: {c_idx+1} | Score: {round(c.get('score', 0), 4)} | "
                     f"Source: {c.get('source', 'N/A')} | Verdict: {c.get('relevance')}\n"
                     f"Content: {c.get('text')}")
            context_blocks.append(block)
        
        relevant_count = sum(1 for c in precise_chunks if c.get('relevance') == 'RELEVANT')
        
        # Store in item dict for the logger to pick up
        item.update({
            'model_answer': model_answer,
            'ret_precision': relevant_count / len(precise_chunks) if precise_chunks else 0,
            'ret_recall': recall_ans,
            'rouge_score': rouge_score,
            'raw_verdit': raw_judge_output,
            'faithfulness_verdict': faithfulness_verdict,
            'traceable_context': "\n\n" + "="*50 + "\n\n".join(context_blocks)
        })

        print(f"[SUCCESS] Q{idx+1} complete. Faithfulness: {str(raw_judge_output)}")
        print(f"Final Verdict: {str(faithfulness_verdict)}")

    return queries
# -----------------------------
# MAIN ROUTER
# -----------------------------
def main(query=None, queries=None, mode='gen',
         top_k=5, chunk_size=800, method='faiss'):

    embedding_model = 'mxbai-embed-large:335m'
    generation_model = 'llama3:8b-instruct-q4_K_M'
    evaluation_model = 'gemma3:12b'
    #judge_model='vicgalle/prometheus-7b-v2.0:latest'
    judge_model='gemma4:e2b'

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
            judge_model,
            top_k,
            method
        )
            
        log_results(evaluated_queries,top_k,BASE_DIR)

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