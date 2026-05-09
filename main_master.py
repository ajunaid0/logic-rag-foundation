import pandas as pd
import src.config.settings as config
#from src.utils.setup import run_system_setup
#run_system_setup()
from src.core.ingestion import ingestion_pipeline
from src.core.embedding import generate_embeddings
from src.core.retrieval import retrieve_chunks  
from src.core.generation import generate_answer
from src.evaluation.auditor import RAGAuditor
from src.utils.logger import log_experiment_results
from src.utils.models import init_all_models
from src.utils.display import chunk_display

def main():
    # 0. Initialize Models (Reranker, LLMs)
    # This should return the rerank_model object needed by your retrieve_chunks

    models = init_all_models(config)
    rerank_model = models.get('reranker')

    # 1. Ingestion Phase
    if config.RUN_INGESTION:
        ingestion_pipeline(config)
        generate_embeddings(config)

    # 2. Data Loading (Your Excel Pattern)
    df = pd.read_excel(config.GOLD_TEST_PATH)
    test_queries = []
    for _, item in df.iterrows():
        test_queries.append({
            'query_type': item['question_type'],
            'query': item['question'],
            'answer': item.get('reference_answer', '') 
        })

    auditor = RAGAuditor(config)
    final_results = []

    # 3. Execution Phase
    for item in test_queries:
        print(f"\nAuditing Query: {item['query']}")

        # --- RETRIEVAL ---
        # Uses your logic: Initial(100) -> Reranker -> Final(5)
        top_chunks = retrieve_chunks(
            query=item['query'],
            config=config,
            rerank_model=rerank_model
        )

        # --- GENERATION ---
        model_ans = ""
        if config.RUN_GENERATION:
            model_ans = generate_answer(
                query=item['query'],
                top_k_chunks=top_chunks,
                config=config
            )

        # --- AUDIT ---
        if config.RUN_AUDIT:
            # Note: We pass top_chunks (the re-ranked final list) to the auditor
            is_unanswerable = "unanswerable" in item['query_type'].lower()
            precise_chunks, recall_v = auditor.evaluate_retrieval(
                item['query'], item['answer'], top_chunks)
            chunk_display(precise_chunks)
            relevant_count = sum(1 for c in precise_chunks if c.get('relevance') == 'RELEVANT')   
            ret_pre= relevant_count / len(precise_chunks) if precise_chunks else 0 
            print(f'Avg Chunk Presision: {relevant_count}/{len(precise_chunks)} = {ret_pre}')
            print(f'Retrieval Recall: {recall_v}')

            if model_ans and not is_unanswerable:
                rouge, report, faith_v = auditor.evaluate_generation(
                item['query'], item['answer'], model_ans, precise_chunks
            )
            elif is_unanswerable:
                rouge = 0
                report="CORRECTLY REJECTED"
                faith_v = 1 if "could not find" in model_ans.lower() else 0

            print(f'Model Answer: {model_ans}')
            print(f'ROUGE-L Score: {rouge}')
            print(f'Faithfullness Report: {report}')
            print(f'Faithfulness Verdict: {faith_v}')
            context_blocks = []
            for c_idx, c in enumerate(precise_chunks):
                block = (f"ID: {c_idx+1} | Similarity Score: {round(c.get('score', 0), 4)} | "
                f"Re-Ranker Score: {round(c.get('relevance_score', 0), 4)} | "
                f"Source: {c.get('source', 'N/A')} | Verdict: {c.get('relevance')}\n"
                f"Content: {c.get('text')}")
            context_blocks.append(block)
            item.update({
                "model_answer": model_ans,
                "ret_precision": ret_pre,
                "ret_recall": recall_v,
                "rouge_score": rouge,
                "faithfulness_verdict": faith_v,
                "faithfulness_report": report,
                "traceable_context": "\n\n" + "="*50 + "\n\n".join(context_blocks)
            })
        else:
            item.update({"model_answer": model_ans})
            
        final_results.append(item)

    # 4. Final Logging
    log_experiment_results(final_results, config)
    print(f"\n[SUCCESS] Experiment iteration {config.ITERATION} complete.")

if __name__ == "__main__":
    main()