import pandas as pd
import os
from datetime import datetime

def log_experiment_results(queries, config):
    """
    Saves a dual-sheet Excel file with your specific naming convention:
    Iteration_X_Logic_RAG_Audit_K_mode_topK_timestamp.xlsx
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Construct filename based on your requested pattern
    filename = (
        f"Iteration_{config.ITERATION}_"
        f"Logic_RAG_Audit_"
        f"{config.CHUNK_MODE}_"
        f"K{config.TOP_K_FINAL}_"
        f"{timestamp}.xlsx"
    )
    
    filepath = os.path.join(config.LOG_DIR, filename)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # --- 1. Summary Metrics Calculation ---
    precision_vals = [q.get('ret_precision', 0) for q in queries]
    
    # Handle "UNANSWERABLE" case for Recall
    recall_vals = [
        1 if q.get('ret_recall') == 'YES' else 0 
        for q in queries if q.get('ret_recall') != 'UNANSWERABLE'
    ]
    
    # Faithfulness (using your 1, 0, -1 logic)
    faithfulness_vals = [
        q.get('faithfulness_verdict', 0) 
        for q in queries if q.get('faithfulness_verdict') != -1
    ]
    
    rouge_vals = [q.get('rouge_score', 0) for q in queries]

    summary_df = pd.DataFrame({
        "Metric": [
            "Iteration",
            "Chunking Mode",
            "Avg Precision@K", 
            "Avg Recall", 
            "Avg Faithfulness (1=Strict)", 
            "Avg ROUGE-L", 
            "Total Queries"
        ],
        "Value": [
            config.ITERATION,
            config.CHUNK_MODE,
            sum(precision_vals)/len(precision_vals) if precision_vals else 0,
            sum(recall_vals)/len(recall_vals) if recall_vals else 0,
            sum(faithfulness_vals)/len(faithfulness_vals) if faithfulness_vals else 0,
            sum(rouge_vals)/len(rouge_vals) if rouge_vals else 0,
            len(queries)
        ]
    })

    # --- 2. Detailed Trace Sheet ---
    detailed_data = []
    for i, q in enumerate(queries):
        detailed_data.append({
            "ID": i + 1,
            "Question": q.get('query'),
            "Reference": q.get('answer'),
            "Model Answer": q.get('model_answer'),
            "Recall Verdict": q.get('ret_recall'),
            "ROUGE-L": q.get('rouge_score'),
            "Faithfulness Verdict": q.get('faithfulness_verdict'),
            "Audit Report (Raw)": q.get('faithfulness_report'),
            "Sources Used": q.get('traceable_context') # All the chunks
        })
    
    detailed_df = pd.DataFrame(detailed_data)

    # --- 3. Final Write ---
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
        detailed_df.to_excel(writer, sheet_name='Detailed_Trace', index=False)

    print(f"\n[LOG] Audit complete. File generated: {filename}")
    return filepath