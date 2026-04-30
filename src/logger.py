import pandas as pd
from datetime import datetime
from collections import defaultdict
import os

def log_results(queries, top_k, base_path):
    # --- 1. DATA PREPARATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"Logic_RAG_Audit_K{top_k}_{timestamp}.xlsx"
    filepath = os.path.join(base_path, 'logs', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    precision = [q['ret_precision'] for q in queries]
    recall = [1 if q['ret_recall'] == 'YES' else 0 for q in queries if q['ret_recall'] != 'UNANSWERABLE']
    rouge_l = [q['rouge_score'] for q in queries if q['rouge_score'] != 'N/A']
    
    avg_precision = sum(precision) / len(precision) if precision else 0
    avg_recall = sum(recall) / len(recall) if recall else 0
    avg_rouge = sum(rouge_l) / len(rouge_l) if rouge_l else 0

    # --- 2. TERMINAL OUTPUT (Preserving Original Summary Metrics) ---
    print('\n' + '='*80)
    print('=== FINAL EVALUATION SUMMARY (TERMINAL LOG) ===')
    print(f'Total Questions: {len(queries)}')
    print(f'Average Precision@{top_k}: {avg_precision:.4f}')
    print(f'Average Recall (Answerable): {avg_recall:.4f}')
    print(f'Average ROUGE-L F1: {avg_rouge:.4f}')
    print('-'*80)

    # ROUGE by Question Type (Terminal)
    type_scores = defaultdict(list)
    for item in queries:
        if item.get('rouge_score') != 'N/A':
            type_scores[item['query_type']].append(item['rouge_score'])

    print("ROUGE-L by Question Type:")
    for qtype, scores in type_scores.items():
        avg = sum(scores) / len(scores)
        print(f"  {qtype.ljust(25)}: {avg:.4f} ({len(scores)} questions)")

    # --- 3. EXCEL EXPORT (Later Reference) ---
    detailed_df = pd.DataFrame([{
        "ID": i + 1,
        "Type": q.get('query_type'),
        "Question": q.get('query'),
        "Reference": q.get('answer'),
        "Model Answer": q.get('model_answer'),
        "Precision@K": q.get('ret_precision'),
        "Recall": q.get('ret_recall'),
        "ROUGE-L": q.get('rouge_score'),
        "Auditor Raw": q.get('raw_verdit'),
        "Auditor Verdict": q.get('faithfulness_verdict'),
        "Traceable Context": q.get('traceable_context')
    } for i, q in enumerate(queries)])

    summary_df = pd.DataFrame({
        "Metric": ["Avg Precision", "Avg Recall", "Avg ROUGE-L", "Total Count"],
        "Value": [avg_precision, avg_recall, avg_rouge, len(queries)]
    })

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='Detailed_Trace', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)

    print('\n' + '='*80)
    print(f"[SUCCESS] Deep Trace Log saved to: {filepath}")
    print('='*80 + '\n')