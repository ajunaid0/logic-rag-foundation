from collections import defaultdict

def log_results(queries,top_k):

    print('\n\n')
    print('=== EVALUATION SUMMARY ===')
    print(f'Total Questions Evaluated: {len(queries)}')
    print("\n" + "-"*80)

    precision = []
    recall = []
    rouge_l = []
    rouge_id = []

    for idx, item in enumerate(queries):
        precision.append(item['ret_precision'])
        recall.append(item['ret_recall'])

        if item.get('rouge_score') != 'N/A':
            rouge_l.append(item['rouge_score'])
            rouge_id.append(idx)

    # -----------------------------
    # RETRIEVAL SUMMARY
    # -----------------------------
    print('\n--- RETRIEVAL EVALUATION SUMMARY ---')

    avg_precision = sum(precision) / len(precision) if precision else 0
    avg_recall = sum(recall) / len(recall) if recall else 0

    print(f'Average Precision@{top_k}: {avg_precision}')
    print(f'Average Recall@{top_k}: {avg_recall}')

    print("\n" + "-"*80)

    print(f'Questions with Perfect Precision (1.0): {precision.count(1.0)}')
    print(f'Questions with Zero Precision (0.0): {precision.count(0.0)}')
    print(f'Recall Failures (NO verdict): {recall.count(0)}')

    # -----------------------------
    # GENERATION SUMMARY
    # -----------------------------
    print('\n--- GENERATION EVALUATION SUMMARY ---')

    avg_rouge = sum(rouge_l) / len(rouge_l) if rouge_l else 0
    print(f'Average ROUGE-L F1: {avg_rouge}')

    if rouge_l:
        max_score = max(rouge_l)
        min_score = min(rouge_l)

        max_id = rouge_id[rouge_l.index(max_score)]
        min_id = rouge_id[rouge_l.index(min_score)]

        print(f'Highest ROUGE-L: {max_score} | Q{max_id+1}: {queries[max_id]["query"]}')
        print(f'Lowest ROUGE-L:  {min_score} | Q{min_id+1}: {queries[min_id]["query"]}')

    # -----------------------------
    # ROUGE BY TYPE
    # -----------------------------
    type_scores = defaultdict(list)

    for item in queries:
        qtype = item['query_type']
        score = item.get('rouge_score', 'N/A')

        if score != 'N/A':
            type_scores[qtype].append(score)

    print("\n" + "-"*80)
    print("ROUGE-L by Question Type:")
    print("-"*80)

    for qtype, scores in type_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {qtype}: {avg:.4f}  (avg across {len(scores)} questions)")
        else:
            print(f"  {qtype}: N/A (excluded)")

    # -----------------------------
    # LOW SCORING
    # -----------------------------
    threshold = 0.40
    print("\n" + "-"*80)
    print(f"Low Scoring Questions (ROUGE-L < {threshold}):")
    print("-"*80)

    for idx, item in enumerate(queries):
        score = item.get('rouge_score', 'N/A')

        if score != 'N/A' and score < threshold:
            print(f"  Q{idx+1}: {item['query'][:45]}... | {score:.4f} | {item['query_type']}")