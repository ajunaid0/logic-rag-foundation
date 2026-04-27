from rouge_score import rouge_scorer

def rouge_score_calculator(gold_answer, model_answer):
    if not gold_answer or not model_answer:
        return 0.0

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(str(gold_answer), str(model_answer))

    return scores['rougeL'].fmeasure


def gen_evaluator(gold_answer, model_answer):
    return rouge_score_calculator(gold_answer, model_answer)