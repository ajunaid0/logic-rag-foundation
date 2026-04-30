from rouge_score import rouge_scorer
from rag_setup import pull_ollama_model
import ollama

def rouge_score_calculator(gold_answer, model_answer):
    if not gold_answer or not model_answer:
        return 0.0

    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(str(gold_answer), str(model_answer))

    return scores['rougeL'].fmeasure

def faithfulness_evaluator(query, chunks, model_answer, judge_model):
    """
    Evaluates the faithfulness of a model's answer against retrieved context chunks
    using 19th-century logic constraints.
    """
    
    system_prompt = '''
    You are a strict Groundedness and Faithfulness Auditor specializing in 19th-century formal logic. 
    Your goal is to ensure every factual claim made by an AI assistant is derived exclusively from the provided text segments.

    The Dataset Context:
    The sources contain archaic, technical definitions from authors like Mill, Jevons, and Read. 
    You must be vigilant against "Modern Logic Leaks"—if an answer uses modern definitions not found 
    in the segments (even if factually correct in 2026), it is UNFAITHFUL.

    Evaluation Protocol:
    1. Claim Extraction: Break the "Model Answer" down into a list of individual factual claims (Atomic Facts).
    2. Evidence Mapping: For each claim, identify the specific Chunk ID that supports it.
    3. Gap Analysis: Identify claims relying on internal training data rather than provided chunks. 
       Pay close attention to author-specific views (e.g., James vs. Read).

    Scoring Rubric:
    - Score 1 (Faithful): Every single claim is directly supported by the retrieved chunks.
    - Score 0 (Unfaithful): One or more claims are missing evidence, contradict chunks, or use modern logical terminology.

    Output Format:
    Claims Checklist: [Claim 1: Supported by Chunk X / Claim 2: No Support]
    Final Verdict: [Score 0 or 1]
    Reasoning: [Briefly explain any hallucinations or "Modern Leaks" found.]
    '''

    # Formatting the chunks into a readable string for the prompt
    context_text = "\n\n".join([
        f"Chunk ID: {idx+1}\nSource: {c['source']}\nContent: {c['text']}"
        for idx, c in enumerate(chunks)
    ])

    user_content = f"""
    ### Question:
    {query}

    ### Retrieved Chunks:
    {context_text}

    ### Model Answer to Evaluate:
    {model_answer}
    """

    response = ollama.chat(model=judge_model, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_content}
    ])

    raw = response['message']['content']
    if 'Final Verdict: 1' in raw or 'Score 1' in raw.upper():
        verdict = 1
    elif 'Final Verdict: 0' in raw or 'Score 0' in raw.upper():
        verdict = 0
    else:
        verdict = -1  # unclear, flag for manual review
    return raw, verdict
def gen_evaluator(query, gold_answer, model_answer,chunks,judge_model):
    rouge_score= rouge_score_calculator(gold_answer, model_answer)
    pull_ollama_model(judge_model)
    raw, faithfullness_verdict= faithfulness_evaluator(query, chunks, model_answer, judge_model)
    return rouge_score, raw, faithfullness_verdict