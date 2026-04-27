from rag_setup import pull_ollama_model
import ollama


def precision_eval(eval_model, query, retrieved_chunks):
    instruction = '''
You are an expert logician specializing in evaluating data related to answering questions about 19th-century formal logic.
You are given a query and passage.
-- Answer ONLY with either 'RELEVANT' or 'NOT RELEVANT', nothing else.
-- Answer RELEVANT if the provided passage contains information that can be used to answer the question.
-- Answer NOT RELEVANT if the provided passage does not contribute in answering the query.
'''

    for chunk in retrieved_chunks:
        response = ollama.chat(
            model=eval_model,
            messages=[
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': f"###Question:\n{query}\n\nPassage:\n{chunk['text']}"}
            ]
        )

        raw = response['message']['content'].strip().upper()
        if 'NOT RELEVANT' in raw:
            verdict = 'NOT RELEVANT'
        elif 'RELEVANT' in raw:
            verdict = 'RELEVANT'
        else:
            verdict = 'UNCLEAR'

        # attach relevance directly to each chunk
        chunk['relevance'] = verdict

    return retrieved_chunks


def recall_eval(eval_model, query, answer, retrieved_chunks):
    instruction = '''
    You are an expert logician specializing in evaluating data related to answering questions about 19th-century formal logic.
    -- You will be given a question/answer pair along with a set of passages.
    -- Respond ONLY with 'YES' or 'NO'.
    -- If the passages together contain the information needed to support the given answer, respond YES.
    -- If key claims in the answer are not supported by or present in the passages, respond NO.
'''

    context_text = "\n\n".join(
        [f"Passage {idx+1}: {c['text']}" for idx, c in enumerate(retrieved_chunks)]
    )

    response = ollama.chat(
        model=eval_model,
        messages=[
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': f"###Question:\n{query}\n\nAnswer:\n{answer}\n\n###Passages:\n{context_text}"}
        ]
    )
    raw = response['message']['content'].strip().upper()
    if 'YES' in raw:
        verdict = 'YES'
    elif 'NO' in raw:
        verdict = 'NO'
    else:
        verdict = 'UNCLEAR'

    return verdict


def ret_evaluator(eval_model, query, answer, retrieved_chunks,recall_mode=True):
    pull_ollama_model(eval_model)

    precise_chunks = precision_eval(eval_model, query, retrieved_chunks)
    if recall_mode:
        recall_result = recall_eval(eval_model, query, answer, retrieved_chunks)
    else: 
      recall_result='UNANSWERABLE'
    return precise_chunks, recall_result