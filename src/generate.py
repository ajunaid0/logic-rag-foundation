from ollama_setup import pull_ollama_model

def response(query,chunks,model):
    import ollama
    instruction= '''You are an expert logician specializing in 19th-century formal logic. Please use the provided context to answer the user's question.'''

    g_ins='''
    While answering make sure:
    1. You always provide the answer from the provided context.
    2. Do not use your internal knowledge to answer any question.
    3. Where you are unsure of the answer, please respond with "I don't know" instead of providing vague answer.
    4. If the answer is not in the context, please respond with "Insufficient Information on this question" instead of a vague answer.
    5. Always cite the source.
    If a question asks for a list or multiple rules, look across all provided chunks to compile the full set.
    '''
    context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in chunks])

    response = ollama.chat(model=model, messages=[
        {'role':'system','content':instruction},
        {'role': 'user', 'content': f'###Instructions:\n{g_ins}\n\n###Context:{context_text}\n\n###Question:{query}'}])
    print(f"Q: {query}\nA: {response['message']['content']}\n{'-'*30}")

    return response['message']['content']

def generate_answer(query,top_k_chunks):
    generation_model='llama3:8b-instruct-q4_K_M'
    pull_ollama_model(generation_model)
    answer=response(query,top_k_chunks,generation_model)

    return answer