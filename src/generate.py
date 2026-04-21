from rag_setup import pull_ollama_model
import ollama

def response(query,chunks,model):
    print(f'[INFO] Generation Model: {model}')

    instruction= '''You are an expert logician specializing in 19th-century formal logic. Please use the provided context to answer the user's question.'''

    g_ins='''
    While answering make sure:
    1. You always provide the answer from the provided context.
    2. Do not use your internal knowledge to answer any question.
    3. If the answer is not in the context or there is no context, please respond with this exactly: "Could not find any probable answer about the query from the source files.", nothing else.
    4. Always cite the source if an answer is found from the context.
    5.If a question asks for a list or multiple rules, look across all provided chunks to compile the full set.
    '''
    context_text = "\n\n".join([
    f"Source: {c['source'].replace('_', ' ').replace('.txt', '')}\nContent: {c['text']}"
    for c in chunks])

    response = ollama.chat(model=model, messages=[
        {'role':'system','content':instruction},
        {'role': 'user', 'content': f'###Instructions:\n{g_ins}\n\n###Context:{context_text}\n\n###Question:{query}'}])
    return response['message']['content']

def generate_answer(query,top_k_chunks):
    generation_model='llama3:8b-instruct-q4_K_M'
    pull_ollama_model(generation_model)
    answer=response(query,top_k_chunks,generation_model)

    return answer