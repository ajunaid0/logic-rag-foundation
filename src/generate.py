from rag_setup import pull_ollama_model
import ollama

def response(query,chunks,generation_model):

    instruction= '''
    You are an expert logician specializing in 19th-century formal logic. 
    Answer questions using ONLY the provided context chunks.

    Rules you must follow without exception:
    1. Never use training knowledge. If the answer is not in the context, respond exactly with: "Could not find any probable answer about the query from the source files."
    2. Cite every claim using the format [Source: filename].
    3. If a question asks for a list, compile answers across all provided chunks before responding.
    4. Be concise and direct. Do not add commentary beyond what the context supports.
    '''
    context_text = "\n\n".join([
    f"Source: {c['source'].replace('_', ' ').replace('.txt', '')}\nContent: {c['text']}"
    for c in chunks])

    response = ollama.chat(model=generation_model, messages=[
        {'role':'system','content':instruction},
        {'role': 'user', 'content': f'###Context:{context_text}\n\n###Question:{query}'}])
    return response['message']['content']

def generate_answer(query,top_k_chunks,generation_model='llama3:8b-instruct-q4_K_M'):
    pull_ollama_model(generation_model)
    answer=response(query,top_k_chunks,generation_model)

    return answer