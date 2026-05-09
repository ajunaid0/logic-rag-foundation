import ollama

def response(query, chunks, generation_model):
    """
    Logic: Preserves the exact 4 rules and the specific rejection phrase 
    used in your 19th-century logic experiments.
    """

    instruction = '''
    You are an expert logician specializing in 19th-century formal logic.
    Answer questions using ONLY the provided context chunks.

    Rules (no exceptions):

    1. Never use training knowledge or modern logical developments. If the answer is not in the context, respond exactly: "Could not find any probable answer about the query from the source files."
    2. No inline citations. List all sources at the end under "Sources used:".
    3. For list questions, compile answers across ALL chunks before responding. Use clean formatting.
    4. Be concise. Paraphrase dense 19th-century language into clear modern prose. Do not block-quote unless a specific definition is required.
    5. For comparative questions: if the context defines two concepts separately without linking them, compare their definitions or functional roles (cause/effect, physical/mental, whole/part, generic/specific) — staying strictly within the provided context.

    Answer format:
    [Your answer in clear prose]

    Sources used:
    [Book title 1]
    [Book title 2]
    '''
  
    # EXACT ORIGINAL CONTEXT FORMATTING
    context_text = "\n\n".join([
        f"Source: {c['source'].replace('_', ' ').replace('.txt', '')}\nContent: {c['text']}"
        for c in chunks
    ])

    # EXACT ORIGINAL CHAT STRUCTURE
    response = ollama.chat(model=generation_model, messages=[
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': f'###Context:{context_text}\n\n###Question:{query}'}
    ])
    
    return response['message']['content']

def generate_answer(query, top_k_chunks, config):
    """
    Orchestrator for generation using the model defined in config.
    Note: Model pulling is now handled by the utils/models.py manager.
    """
    answer = response(
        query=query, 
        chunks=top_k_chunks, 
        generation_model=config.GENERATION_MODEL
    )

    return answer

"""
    # EXACT ORIGINAL INSTRUCTION
    instruction = '''
    You are an expert logician specializing in 19th-century formal logic. 
    Answer questions using ONLY the provided context chunks.

    Rules you must follow without exception:
    1. Never use training knowledge. If the answer is not in the context, respond exactly with: "Could not find any probable answer about the query from the source files."
    2. Cite every claim using the format [Source: filename].
    3. If a question asks for a list, compile answers across all provided chunks before responding.
    4. Be concise and direct. Do not add commentary beyond what the context supports.
    '''
"""