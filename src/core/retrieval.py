import os
import json
import numpy as np
import ollama
import faiss
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model for complexity check
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# CONCEPT DECOMPOSITION LOGIC
# -----------------------------

def is_cross_concept(query):
    """
    Systematic Complexity Auditor with Stem-Aware Matching.
    """
    query_lower = query.lower()
    breakdown = {"thinker_count": 0, "concept_count": 0, "bridge_score": 0, "artifact_score": 0, "total_score": 0}

    thinkers = ["mill", "jevons", "read", "atkinson", "james", "hamilton", "bain", 
                "venn", "keynes", "beneke", "halleck", "hyslop", "brooks", "aristotle", 
                "hobbes", "comte", "lambert", "herschel", "newton", "bacon", "de morgan", 
                "leibnitz", "locke", "reid", "descartes"]

    logic_concepts = ["syllogis", "induct", "deduct", "infer", "proposit", "premise", 
                      "conclu", "fallac", "analog", "predicat", "quantifier", "valid", 
                      "sound", "enthymem", "axiom", "dictum", "substitut", "distribut", 
                      "categor", "hypothet"]

    bridge_roots = ["vs", "versus", "compar", "contrast", "differ", "relat", "impact", 
                    "influenc", "illustrat", "modific"]

    found_thinkers = set([t for t in thinkers if t in query_lower])
    found_concepts = set([c for c in logic_concepts if c in query_lower])
    
    breakdown["thinker_count"] = len(found_thinkers)
    breakdown["concept_count"] = len(found_concepts)
    if any(b in query_lower for b in bridge_roots): breakdown["bridge_score"] = 2
    if re.search(r'["\'].*?["\']', query_lower): breakdown["artifact_score"] = 1

    breakdown["total_score"] = sum([breakdown["thinker_count"], breakdown["concept_count"], 
                                    breakdown["bridge_score"], breakdown["artifact_score"]])

    # AUDIT LOG

    print(f"\n--- STEM-AWARE COMPLEXITY AUDIT ---")
    print(f"Query: {query}")
    print(f"Detected Roots: {list(found_thinkers) + list(found_concepts)}")
    print(f"Score Breakdown: {breakdown}")
    print(f"Final Decision: {'COMPLEX' if breakdown['total_score'] >= 4 else 'SIMPLE'}")
    print(f"-----------------------------------\n")
    return breakdown["total_score"] >= 4

def decompose_query(query, config, debug=True):
    """
    Decomposes complex query into exactly 3 independent search units via LLM.
    """
    response_schema = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["queries"]
    }

    system_prompt = """

    You are a Logic Research Assistant. Your task is to decompose complex queries into exactly 3 independent search units.

    STRICT RULES:

    1. GROUNDING: Break the query into units that 100% portray a part of the original query.

    2. NO FILLER: Strip out lead-ins like "What does the text say", "According to", or "Can you explain".

    3. INDEPENDENCE: Each unit MUST be self-contained. Replace pronouns (it, they, his, this) with the actual subject.

    4. NO HALLUCINATION: Do not add authors or concepts not mentioned in the user query.

    """

    user_prompt = f"""

    ### EXAMPLES OF CORRECT DECOMPOSITION:

    Example 1 (Multi-Concept):

    Query: "How does the concept of 'Syllogism' relate to 'Inductive Logic' in the works of Richard Whately?"

    Decomposition: {{

        "queries": [

            "Richard Whately's definition of Syllogism",

            "Inductive Logic in the works of Richard Whately",

            "Relationship between Syllogism and Inductive Logic"

        ]

    }}

    Example 2 (Procedural/List):

    Query: "Identify the five distinct steps of a logical proof as outlined in the provided text."

    Decomposition: {{

        "queries": [

            "Five distinct steps of a logical proof",

            "Definition of a logical proof",

            "Outline of steps for logical proofs in the text"

        ]

    }}

    Example 3 (Abstract/Thematic):

    Query: "Is there evidence that the principle of 'Excluded Middle' is rejected by modern intuitionists in these archives?"

    Decomposition: {{

        "queries": [

            "Principle of Excluded Middle",

            "Modern intuitionists' views on logic",

            "Rejection of the Principle of Excluded Middle"

        ]

    }}

    ### TASK:
    Decompose this complex query: {query}

    """

    try:
        response = ollama.chat(
            model=config.GENERATION_MODEL,
            format=response_schema,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        parsed = json.loads(response['message']['content'])
        queries = parsed.get("queries", [query])
        final_queries = queries[:3]

        if debug:
            print("-" * 30)
            print(f"[DEBUG MODE - DECOMPOSITION]")
            print(f"Original: {query}")
            for i, q in enumerate(final_queries, 1):
                print(f"  Unit {i}: {q}")

            print("-" * 30)
        return final_queries

    except Exception as e:
        if debug:
            print(f"[DEBUG ERROR] Decomposition failed: {e}")
        return [query]

# -----------------------------
# RETRIEVAL CORES
# -----------------------------

def base_retrieval(query_embedding, embeddings, chunks, top_n):
    """Simple Cosine Similarity search using sklearn."""
    query_vec = np.array(query_embedding).reshape(1, -1)
    # Cosine similarity expects (n_samples, n_features)
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "source": chunk.get("metadata", {}).get("source")
        })
    return results

def faiss_retrieval(query_embedding, index, chunks, top_n):
    """FAISS-based search (assumes Inner Product on Normalized vectors)."""
    if query_embedding is None: return []
    query_vec = np.array(query_embedding).astype("float32").reshape(1, -1)
    if np.linalg.norm(query_vec) == 0: return []

    faiss.normalize_L2(query_vec) 
    distances, indices = index.search(query_vec, top_n)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx != -1 and idx < len(chunks):
            chunk = chunks[idx]
            results.append({
                "score": float(distances[0][i]),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text"),
                "source": chunk.get("metadata", {}).get("source")
            })
    return results

# -----------------------------
# RE-RANKER & HELPERS
# -----------------------------

def reranker(query, chunks, rerank_model):

    if not chunks: return []
    sentence_pairs = [(query, f"Source: {c['source']}\nContent: {c['text']}") for c in chunks]
    scores = rerank_model.predict(sentence_pairs)
    for chunk, score in zip(chunks, scores):
        chunk['relevance_score'] = float(score)
    return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)


def get_query_embedding(model_name, query):
    response = ollama.embeddings(model=model_name, prompt=query)
    return response["embedding"]

def load_chunks(file_path):
    try:
        with open(file_path, "r") as f: return json.load(f)
    except Exception: return []

def execute_search(query, config, top_n):
    """Orchestrates specific retrieval method based on config."""
    vs_dir = config.VECTOR_STORE_DIR
    method = config.RETRIEVAL_METHOD
    q_emb = get_query_embedding(config.EMBEDDING_MODEL, query)

    if method == "base":
        chunks = load_chunks(os.path.join(vs_dir, "base_metadata.json"))
        embeddings = np.load(os.path.join(vs_dir, "embeddings.npy"))
        return base_retrieval(q_emb, embeddings, chunks, top_n)

    elif method == "faiss":
        chunks = load_chunks(os.path.join(vs_dir, "faiss_metadata.json"))
        index = faiss.read_index(os.path.join(vs_dir, "faiss_index.faiss"))
        return faiss_retrieval(q_emb, index, chunks, top_n)

    return []

# -----------------------------
# MAIN ORCHESTRATOR
# -----------------------------

def retrieve_chunks(query, config, rerank_model):
    """
    ITERATION: Decomposition -> Local Rerank per Sub-query -> Round Robin Merge.
    Applies strict similarity thresholding before any reranking occurs.
    """
    final_k = config.TOP_K_FINAL
    sim_threshold = config.SIMILARITY_THRESHOLD

    if is_cross_concept(query):
        sub_queries = decompose_query(query, config)
        per_query_reranked_lists = []

        for sub_q in sub_queries:
            # Search broadly to allow thresholding to act as a filter
            raw_results = execute_search(sub_q, config, top_n=config.TOP_N_INITIAL)
            
            # 1. Similarity Threshold Filter
            clean_results = [
                c for c in raw_results 
                if c.get('score', 0) > sim_threshold and len(c['text'].split()) > 10
            ]
            
            # 2. Local Rerank (Targeted for the sub-concept)
            if clean_results:
                ranked_sub = reranker(sub_q, clean_results, rerank_model)
                per_query_reranked_lists.append(ranked_sub)

        # 3. Round Robin Interleaving
        final_retrieved = []
        unique_cids = set()
        
        if not per_query_reranked_lists: return []
        max_depth = max(len(lst) for lst in per_query_reranked_lists)

        for depth in range(max_depth):
            for sub_list in per_query_reranked_lists:
                if depth < len(sub_list):
                    chunk = sub_list[depth]
                    if chunk['chunk_id'] not in unique_cids:
                        final_retrieved.append(chunk)
                        unique_cids.add(chunk['chunk_id'])
                
                if len(final_retrieved) >= final_k:
                    return final_retrieved
        return final_retrieved

    else:
        # SIMPLE PATH
        raw_results = execute_search(query, config, top_n=config.TOP_N_INITIAL)
        clean_results = [
            c for c in raw_results 
            if c.get('score', 0) > sim_threshold and len(c['text'].split()) > 10
        ]
        if not clean_results: return []
        
        return reranker(query, clean_results, rerank_model)[:final_k]