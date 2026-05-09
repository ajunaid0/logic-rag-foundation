import os
import ollama
from sentence_transformers import CrossEncoder
from transformers import AutoModel


def pull_ollama_model(model_name):
    try:
        print(f"[SYSTEM] Checking Ollama model: {model_name}")
        ollama.pull(model_name)
    except Exception as e:
        print(f"[ERROR] Could not pull {model_name}: {e}")


def init_all_models(config):

    llm_models = [
        config.EMBEDDING_MODEL,
        config.GENERATION_MODEL,
        config.RETRIEVAL_EVAL_MODEL,
        config.JUDGE_MODEL
    ]

    for m in llm_models:
        pull_ollama_model(m)

    print(f"[SYSTEM] Loading Reranker into memory: {config.RERANK_MODEL_NAME}")
    '''
    try:
        reranker = AutoModel.from_pretrained(
        config.RERANK_MODEL_NAME,
        dtype="auto",
        trust_remote_code=True,
        )
        reranker.eval()

    except Exception as e:
        print(f"[ERROR] Reranker load failed: {e}")
        reranker = None
    '''
    try:
        reranker = CrossEncoder(config.RERANK_MODEL_NAME)
    except Exception as e:
        print(f"[ERROR] Reranker load failed: {e}")
        reranker = None
    
    return {"reranker": reranker}