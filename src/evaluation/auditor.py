import ollama
from rouge_score import rouge_scorer
from src.utils.models import pull_ollama_model

class RAGAuditor:
    def __init__(self, config):
        self.config = config
        self.client = ollama # Using the standard ollama interface from your OG code
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )

    # ==========================================
    # GENERATION EVALUATORS (From your OG File)
    # ==========================================

    def evaluate_generation(self, query, gold_answer, model_answer, chunks):
        """
        Equivalent to your OG gen_evaluator. 
        Returns: rouge_l_fmeasure, raw_faithfulness_report, faithfulness_verdict
        """
        # 1. ROUGE Calculation
        scores = self.rouge_scorer.score(str(gold_answer), str(model_answer))
        rouge_l = scores['rougeL'].fmeasure

        # 2. Faithfulness Audit (Strict 19th-Century Logic Protocol)
        system_prompt = '''
        You are a Groundedness and Faithfulness Auditor specializing in 19th-century formal logic.
        Your sole task: verify that every claim in the Model Answer is strictly rooted in the provided context chunks.

        Key threat to watch for — "Modern Logic Leaks": definitions, concepts, or relationships drawn from training knowledge rather than the retrieved text.

        Evaluation Protocol:

        1. Claim Extraction: Break the Model Answer into individual factual claims.
        2. Evidence Mapping: For each claim, identify the Chunk ID that supports it.
        3. Synthesis Validation: For comparative or bridging claims where the context defines concepts separately without explicitly linking them:
        - FAITHFUL: The model derives a relationship from definitions or functional roles present in the text (e.g., cause/effect, physical/mental, whole/part, generic/specific).
        - UNFAITHFUL: The model asserts a relationship, historical influence, or evaluative connection not derivable from the retrieved definitions alone.

        Scoring:
        - Faithful: Every claim is directly supported or is a grounded synthesis based solely on retrieved definitions.
        - Unfaithful: Any claim relies on training knowledge, modern logical terminology, or commentary absent from the source.

        Output Format:
        Final Verdict: Faithful or Unfaithful
        Claim Extraction and Evidence Mapping: [Claim 1: Supported by Chunk X / Claim 2: Grounded Synthesis via Chunks X & Y]
        Reasoning: [Briefly explain any Modern Logic Leaks, hallucinations, or why a synthesis was or was not grounded.]
        '''

        
        context_text = "\n\n".join([
            f"Chunk ID: {idx+1}\nSource: {c['source']}\nContent: {c['text']}"
            for idx, c in enumerate(chunks)
        ])

        user_content = f"### Question:\n{query}\n\n### Retrieved Chunks:\n{context_text}\n\n### Model Answer:\n{model_answer}"

        response = self.client.chat(model=self.config.JUDGE_MODEL, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ])

        raw_report = response['message']['content']
        report_clean = raw_report.lower()
        
        # Verdict Logic from your OG code
        if 'final verdict: faithful' in report_clean:
            verdict = 1
        elif 'final verdict: unfaithful' in report_clean:
            verdict = 0
        else:
        # This handles cases where the LLM might just say "Faithful" 
        # without the full "Final Verdict:" prefix.
            if 'faithful' in report_clean and 'unfaithful' not in report_clean:
                 verdict = 1
            elif 'unfaithful' in report_clean:
                verdict = 0
            else:
                verdict = -1
            
        return rouge_l, raw_report, verdict

    # ==========================================
    # RETRIEVAL EVALUATORS (From your OG File)
    # ==========================================

    def evaluate_retrieval(self, query, gold_answer, retrieved_chunks):
        """
        Equivalent to your OG ret_evaluator.
        Returns: precise_chunks (with metadata), recall_verdict
        """
        # 1. Precision Evaluation
        precision_instruction = '''
        You are an expert logician specializing in evaluating data related to 19th-century formal logic.
        Answer ONLY with 'RELEVANT' or 'NOT RELEVANT'.
        '''
        
        for chunk in retrieved_chunks:
            # We pass the full context_text in each loop as per your OG logic
            chunk_content = f"Source: {chunk['source']}\nContent: {chunk['text']}"
            
            res = self.client.chat(model=self.config.RETRIEVAL_EVAL_MODEL, messages=[
                {'role': 'system', 'content': precision_instruction},
                {'role': 'user', 'content': f"###Question:\n{query}\n\nPassage:\n{chunk_content}"}
            ])
            
            raw_p = res['message']['content'].strip().upper()
            chunk['relevance'] = 'RELEVANT' if 'RELEVANT' in raw_p else 'NOT RELEVANT'

        # 2. Recall Evaluation
        recall_instruction = '''
        Respond ONLY with "YES", "NO", or "UNANSWERABLE".

        - YES: The provided passages, taken together, contain enough information to support the answer — including cases where two concepts are defined separately but their relationship can be grounded via functional roles (cause/effect, physical/mental, whole/part, generic/specific).
        - NO: The answer makes claims not derivable from the passages, even through grounded synthesis.
        - UNANSWERABLE: The question falls outside the scope of the passages entirely (e.g., asks about modern developments absent from the source).
        '''
        context_block = "\n\n".join([f"Passage {i+1}: {c['text']}" for i, c in enumerate(retrieved_chunks)])
            
        res_r = self.client.chat(model=self.config.RETRIEVAL_EVAL_MODEL, messages=[
            {'role': 'system', 'content': recall_instruction},
            {'role': 'user', 'content': f"###Question:\n{query}\n\nAnswer:\n{gold_answer}\n\n###Passages:\n{context_block}"}
            ])
            
        raw_r = res_r['message']['content'].strip().upper()
        recall_verdict = 'UNANSWERABLE' if 'UNANSWERABLE' in raw_r else ('YES' if 'YES' in raw_r else 'NO')

        return retrieved_chunks, recall_verdict