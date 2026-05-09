import os
import re
import json
import statistics
import spacy

# ==========================================================
# SPACY MODEL
#==========================================================

nlp = spacy.load("en_core_web_sm")

# ==========================================================
# DOCUMENT LOADER
# ==========================================================

def load_documents(raw_data_path: str) -> list:
    documents = []

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Path not found: {raw_data_path}")
        return []

    for root, _, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)

                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        documents.append({
                            "filename": file,
                            "path": path,
                            "content": f.read()
                        })

                except Exception as e:
                    print(f"[ERROR] Skip File: {file} | {e}")

    print(f"[INFO] Loaded {len(documents)} documents")
    return documents

# ==========================================================
# GUTENBERG CLEANER
# ==========================================================

def clean_gutenberg_ocr(text):

    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    start = re.search(r'\*\*\* START OF.*?\*\*\*', text, re.I)
    end = re.search(r'\*\*\* END OF.*?\*\*\*', text, re.I)

    start_idx = start.end() if start else 0
    end_idx = end.start() if end else len(text)

    return text[start_idx:end_idx].strip()

# ==========================================================
# TOC + NON-CONTENT REMOVAL
# ==========================================================

def preprocess_core_text(core_text):

    REMOVE_SECTIONS = {
        "CONTENTS",
        "TABLE OF CONTENTS",
        "PREFACE",
        "INDEX",
        "QUESTIONS",
        "EXERCISES",
        "PROBLEMS",
        "APPENDIX",
        "BIBLIOGRAPHY",
        "REFERENCES"
    }

    # Matches structural headers that signal real content resuming
    structural_header_regex = re.compile(
        r'^(BOOK|CHAPTER|SECTION|PROP\.)\s+[IVXLCDM\d]+',
        re.IGNORECASE
    )

    # --------------------------------------------------
    # TOC STRIPPING (original logic — intentionally kept)
    # Slices from TOC marker to use it as a landmark, then
    # paragraph scan below finds and removes it properly.
    # --------------------------------------------------
    for marker in ["TABLE OF CONTENTS", "CONTENTS"]:
        pos = core_text.rfind(marker)

        if pos != -1:
            core_text = core_text[pos:]
            break

    paragraphs = core_text.split("\n\n")

    # --------------------------------------------------
    # FIND FIRST REAL CONTENT
    # --------------------------------------------------
    start_idx = -1

    for i in range(len(paragraphs) - 1):

        if i < 2:
            continue

        if len(paragraphs[i]) > 100 and len(paragraphs[i + 1]) > 100:
            start_idx = i
            break

    if start_idx != -1:
        core_text = "\n\n".join(paragraphs[max(0, start_idx - 2):])

    # --------------------------------------------------
    # REMOVE NON-CONTENT SECTIONS
    # FIX (Issue 1): Original logic only skipped the heading
    # paragraph itself. Now when a REMOVE_SECTION heading is
    # detected, we enter a skipping state and suppress ALL
    # subsequent paragraphs until the next structural header
    # (BOOK/CHAPTER/SECTION/PROP.) is found — at which point
    # real content resumes and skipping is turned off.
    # --------------------------------------------------
    cleaned = []
    skipping_front = True   # True until first real content block
    skipping_section = False  # True when inside a removed section

    for p in core_text.split("\n\n"):

        p = normalize_whitespace(p)

        if not p:
            continue

        upper = p.upper()

        is_bad_heading = any(upper == h or upper.startswith(h) for h in REMOVE_SECTIONS)
        is_structural_header = bool(structural_header_regex.match(p))
        is_long = len(p) > 200

        # --- Front-matter skip (original behaviour preserved) ---
        if skipping_front:
            if is_structural_header or is_long:
                skipping_front = False
            else:
                if is_bad_heading:
                    continue
                continue

        # --- Section-body skip (new) ---
        if skipping_section:
            # Resume only when a structural header is found
            if is_structural_header:
                skipping_section = False
                cleaned.append(p)
            # Everything else inside the bad section is dropped
            continue

        # --- Entering a bad section (new) ---
        if is_bad_heading:
            skipping_section = True
            # Drop the heading itself too
            continue

        cleaned.append(p)

    return "\n\n".join(cleaned)

# ==========================================================
# STRUCTURAL SPLIT
# ==========================================================

def structural_split(core_text):

    patterns = [
        r'\n(?=BOOK\s+)',
        r'\n(?=CHAPTER\s+)',
        r'\n(?=SECTION\s+)',
        r'\n(?=PROP\.\s+)'
    ]

    return re.split("|".join(patterns), core_text, flags=re.I)

# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

# ==========================================================
# SENTENCE BOUNDARY HELPER
# ==========================================================

def get_sentences_spacy(text):
    """Returns a list of sentence strings from text using spaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# ==========================================================
# SEMANTIC BOUNDARY HELPERS
# ==========================================================

def get_sentence_aware_split(text, target_limit):
    """
    Finds the best sentence boundary near target_limit.
    Allows 'Sentential Giants' up to 2x target_limit to keep logic intact.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    current_text = ""
    hard_limit = target_limit * 2  # The "Reasonable Ceiling"
    
    for i, sent in enumerate(sentences):
        # 1. Normal case: fits within target
        if len(current_text) + len(sent) <= target_limit:
            current_text = (current_text + " " + sent).strip()
        
        # 2. Sentential Giant: Over target but under hard ceiling
        elif len(current_text) + len(sent) <= hard_limit:
            current_text = (current_text + " " + sent).strip()
            return current_text, " ".join(sentences[i+1:]).strip()
            
        # 3. Excessive case: Must break to avoid extreme chunk sizes
        else:
            if current_text:
                return current_text, " ".join(sentences[i:]).strip()
            else:
                # Single sentence is massive; emit it as one chunk
                return sent, " ".join(sentences[i+1:]).strip()
                
    return current_text, ""

# ==========================================================
# MASTER CHUNK GENERATION
# ==========================================================

def split_large_paragraph(para, max_size):
    """
    Sentence-aware splitter using SpaCy to ensure logical units 
    are not broken across chunks.
    """
    # Assuming get_sentences_spacy(text) returns doc.sents or a list of strings
    sentences = get_sentences_spacy(para)
    chunks, current = [], ""
    
    for sent in sentences:
        # If adding the next sentence exceeds max_size, save current and start new
        if len(current) + len(sent) <= max_size:
            current = (current + " " + sent).strip()
        else:
            if current: 
                chunks.append(current)
            current = sent
            
    # Don't forget the final trailing chunk
    if current: 
        chunks.append(current)
        
    return chunks

def generate_master_chunks(documents, max_para_size):
    master_chunks = []
    global_master_id = 0
    for doc in documents:
        core_text = clean_gutenberg_ocr(doc["content"])
        core_text = preprocess_core_text(core_text)
        blocks = structural_split(core_text)
        for block in blocks:
            for para in block.split("\n\n"):
                para = normalize_whitespace(para)
                if not para: continue
                
                # Use the safe paragraph splitter if it exceeds target
                parts = split_large_paragraph(para, max_para_size) if len(para) > max_para_size else [para]
                
                for part in parts:
                    master_chunks.append({
                        "master_id": global_master_id,
                        "text": part,
                        "metadata": {"source": doc["filename"], "path": doc["path"]}
                    })
                    global_master_id += 1
    return master_chunks

# ==========================================================
# PACKING & OVERLAP LOGIC
# ==========================================================

def pack_chunks(master_chunks, min_size, mode="clean"):
    max_size = int(min_size * 1.5)
    packed = []
    buffer_text = ""
    last_flushed_text = ""
    current_metadata = None
    original_master_ids = []
    packed_id = 0

    for chunk in master_chunks:
        # Document Boundary Guard: Hard reset on source change
        if current_metadata and chunk["metadata"]["source"] != current_metadata["source"]:
            if buffer_text:
                packed.append(create_entry(packed_id, buffer_text, current_metadata, 
                                         original_master_ids, last_flushed_text, mode, max_size))
                packed_id += 1
            # Empty last_flushed_text ensures no overlap between different sources
            buffer_text, original_master_ids, last_flushed_text = "", [], ""
        
        current_metadata = chunk["metadata"]
        text, m_id = chunk["text"], chunk["master_id"]

        if len(buffer_text) + len(text) > max_size:
            if len(buffer_text) >= min_size:
                # Flush existing buffer
                packed.append(create_entry(packed_id, buffer_text, current_metadata, 
                                         original_master_ids, last_flushed_text, mode, max_size))
                packed_id += 1
                last_flushed_text, buffer_text, original_master_ids = buffer_text, text, [m_id]
            else:
                # Buffer is small; merge and split at sentence boundary
                combined = (buffer_text + " " + text).strip()
                truncated, leftover = get_sentence_aware_split(combined, max_size)
                
                packed.append(create_entry(packed_id, truncated, current_metadata, 
                                         original_master_ids + [m_id], last_flushed_text, mode, max_size))
                packed_id += 1
                last_flushed_text = truncated
                buffer_text = leftover
                original_master_ids = [m_id] if buffer_text else []
        else:
            buffer_text = (buffer_text + " " + text).strip()
            original_master_ids.append(m_id)

    if buffer_text:
        packed.append(create_entry(packed_id, buffer_text, current_metadata, 
                                 original_master_ids, last_flushed_text, mode, max_size))
    return packed

def create_entry(chunk_id, text, metadata, master_ids, prev_text, mode, max_size):
    final_text = text
    
    if mode == "overlap" and prev_text:
        prev_sents = get_sentences_spacy(prev_text)
        
        # PROPORTIONAL OVERLAP: Budget is 30% of the PREVIOUS chunk size
        overlap_budget = int(len(prev_text) * 0.3)
        
        selected_overlap_sents = []
        current_overlap_len = 0
        
        # Greedy search backwards to fill the semantic bridge
        for sent in reversed(prev_sents):
            if current_overlap_len + len(sent) + 1 <= overlap_budget:
                selected_overlap_sents.insert(0, sent)
                current_overlap_len += len(sent) + 1
            else:
                # Semantic logic safeguard: always provide at least one sentence 
                # as long as it isn't an extreme outlier (>50% of current max_size)
                if not selected_overlap_sents and len(sent) < (max_size * 0.5):
                    selected_overlap_sents.append(sent)
                break
        
        if selected_overlap_sents:
            overlap_prefix = " ".join(selected_overlap_sents)
            final_text = (overlap_prefix + " " + text).strip()

    return {
        "chunk_id": chunk_id,
        "text": final_text,
        "size": len(final_text),
        "original_master_ids": master_ids,
        "metadata": metadata
    }

# ==========================================================
# MASTER CHUNK GENERATION
# ==========================================================

def generate_master_chunks(documents, max_para_size):
    """
    Standardizes structural splitting and assigns unique master IDs.
    Returns the list of master chunks.
    """
    master_chunks = []
    global_master_id = 0
    
    for doc in documents:
        core_text = clean_gutenberg_ocr(doc["content"])
        core_text = preprocess_core_text(core_text)
        
        # Structural split (Book, Chapter, etc.)
        blocks = structural_split(core_text)
        
        for block in blocks:
            for para in block.split("\n\n"):
                para = normalize_whitespace(para)
                if not para:
                    continue
                
                # Sentence-aware split for huge paragraphs
                if len(para) > max_para_size:
                    parts = split_large_paragraph(para, max_para_size)
                else:
                    parts = [para]
                
                for part in parts:
                    master_chunks.append({
                        "master_id": global_master_id,
                        "text": part,
                        "metadata": {
                            "source": doc["filename"],
                            "path": doc["path"]
                        }
                    })
                    global_master_id += 1
    return master_chunks

# ==========================================================
# MAIN INGESTION PIPELINE
# ==========================================================
def ingestion_pipeline(config):
    mode = config.CHUNK_MODE.lower()
    output_dir = config.VECTOR_STORE_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Ingestion Mode: {mode.upper()} | Target Size: {config.CHUNK_SIZE}")
    
    docs = load_documents(config.DATA_DIR)
    if not docs:
        return []

    # 1. Generate Master Chunks
    # We split at 1.5x target to ensure they are manageable for the packer
    master_chunks = generate_master_chunks(docs, int(config.CHUNK_SIZE * 1.5))
    
    # --- NEW: Save Master Chunks to the base folder ---
    master_output_path = os.path.join(output_dir, "master_chunks.json")
    with open(master_output_path, "w", encoding="utf-8") as f:
        json.dump(master_chunks, f, indent=2)
    print(f"[INFO] Saved {len(master_chunks)} master chunks to {master_output_path}")

    # 2. Strategic Packing/Merging
    chunks = pack_chunks(master_chunks, min_size=config.CHUNK_SIZE, mode=mode)

    # 3. Final Stats and Save Packed Chunks
    lengths = [len(c["text"]) for c in chunks]
    print("\n" + "=" * 40)
    print("FINAL PACKED STATISTICS")
    print("=" * 40)
    print(f"Total Packed Chunks: {len(chunks)}")
    print(f"Mean Size: {statistics.mean(lengths):.2f}")
    print(f"Max Size: {max(lengths)}")
    print(f"Min Size: {min(lengths)}")
    print("=" * 40 + "\n")
    
    packed_output_path = os.path.join(output_dir, "all_chunks.json")
    with open(packed_output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    return chunks