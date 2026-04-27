import os
import re
import json
import copy
from tqdm import tqdm

# ----------------------------
# DOCUMENT LOADER
# ----------------------------
def load_documents(raw_data_path: str) -> list:
    text_files = []
    
    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Path not found: {raw_data_path}")
        return []

    for root, _, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text_files.append({
                            "filename": file,
                            "path": path,
                            "content": f.read()
                        })
                except Exception as e:
                    print(f"[ERROR] Skip File: {file} | Reason: {e}")

    print(f"[INFO] Action: Load | Status: Success | Total Docs: {len(text_files)}")
    return text_files

# ----------------------------
# SENTENCE-AWARE OVERLAP
# ----------------------------
def get_start_of_sentence_overlap(full_text, overlap_target_size):
    """Ensures overlapping context starts at a clean sentence boundary."""
    if not full_text or len(full_text) <= 10:
        return ""

    actual_size = min(len(full_text), overlap_target_size)
    slice_to_search = full_text[-actual_size:]
    
    # Try to find a sentence boundary (. ! ?) followed by a space and capital letter
    sentence_boundary = re.search(r'[.!?]["\')\]]?\s+[A-Z]', slice_to_search)

    if sentence_boundary:
        start_match = re.search(r'[A-Z]', slice_to_search[sentence_boundary.start():])
        if start_match:
            final_start = sentence_boundary.start() + start_match.start()
            return slice_to_search[final_start:].strip()
    
    # Fallback: Word boundary
    word_boundary = re.search(r'\s+[A-Z]', slice_to_search)
    if word_boundary:
        return slice_to_search[word_boundary.start():].strip()

    first_space = slice_to_search.find(' ')
    return slice_to_search[first_space:].strip() if first_space != -1 else slice_to_search

# ----------------------------
# CHUNKING & CLEANING
# ----------------------------
def chunk_documents(documents: list, chunk_size=800) -> list:
    all_chunks = []
    dynamic_overlap = int(chunk_size * 0.30)
    chunk_id_counter = 0 
    
    MAX_CHUNK_SIZE = int(chunk_size * 1.5)
    MIN_CHUNK_SIZE = int(chunk_size * 0.8)

    header_regex = re.compile(r'^(BOOK|CHAPTER|SECTION|PROP\.)\s+[IVXLCDM\d]+', re.IGNORECASE)
    # sections to remove
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

    for doc in documents:
        text = doc['content']
        filename = doc['filename']
        file_path = doc['path']

        # 1. Clean OCR & Gutenberg Metadata
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        start_g = re.search(r'\*\*\*\s*START OF THE PROJECT GUTENBERG.*?(\*\*\*)', text, re.I)
        end_g = re.search(r'\*\*\*\s*END OF THE PROJECT GUTENBERG.*?(\*\*\*)', text, re.I)
        
        start_idx = start_g.end() if start_g else 0
        end_idx = end_g.start() if end_g else len(text)
        core_text = text[start_idx:end_idx].strip()

        # 2. Table of Contents Stripping
        toc_markers = ['TABLE OF CONTENTS', 'CONTENTS']
        pivot_idx = -1
        for marker in toc_markers:
            pos = core_text.rfind(marker)
            if pos != -1:
                pivot_idx = pos
                break
        if pivot_idx != -1:
            core_text = core_text[pivot_idx:]

        paragraphs = core_text.split('\n\n')
        target_idx = -1
        for idx in range(len(paragraphs) - 1):
            if idx < 2:
                continue
            if len(paragraphs[idx].strip()) > 100 and len(paragraphs[idx+1].strip()) > 100:
                target_idx = idx
                break

        if target_idx != -1:
            keep_from = max(0, target_idx - 2)
            core_text = '\n\n'.join(paragraphs[keep_from:])

        # ----------------------------
        # NEW: remove non-content sections
        # ----------------------------
        cleaned_paragraphs = []
        skipping = True

        for p in core_text.split('\n\n'):
            raw = p.strip()
            if not raw:
                continue

            upper = raw.upper()

            # detect section headers like PREFACE, INDEX, etc.
            is_bad_section = any(h in upper for h in REMOVE_SECTIONS)

            # detect real structural content
            is_real_structure = bool(header_regex.match(raw))

            # heuristic: long paragraph likely real content
            is_long_content = len(raw) > 200

            # stop skipping once real content starts
            if skipping:
                if is_real_structure or is_long_content:
                    skipping = False
                else:
                    if is_bad_section:
                        continue

            if not skipping and is_bad_section:
                continue

            cleaned_paragraphs.append(raw)

        core_text = '\n\n'.join(cleaned_paragraphs)

        # 3. Structural split
        patterns = [
            r'\n(?=BOOK\s+)',
            r'\n(?=CHAPTER\s+)',
            r'\n(?=SECTION\s+)',
            r'\n(?=PROP\.\s+)'
        ]
        combined_pattern = '|'.join(patterns)
        raw_structural_chunks = re.split(combined_pattern, core_text, flags=re.IGNORECASE)

        last_full_text = ""
        current_chunk = ""

        def flush_chunk(content):
            nonlocal chunk_id_counter, last_full_text

            content = content.strip()
            if not content:
                return

            clean_context = get_start_of_sentence_overlap(last_full_text, dynamic_overlap)
            final_text = (clean_context + content).strip()

            if len(final_text) > MAX_CHUNK_SIZE:
                final_text = final_text[:MAX_CHUNK_SIZE]

            all_chunks.append({
                "chunk_id": chunk_id_counter,
                "text": final_text,
                "metadata": {"source": filename, "path": file_path}
            })

            chunk_id_counter += 1
            last_full_text = content

        # 4. Balanced packing (no orphan chunks)
        for raw_chunk in raw_structural_chunks:
            raw_chunk = raw_chunk.strip()
            if not raw_chunk:
                continue

            sub_paragraphs = raw_chunk.split('\n\n')

            for para in sub_paragraphs:
                para = para.strip()
                if not para:
                    continue

                if not current_chunk:
                    current_chunk = para + "\n\n"
                    continue

                projected = len(current_chunk) + len(para)

                if projected <= MAX_CHUNK_SIZE:
                    current_chunk += para + "\n\n"
                else:
                    if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                        flush_chunk(current_chunk)
                        current_chunk = para + "\n\n"
                    else:
                        current_chunk += para + "\n\n"

        # final flush with orphan protection
        if current_chunk.strip():
            if len(current_chunk.strip()) < MIN_CHUNK_SIZE and all_chunks:
                all_chunks[-1]["text"] += " " + current_chunk.strip()
            else:
                flush_chunk(current_chunk)

    return all_chunks


def merge_or_flush_small_chunk(all_chunks, current_chunk, min_chunk_size):
    """
    Prevents orphan tiny chunks by merging into previous chunk instead of creating a standalone one.
    """
    if not current_chunk.strip():
        return all_chunks, ""

    if len(current_chunk.strip()) < min_chunk_size and all_chunks:
        all_chunks[-1]["text"] += " " + current_chunk.strip()
    else:
        all_chunks.append(current_chunk.strip())

    return all_chunks, ""
    
# ----------------------------
# INGESTION PIPELINE
# ----------------------------
def ingestion_pipeline(base_path, chunk_size=800):
    raw_data_path = os.path.join(base_path, "data", "logic_history_corpus")
    output_dir = os.path.join(base_path, "vector_store")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    docs = load_documents(raw_data_path)
    if not docs:
        print("[FAILED] No documents found to process.")
        return []

    # 2. Chunk
    print(f"[INFO] Action: Chunking | Target Size: {chunk_size} chars | Overlap: 30%")
    chunks = chunk_documents(docs, chunk_size)

    # ----------------------------
    # NEW: CHUNK SIZE STATS
    # ----------------------------
    import statistics

    chunk_lengths = [len(c["text"]) for c in chunks]

    max_chunk = max(chunk_lengths)
    min_chunk = min(chunk_lengths)
    mean_chunk = sum(chunk_lengths) / len(chunk_lengths)
    median_chunk = statistics.median(chunk_lengths)

    print("\n" + "=" * 60)
    print("CHUNK SIZE STATISTICS")
    print("=" * 60)
    print(f"Total Chunks: {len(chunks)}")
    print(f"Max Chunk Size: {max_chunk}")
    print(f"Min Chunk Size: {min_chunk}")
    print(f"Mean Chunk Size: {mean_chunk:.2f}")
    print(f"Median Chunk Size: {median_chunk}")
    print("=" * 60 + "\n")

    # 3. Save
    output_path = os.path.join(output_dir, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"[SUCCESS] Pipeline Complete | Total Chunks: {len(chunks)} | File: {output_path}")
    return chunks