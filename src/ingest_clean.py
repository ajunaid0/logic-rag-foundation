import os
import re
import json
import statistics

# ==========================================================
# WHITESPACE NORMALIZATION
# ==========================================================
def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
                    with open(path, "r", encoding="utf-8") as f:
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
# PARAGRAPH SPLITTER (SAFE)
# ==========================================================
def split_large_paragraph(para, max_size):

    sentences = re.split(
        r'(?<=[.!?])\s+(?=(?!\d+\.)[A-Z§])',
        para
    )

    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(sent) > max_size:
            flush()
            chunks.append(sent)
            continue

        if len(current) + len(sent) <= max_size:
            current = f"{current} {sent}".strip()
        else:
            flush()
            current = sent

    flush()
    return chunks


# ==========================================================
# MASTER CHUNKING
# ==========================================================
def chunk_documents(documents: list, chunk_size=800):

    master_chunks = []
    chunk_id = 0

    PARA_MAX_SIZE = int(chunk_size * 1.5)

    header_regex = re.compile(
        r'^(BOOK|CHAPTER|SECTION|PROP\.)\s+[IVXLCDM\d]+',
        re.IGNORECASE
    )

    REMOVE_SECTIONS = {
        "CONTENTS", "TABLE OF CONTENTS", "PREFACE",
        "INDEX", "QUESTIONS", "EXERCISES",
        "PROBLEMS", "APPENDIX", "BIBLIOGRAPHY", "REFERENCES"
    }

    for doc in documents:
        text = doc["content"]
        filename = doc["filename"]
        file_path = doc["path"]

        # --------------------------
        # CLEAN GUTENBERG + OCR
        # --------------------------
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        start = re.search(r'\*\*\* START OF.*?\*\*\*', text, re.I)
        end = re.search(r'\*\*\* END OF.*?\*\*\*', text, re.I)

        start_idx = start.end() if start else 0
        end_idx = end.start() if end else len(text)

        core_text = text[start_idx:end_idx].strip()

        # --------------------------
        # TOC REMOVAL
        # --------------------------
        for marker in ["TABLE OF CONTENTS", "CONTENTS"]:
            pos = core_text.rfind(marker)
            if pos != -1:
                core_text = core_text[pos:]
                break

        paragraphs = core_text.split("\n\n")

        # find first real content block
        start_idx = -1
        for i in range(len(paragraphs) - 1):
            if i < 2:
                continue
            if len(paragraphs[i]) > 100 and len(paragraphs[i + 1]) > 100:
                start_idx = i
                break

        if start_idx != -1:
            core_text = "\n\n".join(paragraphs[max(0, start_idx - 2):])

        # --------------------------
        # REMOVE NON-CONTENT SECTIONS
        # --------------------------
        cleaned = []
        skipping = True

        for p in core_text.split("\n\n"):
            p = normalize_whitespace(p)
            if not p:
                continue

            upper = p.upper()

            is_bad = any(h in upper for h in REMOVE_SECTIONS)
            is_header = bool(header_regex.match(p))
            is_long = len(p) > 200

            if skipping:
                if is_header or is_long:
                    skipping = False
                else:
                    if is_bad:
                        continue

            if not skipping and is_bad:
                continue

            cleaned.append(p)

        core_text = "\n\n".join(cleaned)

        # --------------------------
        # STRUCTURAL SPLIT
        # --------------------------
        patterns = [
            r'\n(?=BOOK\s+)',
            r'\n(?=CHAPTER\s+)',
            r'\n(?=SECTION\s+)',
            r'\n(?=PROP\.\s+)'
        ]

        raw_chunks = re.split("|".join(patterns), core_text, flags=re.I)

        # --------------------------
        # BUILD MASTER CHUNKS
        # --------------------------
        for block in raw_chunks:
            block = block.strip()
            if not block:
                continue

            for para in block.split("\n\n"):
                para = normalize_whitespace(para)
                if not para:
                    continue

                if len(para) > PARA_MAX_SIZE:
                    parts = split_large_paragraph(para, PARA_MAX_SIZE)
                else:
                    parts = [para]

                for part in parts:
                    master_chunks.append({
                        "chunk_id": chunk_id,
                        "text": part,
                        "metadata": {
                            "source": filename,
                            "path": file_path
                        }
                    })
                    chunk_id += 1

    print(f"[INFO] Master chunks created: {len(master_chunks)}")
    return master_chunks


# ==========================================================
# PACKING FUNCTION (SIZE CONTROLLED)
# ==========================================================
def pack_master_chunks(master_chunks, min_size=800):

    max_size = int(min_size * 1.5)

    packed = []

    buffer_text = ""
    buffer_size = 0
    start_idx = 0

    original_ids = []
    base_metadata = None

    packed_id = 0

    for i, chunk in enumerate(master_chunks):

        text = normalize_whitespace(chunk["text"])
        size = len(text)

        # store metadata reference once
        if base_metadata is None:
            base_metadata = chunk["metadata"]

        # --------------------------
        # OVERFLOW SINGLE CHUNK
        # --------------------------
        if size > max_size:

            if buffer_text:
                packed.append({
                    "chunk_id": packed_id,
                    "text": buffer_text,
                    "start_index": start_idx,
                    "end_index": i - 1,
                    "size": buffer_size,
                    "original_chunk_ids": original_ids,
                    "metadata": base_metadata
                })
                packed_id += 1

            packed.append({
                "chunk_id": packed_id,
                "text": text,
                "start_index": i,
                "end_index": i,
                "size": size,
                "original_chunk_ids": [chunk["chunk_id"]],
                "metadata": chunk["metadata"]
            })
            packed_id += 1

            buffer_text = ""
            buffer_size = 0
            start_idx = i + 1
            original_ids = []
            base_metadata = None

            continue

        # --------------------------
        # WOULD EXCEED LIMIT
        # --------------------------
        if buffer_size + size > max_size:

            if buffer_size < min_size:
                buffer_text = f"{buffer_text} {text}".strip()
                buffer_size += size
                original_ids.append(chunk["chunk_id"])
                continue

            packed.append({
                "chunk_id": packed_id,
                "text": buffer_text,
                "start_index": start_idx,
                "end_index": i - 1,
                "size": buffer_size,
                "original_chunk_ids": original_ids,
                "metadata": base_metadata
            })
            packed_id += 1

            buffer_text = text
            buffer_size = size
            start_idx = i
            original_ids = [chunk["chunk_id"]]
            base_metadata = chunk["metadata"]

        else:
            if not buffer_text:
                start_idx = i
                base_metadata = chunk["metadata"]
                original_ids = [chunk["chunk_id"]]
                buffer_text = text
                buffer_size = size
            else:
                buffer_text = f"{buffer_text} {text}".strip()
                buffer_size += size
                original_ids.append(chunk["chunk_id"])

    # --------------------------
    # FINAL FLUSH
    # --------------------------
    if buffer_text:
        packed.append({
            "chunk_id": packed_id,
            "text": buffer_text,
            "start_index": start_idx,
            "end_index": len(master_chunks) - 1,
            "size": buffer_size,
            "original_chunk_ids": original_ids,
            "metadata": base_metadata
        })

    return packed

# ==========================================================
# INGESTION PIPELINE
# ==========================================================
def ingestion_pipeline(base_path, chunk_size=800):
    raw_data_path = os.path.join(base_path, "data", "logic_history_corpus")
    output_dir = os.path.join(base_path, "vector_store")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Action: Chunking | Target Size: {chunk_size} chars")
    docs = load_documents(raw_data_path)

    if not docs:
        print("[FAILED] No documents found.")
        return []

    print("[PIPELINE] Creating master chunks...")
    master_chunks = chunk_documents(docs, chunk_size)

    print("[PIPELINE] Packing chunks...")
    final_chunks = pack_master_chunks(master_chunks, chunk_size)

    print(f"[SUCCESS] Final chunks: {len(final_chunks)}")


    chunk_lengths = [len(c["text"]) for c in final_chunks]

    max_chunk = max(chunk_lengths)
    min_chunk = min(chunk_lengths)
    mean_chunk = sum(chunk_lengths) / len(chunk_lengths)
    median_chunk = statistics.median(chunk_lengths)

    print("\n" + "=" * 60)
    print("CHUNK SIZE STATISTICS")
    print("=" * 60)
    print(f"Total Chunks: {len(final_chunks)}")
    print(f"Max Chunk Size: {max_chunk}")
    print(f"Min Chunk Size: {min_chunk}")
    print(f"Mean Chunk Size: {mean_chunk:.2f}")
    print(f"Median Chunk Size: {median_chunk}")
    print("=" * 60 + "\n")

    # 3. Save
    output_path = os.path.join(output_dir, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2)

    print(f"[SUCCESS] Pipeline Complete | Total Chunks: {len(final_chunks)} | File: {output_path}")


    return final_chunks