
import os
import re
import json
import copy
import gc
from tqdm import tqdm

def load_documents(raw_data_path: str) -> list:
    """
    Loads text content from all .txt files within the specified directory and its subdirectories.

    Args:
        raw_data_path (str): The root directory containing the text files.

    Returns:
        list: A list of dictionaries, where each dictionary represents a file
              and contains 'filename', 'path', and 'content'.
    """
    text_files = []
    if not os.path.exists(raw_data_path):
        print(f"\nError: Path '{raw_data_path}' not found.")
        return text_files

    for root, _, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        text_files.append({
                            'filename': file,
                            'path': file_path,
                            'content': content
                        })
                except Exception as e:
                    print(f"\nError reading file {file_path}: {e}")
    print(f'[INFO] Total Documents Loaded: {len(text_files)} | Document Type: .txt')
    return text_files

def clean_document_content(documents: list) -> list:
    """
    Cleans the content of documents by normalizing line endings, handling paragraph breaks,
    and removing specific patterns.

    Args:
        documents (list): A list of dictionaries, each with at least a 'content' key.

    Returns:
        list: A new list of dictionaries with cleaned content.
    """
    cleaned_docs = copy.deepcopy(documents) # Work on a copy

    for i, file_info in enumerate(cleaned_docs):
        content = file_info['content']

        # Normalize line endings to \n for consistency
        content = content.replace('\r\n', '\n')

        # Apply Rule 1: 'n\n' followed by one or more spaces indicates a paragraph break.
        # This regex replaces any sequence of two or more newlines followed by any whitespace
        # with a paragraph break placeholder, effectively collapsing multiple newlines and spaces.
        content = re.sub(r'\n{2,}\s*', '[PARAGRAPH_BREAK_PLACEHOLDER]', content)

        # Apply Rule 2: '\n' followed by more than one space until a character is found indicates a single space.
        # This targets single newlines followed by two or more spaces.
        content = re.sub(r'\n\s{2,}', ' ', content)

        # Any remaining single '\n' (not part of a collapsed paragraph break, or '\n' followed by 0 or 1 space)
        # should be replaced with a single space.
        content = content.replace('\n', ' ')

        # Restore the paragraph breaks
        content = content.replace('[PARAGRAPH_BREAK_PLACEHOLDER]', '\n\n')

        # Step 3: Remove 'simple' if it occurs before or after a single quote
        # Regex to find 'simple'' (simple followed by a single quote)
        content = re.sub(r"simple'", "'", content, flags=re.IGNORECASE)
        # Regex to find ''simple' (single quote followed by simple)
        content = re.sub(r"'simple", "'", content, flags=re.IGNORECASE)

        file_info['content'] = content

    print(f"✔ Initial cleaning completed.")
    return cleaned_docs

def second_clean_document_content(documents: list) -> list:
    """
    Hardened schematic pruning using the 'Consecutive Dense Blocks' heuristic.
    Locates the book start and strictly preserves the two preceding headers.
    """
    re_cleaned_docs = copy.deepcopy(documents)

    start_g_pattern = r'\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*'
    end_g_pattern = r'\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*'

    for file_info in re_cleaned_docs:
        content = file_info['content']

        # 1. Strip the Gutenberg wrapper
        s_match = re.search(start_g_pattern, content, re.IGNORECASE)
        e_match = re.search(end_g_pattern, content, re.IGNORECASE)
        start_idx = s_match.end() if s_match else 0
        end_idx = e_match.start() if e_match else len(content)
        core_text = content[start_idx:end_idx].strip()

        # 2. Pivot to 'CONTENTS' (UPPERCASE)
        toc_markers = ['TABLE OF CONTENTS', 'CONTENTS']
        pivot_idx = -1
        for marker in toc_markers:
            pos = core_text.rfind(marker)
            if pos != -1:
                pivot_idx = pos
                break

        if pivot_idx != -1:
            core_text = core_text[pivot_idx:]

        # 3. Apply the Consecutive Dense Block Pivot
        paragraphs = core_text.split('\n\n')
        target_idx = -1

        for idx in range(len(paragraphs) - 1):
            # Skip the 'CONTENTS' header area
            if idx < 2:
                continue

            para_1 = paragraphs[idx].strip()
            para_2 = paragraphs[idx+1].strip()

            # THE CRITERIA:
            # Sequential paragraphs > 100 characters to break the TOC pattern.
            if len(para_1) > 100 and len(para_2) > 100:
                target_idx = idx
                break

        if target_idx != -1:
            # STRICT BACKTRACK:
            # Grab exactly two paragraphs before the first dense block.
            # This captures [Chapter Number] and [Chapter Title].
            keep_from = max(0, target_idx - 2)
            core_text = '\n\n'.join(paragraphs[keep_from:])

        file_info['content'] = core_text.strip()

    print(f"✔ Schematic Cleaning Completed.")
    return re_cleaned_docs

def find_semantic_overlap_start(content, end_pos, overlap_target):
    """
    Guarantees overlap by looking back from the end_pos and finding
    the best semantic break near the overlap_target.
    """
    # Define the search window for the overlap start
    # We look back from the end of the previous chunk
    search_start = max(0, end_pos - (overlap_target + 100))
    search_end = end_pos - int(overlap_target * 0.5) # Ensure at least 50% of target overlap

    overlap_window = content[search_start:search_end]

    # Priority 1: Double Newline
    indices = [m.start() for m in re.finditer(r'\n\n', overlap_window)]
    if indices:
        return search_start + indices[-1] + 2

    # Priority 2: Sentence end
    indices = [m.end() for m in re.finditer(r'[.!?](\s+)', overlap_window)]
    if indices:
        return search_start + indices[-1]

    # Fallback: Just use the mathematical overlap
    return max(0, end_pos - overlap_target)

def chunk_documents(documents, chunk_size, overlap) -> list:
    all_chunks = []
    chunk_id_counter = 0

    for file_info in tqdm(documents, desc="Chunking Progress"):
        content = file_info['content']
        filename = file_info['filename']
        filepath = file_info['path']

        total_len = len(content)
        current_pos = 0

        while current_pos < total_len:
            # 1. Determine End Point (Backtrack from chunk_size)
            end_pos = min(current_pos + chunk_size, total_len)

            if end_pos < total_len:
                chunk_raw = content[current_pos:end_pos]
                # Look for end-of-chunk break in the last 30% of the window
                search_buffer_start = int(chunk_size * 0.7)
                search_buffer = chunk_raw[search_buffer_start:]

                # Prioritize paragraph/sentence ends for the CHUNK END
                break_points = list(re.finditer(r'(\n\n|\n|[.!?]\s+)', search_buffer))
                if break_points:
                    end_pos = current_pos + search_buffer_start + break_points[-1].end()
                else:
                    # Fallback to space
                    last_space = chunk_raw.rfind(' ')
                    if last_space != -1 and last_space > (chunk_size * 0.5):
                        end_pos = current_pos + last_space + 1

            chunk_text = content[current_pos:end_pos].strip()

            if chunk_text:
                all_chunks.append({
                    "chunk_id": chunk_id_counter,
                    "text": chunk_text,
                    "source": filename,
                    "path": filepath,
                    "char_start": current_pos
                })
                chunk_id_counter += 1

            if end_pos >= total_len:
                break

            # 2. GUARANTEED OVERLAP LOGIC
            new_pos = find_semantic_overlap_start(content, end_pos, overlap)

            # Safety check
            if new_pos <= current_pos:
                new_pos = end_pos

            current_pos = new_pos

        gc.collect()

    return all_chunks
    
def save_chunks_to_json(chunks: list, output_directory: str, filename: str = 'all_chunks.json'):
    """
    Saves a list of chunks to a JSON file in the specified directory.

    Args:
        chunks (list): The list of chunk dictionaries to save.
        output_directory (str): The directory where the JSON file will be saved.
        filename (str): The name of the JSON file.
    """
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, filename)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)
        print(f"✔ Chunks Saved as JSON.")
    except Exception as e:
        print(f"❌ Error saving chunks to {output_path}: {e}")

def ingestion_pipeline(base_path, chunk_size:int=200, overlap:int=50):
    """
    Main function to orchestrate the document processing workflow.
    Allows for dynamic adjustment of chunking parameters.
    """
    # Join the base_path with the specific sub-folders
    raw_data_path = os.path.join(base_path, 'data', 'logic_history_corpus')
    output_dir = os.path.join(base_path, 'vector_store')
    
    output_filename = 'all_chunks.json'

    # Step 1: Load documents
    documents = load_documents(raw_data_path)
    if not documents:
        print("❌ No documents loaded. Please Ensure to upload atleast 1 document to start the chunking process.")
        return

    # Step 2: Clean document content
    cleaned_documents = clean_document_content(documents)
    second_cleaned_documents = second_clean_document_content(cleaned_documents)
    
    # Step 3: Chunk documents with provided parameters
    chunks = chunk_documents(second_cleaned_documents, chunk_size=chunk_size, overlap=overlap)
    print(f'[INFO] Chunk Size: {chunk_size} | Chunk Overlap: {overlap} | Total Chunks Created: {len(chunks)}')

    # Step 4: Save chunks to JSON
    save_chunks_to_json(chunks, output_dir, output_filename)
    return chunks