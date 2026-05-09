import pandas as pd
from tabulate import tabulate
import textwrap

def chunk_display(chunks, title=None):
    """Prints a beautiful table of retrieved chunks to the terminal."""
    if not chunks:
        print(f"\n[WARN] No chunks found for {title}")
        return

    if title:
        print(f"\n>>> {title}")

    display_list = []
    for idx, chunk in enumerate(chunks):
        # Get source from dictionary or nested metadata
        source = chunk.get('source') or chunk.get('metadata', {}).get('source', 'N/A')
        
        display_list.append({
            "Rank": idx + 1,
            "Source": textwrap.fill(source, width=20),
            "Similarity Score": round(chunk.get('score', 0), 4) if chunk.get('score') else "N/A",
            "Re-Ranker Score": round(chunk.get('relevance_score', 0), 4) if chunk.get('relevance_score') else "N/A",
            "Precision Verdict": chunk.get('relevance', '-'),
            "Text Snippet": textwrap.fill(chunk.get('text', 'N/A'), width=100)
        })

    df = pd.DataFrame(display_list)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))