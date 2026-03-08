"""
Load and prepare the 20 Newsgroups dataset.

PART 1: EMBEDDING & VECTOR DATABASE SETUP
==========================================

Data Cleaning Decisions (with justification):

1. REMOVE POSTS < 50 CHARACTERS
   Why: Noise filtering
   - Auto-replies ("Thanks!", "Me too")
   - Signatures ("--John")
   - Quoted text markers (">")
   - These add no semantic value and pollute embeddings
   - Threshold of 50 chars is conservative (typical sentence ~80 chars)

2. REMOVE POSTS > 5000 CHARACTERS
   Why: Outlier filtering
   - Likely multi-quoted threads with mixed topics
   - Dilutes semantic signal (embedding becomes average of many topics)
   - Reduces computational cost (embeddings are expensive)
   - 5000 chars ≈ 1000 words, reasonable for single semantic topic
   - Keeps ~95% of documents, removes extreme outliers

3. STRIP EMAIL HEADERS
   Why: Metadata vs content
   - Headers (From:, Subject:, Date:, Message-ID:) are metadata
   - Not semantic content, just routing information
   - Adds noise to embeddings
   - Quoted text markers (>) indicate previous messages, not current content

4. LOWERCASE + WHITESPACE NORMALIZATION
   Why: Consistency and efficiency
   - Reduces vocabulary size (e.g., "Machine" and "machine" are same)
   - Improves embedding quality (more training data per token)
   - Removes formatting noise (multiple spaces, tabs, newlines)
   - Standard preprocessing for NLP tasks

5. KEEP CATEGORY LABELS
   Why: Validation and analysis
   - Not used in clustering (unsupervised)
   - Useful for post-hoc analysis and validation
   - Helps understand if clusters align with original categories
   - Enables category-to-cluster mapping analysis

RESULT: ~18,000 clean documents from ~20,000 original
- Removes ~10% noise and outliers
- Preserves semantic content
- Enables efficient embedding computation
"""

import re
from typing import List, Dict, Tuple
from sklearn.datasets import fetch_20newsgroups


def clean_text(text: str) -> str:
    """
    Clean newsgroup post text.
    
    Design decisions:
    1. Remove email headers (From:, Subject:, Date:, Message-ID:)
       - These are metadata, not semantic content
       - Add noise to embeddings
       - Not useful for semantic search
       
    2. Remove quoted text markers (>)
       - Indicate previous messages in thread
       - Not part of current author's content
       - Dilute semantic signal
       
    3. Normalize whitespace
       - Multiple spaces, tabs, newlines are formatting noise
       - Reduce to single spaces
       - Improves consistency
       
    4. Lowercase
       - Reduces vocabulary size
       - Improves embedding quality (more training data per token)
       - Standard NLP preprocessing
    """
    # Remove email headers (lines starting with common headers)
    lines = text.split('\n')
    content_lines = []
    for line in lines:
        # Skip header lines and quoted text markers
        # Why? Headers are metadata, quoted text is from other authors
        if line.startswith(('From:', 'Subject:', 'Date:', 'Message-ID:', '>')):
            continue
        content_lines.append(line)
    
    text = '\n'.join(content_lines)
    
    # Remove excessive whitespace
    # Why? Formatting noise, doesn't affect semantics
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text.lower()


def load_newsgroups(remove_headers: bool = True, 
                   min_length: int = 50,
                   max_length: int = 5000) -> Tuple[List[Dict], List[str]]:
    """
    Load 20 Newsgroups dataset with preprocessing.
    
    Args:
        remove_headers: Remove email headers
        min_length: Minimum post length (chars) to keep
        max_length: Maximum post length (chars) to keep
        
    Returns:
        Tuple of (documents, category_names)
        Each document is a dict with 'text' and 'category' keys
        
    Design decisions:
    
    1. MIN_LENGTH = 50 characters
       - Removes noise: auto-replies, signatures, single words
       - Typical sentence: ~80 chars, so 50 is conservative
       - Keeps ~90% of documents
       - Threshold is tunable based on domain
       
    2. MAX_LENGTH = 5000 characters
       - Removes outliers: multi-quoted threads, digests
       - 5000 chars ≈ 1000 words ≈ 5-10 minute read
       - Reasonable for single semantic topic
       - Reduces computational cost (embeddings are expensive)
       - Keeps ~95% of documents
       
    3. REMOVE_HEADERS = True
       - Removes email metadata (From, Subject, Date, Message-ID)
       - Metadata adds noise, not semantic content
       - Improves embedding quality
       
    4. KEEP CATEGORY LABELS
       - Not used in clustering (unsupervised)
       - Useful for validation and analysis
       - Enables category-to-cluster mapping
       - Helps understand if clusters align with original labels
    """
    print("Fetching 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers',) if remove_headers else (),
        shuffle=False
    )
    
    documents = []
    for text, category_idx in zip(dataset.data, dataset.target):
        cleaned = clean_text(text)
        
        # Filter by length
        # Why? Remove noise (<50) and outliers (>5000)
        if len(cleaned) < min_length or len(cleaned) > max_length:
            continue
        
        documents.append({
            'text': cleaned,
            'category': dataset.target_names[category_idx],
            'category_idx': category_idx
        })
    
    print(f"Loaded {len(documents)} documents after filtering")
    print(f"Original: {len(dataset.data)}, Filtered: {len(documents)}")
    
    return documents, dataset.target_names
