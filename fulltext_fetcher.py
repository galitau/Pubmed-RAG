"""
fulltext_fetcher.py
Module for fetching and processing full-text articles from PubMed Central (PMC).

This module provides a complete pipeline for:
  1. Converting PMID (PubMed ID) to PMC ID
  2. Fetching full-text XML from NCBI eFetch API
  3. Cleaning/parsing XML to readable plain text
  4. Splitting text into overlapping chunks for RAG
  5. Resolving user article references to actual article dicts
  
Used in the hierarchical RAG flow: when user references a specific paper,
this module enables fetching the full-text instead of just the abstract.
"""

import re
import time
import requests
from typing import Optional


# ── PMC Full-Text Fetch ──────────────────────────────────────────────────────

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pmc_fulltext(pmc_id: str, ncbi_api_key: Optional[str] = None) -> Optional[str]:
    """
    Fetch full-text article from PubMed Central via NCBI eFetch API.
    
    WHAT IT DOES:
      1. Takes a PMC ID (e.g., "PMC1234567" or just "1234567")
      2. Calls NCBI eFetch API to retrieve the full article as XML
      3. Cleans the XML to remove tags, figures, tables, formulas
      4. Returns readable plain text
    
    WHY PMC ID vs PMID:
      - PMID = PubMed ID (identifier in PubMed database)
      - PMC ID = PubMed Central ID (identifier in open-access PMC database)
      - PMC articles are freely accessible; paywalled articles aren't in PMC
      - The get_pmc_id() function converts PMID → PMC ID
    
    PARAMETERS:
      pmc_id: String like "PMC1234567" or "1234567" (handles both)
      ncbi_api_key: Optional NCBI API key (rate limit: 3 requests/sec without key, 
                    10 requests/sec with key). Define in .env as NCBI_API_KEY
    
    RETURNS:
      Clean text string of the full article, or None if:
        - Network request fails
        - PMC ID not found in NCBI
        - Response is not an article (empty/error response)
        - XML parsing fails
    
    EXAMPLE:
      >>> text = fetch_pmc_fulltext("PMC3879347")
      >>> print(len(text))  # Thousands of characters of clean article text
    """
    # Normalize PMC ID: "PMC1234567" → "1234567"
    # This allows callers to pass either format
    pmc_numeric = str(pmc_id).upper().lstrip("PMC")

    # Build NCBI eFetch API request parameters
    # See: https://www.ncbi.nlm.nih.gov/books/NBK25499/
    params = {
        "db": "pmc",                # Database: PubMed Central
        "id": pmc_numeric,         # Article ID
        "rettype": "full",         # Return full article (not just abstract)
        "retmode": "xml",          # XML format (structured, easier to parse than HTML)
    }
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key  # Rate limit boost: 10 req/sec instead of 3

    try:
        # Make HTTP GET request to NCBI eFetch
        resp = requests.get(EFETCH_URL, params=params, timeout=15)
        resp.raise_for_status()  # Raise exception if status code != 200
        xml_text = resp.text

        # Sanity check: verify we got a real article, not an empty response
        # NCBI might return empty XML or error XML if the ID is invalid
        if "<PubmedArticle>" not in xml_text and "<article" not in xml_text.lower():
            return None

        return _clean_xml(xml_text)

    except Exception:
        # Network error, timeout, malformed response, etc.
        return None


def _clean_xml(xml: str) -> str:
    """
    Convert NCBI XML response to readable plain text.
    
    WHAT IT DOES:
      1. Removes XML declaration/DTD lines (<?xml...>, <!DOCTYPE...>)
      2. Strips out large blocks (figures, tables, formulas) that add noise
      3. Removes all XML tags (<tag>, </tag>, etc.)
      4. Decodes XML entities (&amp; → &, &#x2013; → –, etc.)
      5. Collapses multiple spaces into single spaces
    
    WHY THIS CLEANING IS NEEDED:
      Raw NCBI XML contains lots of structural markup that's:
        - Not useful for semantic search (figures, tables don't help RAG)
        - Creates noise in embeddings
        - Wastes tokens when passed to LLM
      
      This function produces clean, readable text suitable for:
        - Embedding by Google Generative AI (semantic search)
        - Token-efficient LLM input
        - Human-readable context in chat
    
    EXAMPLE INPUT (XML):
      <article>
        <title>How to Cure Everything</title>
        <body>
          <p>The answer is &amp; <bold>42</bold>.</p>
          <fig><label>Fig 1</label><caption>A figure</caption></fig>
        </body>
      </article>
    
    EXAMPLE OUTPUT (text):
      "How to Cure Everything The answer is & 42."
    """
    # Step 1: Remove XML metadata lines
    text = re.sub(r"<\?xml[^>]*\?>", "", xml)  # <?xml version="1.0"?>
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text)  # <!DOCTYPE article PUBLIC ...>

    # Step 2: Strip out noisy block elements
    # These contain metadata/figures/tables that clutter the text
    # re.DOTALL makes . match newlines too, so we can match multi-line blocks
    for block_tag in ("fig", "table-wrap", "disp-formula", "inline-formula", "supplementary-material"):
        text = re.sub(
            rf"<{block_tag}[\s>].*?</{block_tag}>",
            " ",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Step 3: Remove all remaining XML tags
    # Replace any <...> with a space to preserve word boundaries
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 4: Decode XML entities
    # These are escape sequences for special characters in XML
    # &amp; = &, &lt; = <, &#x2013; = – (en dash), etc.
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&#x2013;", "–")  # en dash
        .replace("&#x2014;", "—")  # em dash
        .replace("&#x00A0;", " ")  # non-breaking space
    )

    # Step 5: Normalize whitespace
    # Replace all multi-space sequences with single space
    # Remove leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping word-level chunks for RAG embedding.
    
    WHY CHUNKING IS NEEDED:
      - LLMs have token limits (e.g., 128K tokens for Gemini 2.5-flash)
      - Full-text articles are thousands of words = tens of thousands of tokens
      - Chunking allows us to pass only relevant sections to the LLM
      - Shorter contexts = better relevance + faster responses + lower cost
      - Overlapping chunks prevent losing context at boundaries
    
    PARAMETERS:
      text: Full article text to split
      chunk_size: ~words per chunk (default: 800 words ≈ 1000-1200 tokens)
      overlap: words shared between consecutive chunks (default: 100)
               Prevents losing information at chunk boundaries
    
    HOW IT WORKS:
      1. Split text into words (simple whitespace split)
      2. Create sliding window of size chunk_size
      3. Overlap previous window by 'overlap' words
      4. Example: chunk_size=3, overlap=1:
         Input: ["A", "B", "C", "D", "E"]
         Output: ["A B C", "C D E"]
    
    RETURNS:
      List of overlapping text chunks (each is a string)
    
    EXAMPLE:
      >>> chunks = chunk_text("The quick brown fox jumps", chunk_size=2, overlap=1)
      >>> print(chunks)
      ["The quick", "quick brown", "brown fox", "fox jumps"]
    """
    words = text.split()  # Simple whitespace tokenization
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        # Calculate end position: either start + chunk_size or end of text
        end = min(start + chunk_size, len(words))
        
        # Create chunk from word range [start:end]
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # If we've reached the end, stop
        if end == len(words):
            break
        
        # Slide window forward by (chunk_size - overlap)
        # This creates the overlapping effect
        # Example: chunk_size=100, overlap=20 → stride forward by 80 words
        start += chunk_size - overlap

    return chunks


# ── Article Resolution ────────────────────────────────────────────────────────

def resolve_article(identifier: str, abstracts_text: str) -> Optional[dict]:
    """
    Match a user's article reference to an actual article dict.
    
    WHAT IT DOES:
      Converts vague user input like "source 2", "the Johnson study", or "paper about biocompatibility"
      into a concrete article dict with: {pmid, title, authors, link, is_free}
      
      Uses a 4-tier matching strategy (in order of specificity):
        1. Numeric references: "source 2" → article #2
        2. PMID match: "12345678" → article with that PMID
        3. Title substring: "biocompatibility" → article with "biocompatibility" in title
        4. Author match: "Johnson" → article by Johnson
    
    PARAMETERS:
      identifier: User's freeform reference to an article (e.g., "source 2", "Johnson", "PMID 12345678")
      abstracts_text: Full session abstracts string (formatted by app.py with TITLE/AUTHORS/LINK sections)
    
    RETURNS:
      Article dict with keys: {pmid, title, authors, link, is_free}
      OR None if no match found (fall back to general vector search)
    
    EXPECTED FORMAT OF abstracts_text:
      TITLE: Article 1 Title
      AUTHORS: Smith J, Doe A
      ABSTRACT: Lorem ipsum...
      FREE: True
      LINK: https://pubmed.ncbi.nlm.nih.gov/12345678/
      --------------------
      TITLE: Article 2 Title
      AUTHORS: Johnson B
      ...
    
    EXAMPLE MATCHES:
      - "source 2" → matches article at index 1 (2nd article)
      - "Johnson" → matches first article with "Johnson" in authors
      - "biocompatibility" → matches article with "biocompatibility" in title (>40% word overlap)
      - "PMID 12345678" → exact PMID match
    """
    # Parse all articles from the session abstracts
    blocks = _parse_abstract_blocks(abstracts_text)
    if not blocks:
        return None

    ident_lower = identifier.lower().strip()

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 1: Numeric reference ("source N", "reference N", "paper N", etc.)
    # ═══════════════════════════════════════════════════════════════════════════
    # Match patterns like "source 2", "#3", "paper 1", "the first study"
    num_match = re.search(r"\b(\d+)\b", ident_lower)  # Find first number
    if num_match and any(kw in ident_lower for kw in ("source", "reference", "paper", "ref", "article", "study", "no", "#")):
        # Found a number and a keyword indicating ordinal reference
        idx = int(num_match.group(1)) - 1  # Convert 1-based (user) to 0-based (Python)
        if 0 <= idx < len(blocks):
            return blocks[idx]

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 2: Direct PMID match
    # ═══════════════════════════════════════════════════════════════════════════
    # User might say "PMID 12345678" or just paste the number
    for block in blocks:
        if block.get("pmid") and block["pmid"] in ident_lower:
            return block

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 3: Title substring match (fuzzy scoring)
    # ═══════════════════════════════════════════════════════════════════════════
    # User might say "the paper about biocompatibility" or "3D printed bone"
    # Score each article by fraction of multi-word user input found in its title
    best_block = None
    best_score = 0
    for block in blocks:
        title_lower = block.get("title", "").lower()
        # Extract only meaningful words (>3 chars) from user input to avoid matching "the", "and", etc.
        words = [w for w in ident_lower.split() if len(w) > 3]
        if not words:
            continue
        # Score = fraction of user's words found in the title
        # Example: user says "bone scaffolds biocompatibility"
        #          title has "bone" and "scaffolds" (2/3) → score = 0.67
        hits = sum(1 for w in words if w in title_lower)
        score = hits / len(words)
        if score > best_score:
            best_score = score
            best_block = block

    # Threshold: require at least 40% word match to count as a title hit
    if best_score >= 0.4:
        return best_block

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 4: Author name match
    # ═══════════════════════════════════════════════════════════════════════════
    # User might say "the Johnson study" or "Smith et al"
    # Check if any multi-char word from identifier appears in authors
    for block in blocks:
        authors_lower = block.get("authors", "").lower()
        for word in ident_lower.split():
            if len(word) > 3 and word in authors_lower:
                return block

    # No match found using any strategy
    return None


def _parse_abstract_blocks(abstracts_text: str) -> list[dict]:
    """
    Parse session abstracts string into structured article dicts.
    
    WHAT IT DOES:
      Takes the concatenated abstracts stored in st.session_state['abstracts']
      and parses it into a list of article dicts, each with:
        {
          "title": str,
          "authors": str (comma-separated),
          "link": str (URL to PubMed),
          "pmid": str (extracted from link),
          "is_free": bool (from FREE field)
        }
    
    INPUT FORMAT (created by app.py):
      Each article is a block separated by "────────────────────" (20 dashes):
      
      TITLE: Article Title Here
      AUTHORS: Smith J, Doe A, Johnson B
      ABSTRACT: The abstract text...
      FREE: True
      LINK: https://pubmed.ncbi.nlm.nih.gov/12345678/
      ────────────────────
      TITLE: Another Article
      ...
    
    PARSING STRATEGY:
      1. Split on dashes to isolate blocks
      2. Use regex to find TITLE: ... AUTHORS: ... LINK: ... FREE: ... fields
      3. Extract PMID from PubMed URL
      4. Collect into dicts
    
    RETURNS:
      List of article dicts (one per article block with a title)
      Empty list if no articles found
    """
    blocks = []
    raw_blocks = abstracts_text.split("-" * 20)  # Split on 20-dash separator

    for raw in raw_blocks:
        raw = raw.strip()
        if not raw:
            continue

        block = {}

        # Extract TITLE field: everything after "TITLE:" until newline or next field
        title_m = re.search(r"TITLE:\s*(.+?)(?:\n|AUTHORS:|$)", raw, re.IGNORECASE | re.DOTALL)
        if title_m:
            block["title"] = title_m.group(1).strip()

        # Extract AUTHORS field
        authors_m = re.search(r"AUTHORS:\s*(.+?)(?:\n|ABSTRACT:|$)", raw, re.IGNORECASE | re.DOTALL)
        if authors_m:
            block["authors"] = authors_m.group(1).strip()

        # Extract LINK field (full HTTP URL)
        link_m = re.search(r"LINK:\s*(https?://\S+)", raw, re.IGNORECASE)
        if link_m:
            block["link"] = link_m.group(1).strip()
            # Extract PMID from the URL: pubmed.ncbi.nlm.nih.gov/{PMID}
            pmid_m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", block["link"])
            if pmid_m:
                block["pmid"] = pmid_m.group(1)

        # Extract FREE field: indicates if article is in PMC (open-access)
        free_m = re.search(r"FREE:\s*(True|False)", raw, re.IGNORECASE)
        if free_m:
            block["is_free"] = free_m.group(1).lower() == "true"

        # Only add block if it has a title (valid article)
        if block.get("title"):
            blocks.append(block)

    return blocks


# ── PMC ID Lookup ─────────────────────────────────────────────────────────────

def get_pmc_id(pmid: str, ncbi_api_key: Optional[str] = None) -> Optional[str]:
    """
    Convert PMID (PubMed ID) to PMC ID (PubMed Central ID).
    
    WHY THIS CONVERSION:
      - PMID and PMC ID are different identifiers in different NCBI databases
      - Not all PMIDs have corresponding PMC IDs
        - Only open-access articles are in PMC
        - Paywalled/subscription articles are NOT in PMC
      - fetch_pmc_fulltext() requires PMC ID, not PMID
      - This function handles the conversion via NCBI ID Converter API
    
    WHAT IT DOES:
      1. Calls NCBI ID Converter API with PMID
      2. Returns PMC ID if found (e.g., "PMC1234567")
      3. Returns None if conversion fails or PMID not in PMC
    
    PARAMETERS:
      pmid: String with just the numeric ID (e.g., "12345678", not "PMID 12345678")
      ncbi_api_key: Optional API key for rate limiting (5 requests/sec with key vs 2 without)
    
    RETURNS:
      PMC ID string like "PMC1234567", or None if:
        - Network error
        - PMID not found in NCBI
        - Article is paywalled (no PMC ID available)
        - API timeout
    
    EXAMPLE:
      >>> pmc_id = get_pmc_id("12345678")
      >>> print(pmc_id)  # "PMC1234567" or None
      >>> full_text = fetch_pmc_fulltext(pmc_id)  # Now we can fetch it
    
    NOTE:
      - Call this BEFORE calling fetch_pmc_fulltext()
      - Check if result is not None before proceeding
      - Calling with None/empty PMID returns None safely
    """
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "ids": pmid,           # The PMID to convert
        "format": "json"       # Return JSON (easier to parse than XML)
    }
    if ncbi_api_key:
        params["api_key"] = ncbi_api_key  # Rate limit boost

    try:
        # Make request to NCBI ID Converter
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # NCBI returns a dict with "records" list
        # Each record has fields like "pmid", "pmcid", etc.
        records = data.get("records", [])
        
        # If we have results and the first record has a pmcid field, return it
        if records and "pmcid" in records[0]:
            return records[0]["pmcid"]  # e.g., "PMC1234567"
    except Exception:
        # Network error, timeout, JSON parsing error, etc.
        pass
    
    return None  # No PMC ID found or API error