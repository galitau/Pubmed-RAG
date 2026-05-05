import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None


class ResearchDB:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "pubmed"):
        self.enabled = chromadb is not None
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_function = None

        if not self.enabled:
            return

        api_key = os.getenv("GEMINI_API_KEY")

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
        except Exception:
            self.enabled = False

    # ── Abstract collection (existing) ────────────────────────────────────────

    def reset_collection(self):
        """Delete the current collection and recreate an empty one."""
        if not self.enabled or not self.client:
            return False
        try:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            return self.collection is not None
        except Exception:
            return False

    def add_abstracts(self, abstracts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add documents to the ChromaDB collection."""
        if not self.enabled or not self.collection:
            return False
        if not (len(abstracts) == len(metadatas) == len(ids)):
            raise ValueError("Lengths of abstracts, metadatas and ids must match")
        try:
            self.collection.upsert(documents=abstracts, metadatas=metadatas, ids=ids)
            return True
        except Exception:
            return False

    def query_db(self, query_text: str, n_results: int = 5):
        """Query the abstract collection and return top results."""
        if not self.enabled or not self.collection:
            return {"ids": [], "documents": [], "metadatas": []}
        try:
            include_fields = ["documents", "metadatas", "distances", "embeddings"]
            res = self.collection.query(query_texts=[query_text], n_results=n_results, include=include_fields)
            docs = res.get("documents", [[]])[0]
            metadatas = res.get("metadatas", [[]])[0]
            ids = []
            if "ids" in res:
                try:
                    ids = res.get("ids", [[]])[0]
                except Exception:
                    ids = []
            return {"ids": ids, "documents": docs, "metadatas": metadatas}
        except Exception:
            return {"ids": [], "documents": [], "metadatas": []}

    # ── Per-article full-text collections ─────────────────────────────────────
    # 
    # ChromaDB allows multiple collections. The main collection stores ALL abstracts
    # for broad vector search. Per-article collections store FULL-TEXT CHUNKS for
    # deep semantic search within a single article. This creates a 2-tier index:
    #   - Abstract collection: fast broad search across all papers
    #   - Article collection (fulltext_PMID): precise search within one paper

    def _article_collection_name(self, pmid: str) -> str:
        """
        Generate a stable, deterministic collection name for an article's full-text chunks.
        
        Format: 'fulltext_{PMID}'
        Example: 'fulltext_12345678' for article with PMID 12345678
        
        This naming scheme ensures that if the same article is fetched multiple times,
        it always maps to the same collection (idempotent).
        """
        return f"fulltext_{pmid}"

    def article_collection_exists(self, pmid: str) -> bool:
        """
        Check if we've already stored full-text chunks for this PMID.
        
        This is used to avoid re-fetching and re-embedding the same article.
        If True, we can directly query the cached chunks via query_article_fulltext().
        If False, we need to fetch from PMC and call store_article_chunks().
        
        This provides a 2-tier cache:
          - Tier 1 (session memory): st.session_state['abstracts'] 
          - Tier 2 (disk cache): ChromaDB per-article collection
        """
        if not self.enabled or not self.client:
            return False
        try:
            col_name = self._article_collection_name(pmid)
            # List all existing collection names in the ChromaDB instance
            existing = [c.name for c in self.client.list_collections()]
            return col_name in existing
        except Exception:
            return False

    def store_article_chunks(
        self,
        pmid: str,
        chunks: List[str],
        title: str = "",
    ) -> bool:
        """
        Store full-text chunks for one article in its own dedicated ChromaDB collection.
        
        WHAT HAPPENS:
          1. Creates a collection named 'fulltext_{pmid}' if it doesn't exist
          2. Splits chunks into embeddings using Google Generative AI
          3. Stores each chunk with:
             - Document text (the actual paragraph/section)
             - ID: {pmid}_chunk_{index} (unique identifier)
             - Metadata: {pmid, title, chunk_index} (for tracking)
        
        WHY SEPARATE COLLECTION PER ARTICLE:
          - Semantic search is scoped to ONE article instead of all abstracts
          - Results in higher-quality retrieval within that article
          - Chunks stay organized and reusable
          - Can query independently or compare to other articles
        
        IDEMPOTENT (safe to call multiple times):
          - Uses upsert() instead of add(), so duplicate chunk IDs replace old values
          - Calling twice with same PMID just overwrites the stored chunks
        """
        if not self.enabled or not self.client:
            return False
        if not chunks:
            return False
        try:
            col_name = self._article_collection_name(pmid)
            # Get or create a collection specific to this article
            col = self.client.get_or_create_collection(
                name=col_name,
                embedding_function=self.embedding_function,  # Uses Google AI embeddings
            )
            # Generate unique, traceable IDs for each chunk
            ids = [f"{pmid}_chunk_{i}" for i in range(len(chunks))]
            # Store metadata so we know which article/chunk this is
            metadatas = [{"pmid": pmid, "title": title, "chunk_index": i} for i in range(len(chunks))]
            # Upsert = "insert if new, update if exists"
            col.upsert(documents=chunks, metadatas=metadatas, ids=ids)
            return True
        except Exception:
            return False

    def query_article_fulltext(
        self,
        pmid: str,
        query_text: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Semantic search WITHIN ONE ARTICLE's full-text chunks.
        
        WHAT IT DOES:
          1. Gets the article's collection (fulltext_{pmid})
          2. Embeds the user's query text using Google AI
          3. Finds the N most semantically similar chunks
          4. Returns them scored by relevance (closest matches first)
        
        RETURNS (same format as query_db for consistency):
          {
            "ids": ["PMID_chunk_2", "PMID_chunk_5", ...],
            "documents": ["chunk text...", "chunk text...", ...],
            "metadatas": [{"pmid": "...", "title": "...", "chunk_index": 2}, ...]
          }
        
        EXAMPLE USE CASE:
          - User asks: "How does this paper measure biocompatibility?"
          - This searches only article X's chunks for the most relevant sections
          - Returns highest-scoring sections instead of the whole abstract
          - Much better retrieval quality than keyword search
        
        NOTE: Returns empty dict if collection doesn't exist or has no chunks
        """
        if not self.enabled or not self.client:
            return {"ids": [], "documents": [], "metadatas": []}
        try:
            col_name = self._article_collection_name(pmid)
            col = self.client.get_or_create_collection(
                name=col_name,
                embedding_function=self.embedding_function,
            )
            # Sanity check: don't ask for more results than we have chunks
            count = col.count()
            if count == 0:
                # Empty collection = no chunks stored for this article yet
                return {"ids": [], "documents": [], "metadatas": []}
            n = min(n_results, count)
            
            # Perform semantic vector search
            # ChromaDB will:
            #   1. Embed query_text using the same embedding_function
            #   2. Compute cosine similarity to all stored chunks
            #   3. Return top N by similarity score
            res = col.query(
                query_texts=[query_text],
                n_results=n,
                include=["documents", "metadatas"],
            )
            
            # Extract results from ChromaDB's nested list format
            docs = res.get("documents", [[]])[0]  # First (and only) query's results
            metas = res.get("metadatas", [[]])[0]
            ids = []
            if "ids" in res:
                try:
                    ids = res["ids"][0]
                except Exception:
                    ids = []
            
            return {"ids": ids, "documents": docs, "metadatas": metas}
        except Exception:
            # If anything fails (collection doesn't exist, network error, etc.), return empty
            return {"ids": [], "documents": [], "metadatas": []}