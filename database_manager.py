import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None


class ResearchDB:
    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "pubmed"):
        self.enabled = chromadb is not None # checks to make sure it was successfully imported
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = None

        if not self.enabled:
            return

        # Read API key for Google Generative AI embeddings
        api_key = os.getenv("GEMINI_API_KEY")

        # Initialize client and embedding function
        try:
            # Persistent client (so data is saved between restarts) - stores DB on disk
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Embedding function wrapper for Google Generative AI, which will be used to convert abstracts and queries into vector embeddings for similarity search
            self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)

            # Get or create collection with embedding function
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )

        except Exception as e:
            # Disable DB usage if initialization fails
            self.enabled = False

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
        """Add documents to the ChromaDB collection.

        abstracts: list of document text
        metadatas: list of dicts containing metadata (e.g., year, link)
        ids: list of unique ids for each document
        """
        if not self.enabled or not self.collection:
            return False

        # Ensure input lengths match
        if not (len(abstracts) == len(metadatas) == len(ids)):
            raise ValueError("Lengths of abstracts, metadatas and ids must match")

        try:
            self.collection.upsert(documents=abstracts, metadatas=metadatas, ids=ids)
            return True
        except Exception:
            return False

    def query_db(self, query_text: str, n_results: int = 5):
        """Query the collection and return top results.

        Returns a dict with keys: 'ids', 'documents', 'metadatas'
        """
        if not self.enabled or not self.collection:
            return {"ids": [], "documents": [], "metadatas": []}

        try:
            # Use include fields that are broadly supported by Chroma
            include_fields = ["documents", "metadatas", "distances", "embeddings"]
            res = self.collection.query(query_texts=[query_text], n_results=n_results, include=include_fields)

            # chroma returns nested lists for batch queries
            docs = res.get("documents", [[]])[0]
            metadatas = res.get("metadatas", [[]])[0]

            # Some Chroma versions return ids under 'ids' even if not requested; handle gracefully
            ids = []
            if "ids" in res:
                try:
                    ids = res.get("ids", [[]])[0]
                except Exception:
                    ids = []

            return {"ids": ids, "documents": docs, "metadatas": metadatas}
        except Exception:
            return {"ids": [], "documents": [], "metadatas": []}
