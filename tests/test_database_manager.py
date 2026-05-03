import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import database_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env so GEMINI_API_KEY is available to tests
from dotenv import load_dotenv
load_dotenv()

from database_manager import ResearchDB


class TestResearchDB:
    """Unit tests for ResearchDB class"""

    def setup_method(self):
        """Create a test database before each test"""
        self.db = ResearchDB(persist_directory="test_chroma_db", collection_name="test_pubmed")

    def test_initialization(self):
        """Test that ResearchDB initializes correctly"""
        assert self.db is not None
        assert self.db.collection_name == "test_pubmed"

    def test_add_abstracts_valid(self):
        """Test adding abstracts with matching lengths"""
        abstracts = [
            "TITLE: Paper 1\nABSTRACT: This is about bone scaffolds",
            "TITLE: Paper 2\nABSTRACT: This is about biocompatibility"
        ]
        metadatas = [
            {"year": "2020", "link": "http://example.com/1"},
            {"year": "2021", "link": "http://example.com/2"}
        ]
        ids = ["pmid_1", "pmid_2"]

        if self.db.enabled:
            result = self.db.add_abstracts(abstracts, metadatas, ids)
            assert result is True

    def test_add_abstracts_mismatched_lengths(self):
        """Test that mismatched lengths raise an error"""
        abstracts = ["Abstract 1", "Abstract 2"]
        metadatas = [{"year": "2020"}]  # Only 1, but 2 abstracts
        ids = ["id_1", "id_2"]

        if self.db.enabled:
            with pytest.raises(ValueError):
                self.db.add_abstracts(abstracts, metadatas, ids)

    def test_query_db_returns_dict(self):
        """Test that query_db returns expected dict structure"""
        result = self.db.query_db("bone scaffolds", n_results=5)
        
        assert isinstance(result, dict)
        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert isinstance(result["ids"], list)
        assert isinstance(result["documents"], list)
        assert isinstance(result["metadatas"], list)

    def test_query_db_with_data(self):
        """Test querying after adding data"""
        if not self.db.enabled:
            pytest.skip("ChromaDB not enabled")

        # Add test data
        abstracts = [
            "TITLE: Bone Scaffolds\nABSTRACT: Study on biocompatible bone scaffolds for orthopedic applications",
            "TITLE: Dental Implants\nABSTRACT: Analysis of titanium implants and osseointegration"
        ]
        metadatas = [
            {"year": "2020", "link": "http://example.com/1"},
            {"year": "2021", "link": "http://example.com/2"}
        ]
        ids = ["pmid_1", "pmid_2"]

        # Add to DB
        self.db.add_abstracts(abstracts, metadatas, ids)

        # Query
        result = self.db.query_db("bone implants", n_results=2)

        # Verify we got results
        assert len(result["ids"]) > 0
        assert len(result["documents"]) > 0
        assert len(result["metadatas"]) > 0

    def test_disabled_db_fallback(self):
        """Test that disabled DB returns empty results gracefully"""
        db = ResearchDB(persist_directory="test_chroma_db_disabled")
        db.enabled = False  # Force disable

        # Should return empty, not crash
        result = db.query_db("test query")
        assert result["ids"] == []
        assert result["documents"] == []
        assert result["metadatas"] == []

        # add_abstracts should return False
        add_result = db.add_abstracts(["test"], [{"year": "2020"}], ["id_1"])
        assert add_result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
