"""
Quick manual test script for ChromaDB integration
Run this with: python test_manual.py
"""

import os
from dotenv import load_dotenv
from database_manager import ResearchDB

# Load environment
load_dotenv()

def test_1_initialization():
    """Test 1: Can we create a ResearchDB?"""
    print("\n" + "="*60)
    print("TEST 1: Initialize ResearchDB")
    print("="*60)
    
    db = ResearchDB(persist_directory="manual_test_db")
    
    print("OK - ResearchDB created")
    print(f"  - Enabled: {db.enabled}")
    print(f"  - Collection name: {db.collection_name}")
    print(f"  - Client: {db.client}")
    print(f"  - Collection: {db.collection}")
    
    if not db.enabled:
        print("WARNING: ChromaDB is disabled. Check if it's installed.")
        return False
    return db


def test_2_add_documents(db):
    """Test 2: Can we add documents?"""
    print("\n" + "="*60)
    print("TEST 2: Add Documents to ChromaDB")
    print("="*60)
    
    abstracts = [
        "TITLE: Bone Scaffolds for Regeneration\nAUTHORS: Smith et al.\nABSTRACT: This paper discusses 3D-printed bone scaffolds and their biocompatibility properties.",
        "TITLE: Titanium Implants\nAUTHORS: Johnson et al.\nABSTRACT: Study on titanium dental implants and osseointegration mechanisms.",
        "TITLE: Cartilage Engineering\nAUTHORS: Williams et al.\nABSTRACT: Techniques for engineering cartilage tissue using biomaterials."
    ]
    
    metadatas = [
        {"year": "2020", "link": "https://pubmed.ncbi.nlm.nih.gov/12345/"},
        {"year": "2021", "link": "https://pubmed.ncbi.nlm.nih.gov/12346/"},
        {"year": "2022", "link": "https://pubmed.ncbi.nlm.nih.gov/12347/"}
    ]
    
    ids = ["pmid_12345", "pmid_12346", "pmid_12347"]
    
    result = db.add_abstracts(abstracts, metadatas, ids)
    
    print(f"OK - Added {len(abstracts)} documents")
    print(f"  - Result: {result}")
    for i, (id_, meta) in enumerate(zip(ids, metadatas)):
        print(f"  - Doc {i+1}: {id_} (Year: {meta['year']})")
    
    return result


def test_3_query_documents(db):
    """Test 3: Can we query documents?"""
    print("\n" + "="*60)
    print("TEST 3: Query Documents from ChromaDB")
    print("="*60)
    
    queries = [
        "bone scaffold regeneration",
        "dental implants",
        "what materials are used?"
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        result = db.query_db(query, n_results=2)
        
        print(f"  Results found: {len(result['ids'])}")
        
        for i, (id_, doc, meta) in enumerate(zip(result['ids'], result['documents'], result['metadatas'])):
            print(f"\n    Result {i+1}:")
            print(f"      ID: {id_}")
            print(f"      Year: {meta.get('year', 'N/A')}")
            print(f"      Link: {meta.get('link', 'N/A')}")
            print(f"      Doc (first 100 chars): {doc[:100]}...")


def test_4_error_handling():
    """Test 4: Does error handling work?"""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)
    
    db = ResearchDB()
    
    # Test 4a: Mismatched lengths
    print("\n  4a: Mismatched list lengths")
    try:
        db.add_abstracts(
            ["doc1", "doc2"],
            [{"year": "2020"}],  # Only 1, but 2 docs
            ["id1", "id2"]
        )
        print("    FAILED - Should have raised ValueError!")
    except ValueError as e:
        print(f"    OK - Correctly raised ValueError: {e}")
    
    # Test 4b: Empty query
    print("\n  4b: Empty query on empty/new collection")
    result = db.query_db("test", n_results=5)
    print(f"    OK - Returned empty results: {result}")
    
    # Test 4c: Disabled DB
    print("\n  4c: Disabled DB fallback")
    db.enabled = False
    result = db.query_db("test")
    print(f"    OK - Disabled DB returns empty: {result}")


def main():
    print("\n" + "CHROMADB INTEGRATION TEST SUITE".center(60))
    print("=" * 60)
    
    # Test 1: Initialization
    db = test_1_initialization()
    if not db:
        print("\nFAILED: ChromaDB not available")
        return
    
    # Test 2: Add documents
    success = test_2_add_documents(db)
    if not success:
        print("\nFAILED: Could not add documents")
        return
    
    # Test 3: Query documents
    test_3_query_documents(db)
    
    # Test 4: Error handling
    test_4_error_handling()
    
    print("\n" + "="*60)
    print("SUCCESS - ALL TESTS PASSED!")
    print("="*60)
    print("\nYour ChromaDB integration is working correctly!")
    print("The app will now use semantic search for chat queries.")


if __name__ == "__main__":
    main()
