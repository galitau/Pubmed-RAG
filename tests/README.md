# Test Suite for PubMed RAG

This directory contains unit tests for the ChromaDB integration (`ResearchDB` class).

## Running the Tests

### Prerequisites
Make sure you have pytest installed:
```bash
pip install pytest
```

### Run All Tests
From the project root directory:
```bash
pytest tests/test_database_manager.py -v
```

### Run a Specific Test
```bash
pytest tests/test_database_manager.py::TestResearchDB::test_initialization -v
```

### Run with Verbose Output
```bash
pytest tests/test_database_manager.py -v -s
```
The `-s` flag shows print statements.

---

## Test Details

### `test_initialization()`
**What it tests:** Verifies that ResearchDB initializes correctly with proper attributes.

**Checks:**
- ResearchDB object is created
- Collection name is set correctly to "test_pubmed"

---

### `test_add_abstracts_valid()`
**What it tests:** Verifies that documents can be added to ChromaDB with matching metadata and IDs.

**Checks:**
- 2 sample abstracts are added successfully
- Returns `True` on success
- Metadata (year, link) is stored correctly
- IDs are assigned properly

---

### `test_add_abstracts_mismatched_lengths()`
**What it tests:** Verifies error handling when list lengths don't match.

**Checks:**
- Raises `ValueError` when abstracts, metadatas, and ids have different lengths
- Error message is descriptive

---

### `test_query_db_returns_dict()`
**What it tests:** Verifies that query_db returns the correct data structure.

**Checks:**
- Result is a dictionary with keys: 'ids', 'documents', 'metadatas'
- Each value is a list (even if empty)

---

### `test_query_db_with_data()`
**What it tests:** Verifies semantic search works after adding data.

**Checks:**
- Documents are added to the collection
- Query returns results (if embeddings are working)
- Results contain ids, documents, and metadata

---

### `test_disabled_db_fallback()`
**What it tests:** Verifies graceful degradation when ChromaDB is disabled.

**Checks:**
- Disabled DB returns empty results instead of crashing
- `add_abstracts()` returns `False` when disabled
- `query_db()` returns empty lists when disabled

---

## What Each Test Does

| Test Name | Purpose | Status |
|-----------|---------|--------|
| `test_initialization` | Check ResearchDB setup | ✓ Unit test |
| `test_add_abstracts_valid` | Add documents to DB | ✓ Unit test |
| `test_add_abstracts_mismatched_lengths` | Error handling | ✓ Unit test |
| `test_query_db_returns_dict` | Check return structure | ✓ Unit test |
| `test_query_db_with_data` | Semantic search | ✓ Integration test |
| `test_disabled_db_fallback` | Graceful degradation | ✓ Unit test |

---

## Expected Output

```
platform win32 -- Python 3.14.0, pytest-9.0.3, pluggy-1.6.0
collected 6 items

tests/test_database_manager.py::TestResearchDB::test_initialization PASSED
tests/test_database_manager.py::TestResearchDB::test_add_abstracts_valid PASSED
tests/test_database_manager.py::TestResearchDB::test_add_abstracts_mismatched_lengths PASSED
tests/test_database_manager.py::TestResearchDB::test_query_db_returns_dict PASSED
tests/test_database_manager.py::TestResearchDB::test_query_db_with_data PASSED
tests/test_database_manager.py::TestResearchDB::test_disabled_db_fallback PASSED

======================== 6 passed in X.XXs ========================
```

---

## Troubleshooting

### ModuleNotFoundError: No module named 'database_manager'
- Make sure you're running pytest from the project root directory
- The test file adds the parent directory to sys.path automatically

### Import Error: No module named 'chromadb'
- Install chromadb: `pip install chromadb`
- Or install all dependencies: `pip install -r requirements.txt`

### Tests Skip or Fail
- Ensure `.env` file exists with `GEMINI_API_KEY` set
- Check that `chromadb` is properly installed in your virtual environment
- Delete `chroma_db/` and `manual_test_db/` directories to reset the test databases

---

## Integration Test (Manual)

There's also a manual integration test in the root directory:
```bash
python test_manual.py
```

This tests the full workflow:
- Initialize ResearchDB
- Add multiple documents
- Query the database
- Error handling

See [../test_manual.py](../test_manual.py) for details.
