"""
Unit Tests for Retrieval Store

Tests the vector store protocol and InMemoryVectorStore behavior.

STAFF ENGINEER PATTERNS:
------------------------
1. Test through the protocol interface
2. Mock embeddings to avoid API calls
3. Verify search behavior and ranking
"""

import pytest
from unittest.mock import MagicMock
import numpy as np

from agent_eval_pipeline.retrieval.store import InMemoryVectorStore, get_vector_store
from agent_eval_pipeline.retrieval.document import Document


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings provider."""
    embeddings = MagicMock()

    # embed() takes a single text string and returns a single numpy array
    def mock_embed(text):
        # Simple mock: different vectors based on text content
        if "thyroid" in text.lower() or "tsh" in text.lower():
            return np.array([1.0, 0.0, 0.0])
        elif "cholesterol" in text.lower():
            return np.array([0.0, 1.0, 0.0])
        else:
            return np.array([0.0, 0.0, 1.0])

    # embed_batch() takes a list of texts and returns a list of numpy arrays
    def mock_embed_batch(texts):
        return [mock_embed(text) for text in texts]

    embeddings.embed.side_effect = mock_embed
    embeddings.embed_batch.side_effect = mock_embed_batch
    return embeddings


@pytest.fixture
def store_with_docs(mock_embeddings):
    """Create a store with test documents."""
    store = InMemoryVectorStore(mock_embeddings)

    # Add test documents using insert_document (the actual API)
    store.insert_document(Document(
        id="thyroid-001",
        title="TSH Overview",
        content="TSH controls thyroid function. Normal range is 0.4-4.0 mIU/L.",
        markers=["TSH"],
    ))
    store.insert_document(Document(
        id="thyroid-002",
        title="Free T4 Overview",
        content="Free T4 is the active thyroid hormone.",
        markers=["Free T4"],
    ))
    store.insert_document(Document(
        id="cholesterol-001",
        title="LDL Overview",
        content="LDL cholesterol affects heart health.",
        markers=["LDL"],
    ))

    return store


# ---------------------------------------------------------------------------
# BASIC OPERATIONS
# ---------------------------------------------------------------------------


class TestInMemoryVectorStore:
    """Test InMemoryVectorStore basic operations."""

    def test_create_store_with_embeddings(self, mock_embeddings):
        """Should create store with embeddings provider."""
        store = InMemoryVectorStore(mock_embeddings)
        assert store is not None

    def test_insert_document(self, mock_embeddings):
        """Should insert document into store."""
        store = InMemoryVectorStore(mock_embeddings)

        doc = Document(
            id="test-001",
            title="Test Doc",
            content="Test content about thyroid",
            markers=["TSH"],
        )

        store.insert_document(doc)

        # Should be searchable
        results = store.search("thyroid", limit=5)
        assert len(results) > 0

    def test_search_returns_results(self, store_with_docs):
        """Search should return matching documents."""
        results = store_with_docs.search("thyroid TSH", limit=5)

        assert len(results) > 0
        # Results should have expected attributes
        assert hasattr(results[0], 'id')
        assert hasattr(results[0], 'content')

    def test_search_respects_limit(self, store_with_docs):
        """Search should respect limit parameter."""
        results = store_with_docs.search("health", limit=1)

        assert len(results) <= 1

    def test_search_empty_store(self, mock_embeddings):
        """Search on empty store should return empty list."""
        store = InMemoryVectorStore(mock_embeddings)

        results = store.search("anything", limit=5)

        assert results == []


# ---------------------------------------------------------------------------
# DOCUMENT MODEL
# ---------------------------------------------------------------------------


class TestDocument:
    """Test Document dataclass."""

    def test_document_creation_with_required_fields(self):
        """Document should be created with required fields."""
        doc = Document(
            id="test-id",
            title="Test Title",
            content="Test content",
            markers=["TSH"],
        )

        assert doc.id == "test-id"
        assert doc.title == "Test Title"
        assert doc.content == "Test content"
        assert doc.markers == ["TSH"]

    def test_document_default_fields(self):
        """Document should have default values for optional fields."""
        doc = Document(
            id="test-id",
            title="Test Title",
            content="Test content",
            markers=[],
        )

        assert doc.embedding is None
        assert doc.score is None

    def test_document_with_embedding(self):
        """Document should accept embedding array."""
        embedding = np.array([0.1, 0.2, 0.3])
        doc = Document(
            id="test-id",
            title="Test Title",
            content="Test content",
            markers=[],
            embedding=embedding,
        )

        assert doc.embedding is not None
        assert len(doc.embedding) == 3


# ---------------------------------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------------------------------


class TestGetVectorStore:
    """Test the get_vector_store factory function."""

    def test_get_vector_store_returns_store(self, mock_embeddings):
        """Factory should return a vector store instance."""
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("USE_POSTGRES", "false")
            mp.setenv("USE_MOCK_EMBEDDINGS", "true")

            # Factory uses use_postgres=bool, embeddings=provider
            store = get_vector_store(use_postgres=False, embeddings=mock_embeddings)

            assert store is not None
            assert hasattr(store, 'search')
            assert hasattr(store, 'insert_document')


# ---------------------------------------------------------------------------
# SEARCH RANKING
# ---------------------------------------------------------------------------


class TestSearchRanking:
    """Test that search results are properly ranked."""

    def test_relevant_docs_ranked_higher(self, store_with_docs):
        """Documents matching query should be ranked higher."""
        results = store_with_docs.search("TSH thyroid", limit=10)

        if len(results) >= 2:
            # Thyroid docs should be ranked before cholesterol
            thyroid_ids = [r.id for r in results if "thyroid" in r.id]
            cholesterol_ids = [r.id for r in results if "cholesterol" in r.id]

            if thyroid_ids and cholesterol_ids:
                # Get positions
                thyroid_positions = [
                    i for i, r in enumerate(results) if r.id in thyroid_ids
                ]
                cholesterol_positions = [
                    i for i, r in enumerate(results) if r.id in cholesterol_ids
                ]

                # At least one thyroid doc should be before cholesterol
                assert min(thyroid_positions) < max(cholesterol_positions)


# ---------------------------------------------------------------------------
# PROTOCOL COMPLIANCE
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    """Test that InMemoryVectorStore implements VectorStore protocol."""

    def test_has_search_method(self, mock_embeddings):
        """Store should have search method."""
        store = InMemoryVectorStore(mock_embeddings)

        assert hasattr(store, 'search')
        assert callable(store.search)

    def test_has_insert_document_method(self, mock_embeddings):
        """Store should have insert_document method."""
        store = InMemoryVectorStore(mock_embeddings)

        assert hasattr(store, 'insert_document')
        assert callable(store.insert_document)

    def test_search_returns_list(self, store_with_docs):
        """Search should return a list."""
        results = store_with_docs.search("test", limit=5)

        assert isinstance(results, list)
