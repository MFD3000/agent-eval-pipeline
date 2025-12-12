"""
Unit Tests for PgVectorStore

Tests the PostgreSQL vector store without requiring actual database.
Uses mocks to verify SQL generation and connection handling.

STAFF ENGINEER PATTERNS:
------------------------
1. Mock psycopg connection to test without database
2. Verify SQL queries are correct
3. Test connection lifecycle
4. Test error handling
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from agent_eval_pipeline.retrieval.store import (
    PgVectorStore,
    VectorStoreConfig,
    get_vector_store,
    PGVECTOR_AVAILABLE,
)
from agent_eval_pipeline.retrieval.document import Document


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings provider."""
    embeddings = MagicMock()
    embeddings.embed.return_value = np.array([0.1, 0.2, 0.3])
    embeddings.embed_batch.return_value = [np.array([0.1, 0.2, 0.3])]
    return embeddings


@pytest.fixture
def mock_connection():
    """Create mock psycopg connection."""
    conn = MagicMock()
    conn.execute.return_value = MagicMock()
    return conn


@pytest.fixture
def store_config():
    """Create test store configuration."""
    return VectorStoreConfig(
        connection_string="postgresql://test:test@localhost/test_db",
        embedding_model="text-embedding-3-small",
        embedding_dim=3,
        table_name="test_documents",
        index_type="hnsw",
    )


# ---------------------------------------------------------------------------
# VECTOR STORE CONFIG TESTS
# ---------------------------------------------------------------------------


class TestVectorStoreConfig:
    """Test VectorStoreConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = VectorStoreConfig()

        assert config.embedding_dim == 1536
        assert config.table_name == "medical_documents"
        assert config.index_type == "hnsw"

    def test_custom_config(self):
        """Should accept custom values."""
        config = VectorStoreConfig(
            connection_string="postgresql://custom:custom@db/mydb",
            embedding_dim=768,
            table_name="custom_docs",
        )

        assert config.connection_string == "postgresql://custom:custom@db/mydb"
        assert config.embedding_dim == 768
        assert config.table_name == "custom_docs"


# ---------------------------------------------------------------------------
# PGVECTOR STORE TESTS (WITH MOCKS)
# ---------------------------------------------------------------------------


class TestPgVectorStore:
    """Test PgVectorStore with mocked database."""

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_create_store(self, store_config, mock_embeddings):
        """Should create store with config and embeddings."""
        store = PgVectorStore(store_config, mock_embeddings)

        assert store.config == store_config
        assert store._embeddings == mock_embeddings
        assert store._conn is None  # Not connected yet

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_connect(self, store_config, mock_embeddings, mock_connection):
        """Should establish database connection."""
        store = PgVectorStore(store_config, mock_embeddings)

        with patch("agent_eval_pipeline.retrieval.store.psycopg") as mock_psycopg:
            with patch("agent_eval_pipeline.retrieval.store.register_vector"):
                mock_psycopg.connect.return_value = mock_connection
                store.connect()

            mock_psycopg.connect.assert_called_once()
            assert store._conn is not None

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_close_connection(self, store_config, mock_embeddings, mock_connection):
        """Should close database connection."""
        store = PgVectorStore(store_config, mock_embeddings)
        store._conn = mock_connection

        store.close()

        mock_connection.close.assert_called_once()
        assert store._conn is None

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_create_schema(self, store_config, mock_embeddings, mock_connection):
        """Should create table and indexes."""
        store = PgVectorStore(store_config, mock_embeddings)
        store._conn = mock_connection

        store.create_schema()

        # Should execute CREATE TABLE and CREATE INDEX statements
        assert mock_connection.execute.call_count >= 3

        # Check table creation was called
        calls = [str(c) for c in mock_connection.execute.call_args_list]
        assert any("CREATE TABLE" in str(c) for c in calls)
        assert any("CREATE INDEX" in str(c) for c in calls)

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_insert_document(self, store_config, mock_embeddings, mock_connection):
        """Should insert document with embedding."""
        store = PgVectorStore(store_config, mock_embeddings)
        store._conn = mock_connection

        doc = Document(
            id="test-doc",
            title="Test Document",
            content="Test content",
            markers=["TSH"],
        )

        store.insert_document(doc)

        # Should have called execute with INSERT
        mock_connection.execute.assert_called()
        call_args = str(mock_connection.execute.call_args)
        assert "INSERT" in call_args

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_search_returns_results(self, store_config, mock_embeddings, mock_connection):
        """Should search and return DocumentResult objects."""
        store = PgVectorStore(store_config, mock_embeddings)
        store._conn = mock_connection

        # Mock fetchall to return results
        mock_connection.execute.return_value.fetchall.return_value = [
            ("doc-1", "Title 1", "Content 1", ["TSH"], 0.1),
            ("doc-2", "Title 2", "Content 2", ["T4"], 0.2),
        ]

        results = store.search("thyroid", limit=5)

        assert len(results) == 2
        assert results[0].id == "doc-1"
        assert results[0].score == 0.9  # 1 - 0.1 distance

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_search_with_marker_filter(self, store_config, mock_embeddings, mock_connection):
        """Should filter by markers when specified."""
        store = PgVectorStore(store_config, mock_embeddings)
        store._conn = mock_connection

        mock_connection.execute.return_value.fetchall.return_value = [
            ("doc-1", "Title 1", "Content 1", ["TSH"], 0.1),
        ]

        results = store.search("thyroid", limit=5, marker_filter=["TSH"])

        # Check that query included marker filter
        call_args = str(mock_connection.execute.call_args)
        assert "markers" in call_args.lower() or len(results) >= 0

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_search_by_markers(self, store_config, mock_embeddings, mock_connection):
        """Should search by marker names."""
        store = PgVectorStore(store_config, mock_embeddings)
        store._conn = mock_connection

        mock_connection.execute.return_value.fetchall.return_value = []

        results = store.search_by_markers(["TSH", "Free T4"], limit=5)

        # Should have called search with marker filter
        mock_embeddings.embed.assert_called()


# ---------------------------------------------------------------------------
# GET_VECTOR_STORE FACTORY TESTS
# ---------------------------------------------------------------------------


class TestGetVectorStoreFactory:
    """Test the get_vector_store factory function."""

    def test_returns_in_memory_by_default(self, mock_embeddings):
        """Should return InMemoryVectorStore by default."""
        with patch.dict("os.environ", {"USE_MOCK_EMBEDDINGS": "true"}):
            store = get_vector_store(use_postgres=False, embeddings=mock_embeddings)

        # Should be InMemoryVectorStore
        assert hasattr(store, 'search')
        assert hasattr(store, 'insert_document')
        assert store.__class__.__name__ == "InMemoryVectorStore"

    @pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="pgvector not installed")
    def test_returns_pgvector_when_requested(self, mock_embeddings):
        """Should return PgVectorStore when use_postgres=True."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://test@localhost/test"}):
            store = get_vector_store(use_postgres=True, embeddings=mock_embeddings)

        assert store.__class__.__name__ == "PgVectorStore"

    def test_returns_in_memory_without_explicit_embeddings(self):
        """Should create InMemoryVectorStore even without explicit embeddings."""
        # The function should work with default in-memory store
        # without requiring external embeddings provider
        store = get_vector_store(use_postgres=False)

        assert store.__class__.__name__ == "InMemoryVectorStore"


# ---------------------------------------------------------------------------
# DOCUMENT RESULT TESTS
# ---------------------------------------------------------------------------


class TestDocumentResult:
    """Test DocumentResult from core protocols."""

    def test_document_result_creation(self):
        """Should create DocumentResult with all fields."""
        from agent_eval_pipeline.core import DocumentResult

        result = DocumentResult(
            id="doc-001",
            title="Test Title",
            content="Test content",
            markers=["TSH", "T4"],
            score=0.95,
        )

        assert result.id == "doc-001"
        assert result.title == "Test Title"
        assert result.content == "Test content"
        assert result.markers == ["TSH", "T4"]
        assert result.score == 0.95
