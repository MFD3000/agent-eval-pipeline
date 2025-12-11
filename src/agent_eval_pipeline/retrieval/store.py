"""
Vector store implementations following the gold standard pattern.

Pattern: Protocol → Production impl → Test double → Factory

This module contains:
1. VectorStoreConfig - Configuration dataclass
2. PgVectorStore - PostgreSQL with pgvector (production)
3. InMemoryVectorStore - In-memory store (testing/development)
4. get_vector_store() - Factory function

INTERVIEW TALKING POINT:
------------------------
"The retrieval layer follows hexagonal architecture. VectorStore is a Protocol
that defines the contract. PgVectorStore implements it for production with
PostgreSQL. InMemoryVectorStore implements it for testing - no database needed.
The factory function decides which to use based on configuration. This means
I can test retrieval logic in under 1ms without touching a database."
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from agent_eval_pipeline.core import DocumentResult, EmbeddingProvider
from agent_eval_pipeline.retrieval.document import Document

if TYPE_CHECKING:
    pass

# Optional: Only import psycopg if available (for local dev without postgres)
try:
    import psycopg
    from pgvector.psycopg import register_vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""

    connection_string: str = "postgresql://localhost/eval_pipeline"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    table_name: str = "medical_documents"
    index_type: str = "hnsw"  # or "ivfflat"


# ---------------------------------------------------------------------------
# PGVECTOR STORE (Production)
# ---------------------------------------------------------------------------


class PgVectorStore:
    """
    PostgreSQL vector store using pgvector.

    Dependencies are INJECTED, not created internally.
    This enables testing with mock embeddings.

    WHY PGVECTOR:
    - POSTGRES: Battle-tested, ACID compliant, your team already knows it
    - HYBRID SEARCH: Combine vector similarity with traditional SQL filters
    - HNSW INDEX: Fast approximate nearest neighbor search
    - NO VENDOR LOCK: Open source, runs anywhere
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        embeddings: EmbeddingProvider,
    ):
        """
        Initialize with injected dependencies.

        Args:
            config: Store configuration
            embeddings: Embedding provider (injected, not created here)
        """
        self.config = config
        self._embeddings = embeddings
        self._conn = None

    def connect(self) -> None:
        """Establish database connection."""
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector not available. Install with: pip install pgvector psycopg[binary]"
            )

        self._conn = psycopg.connect(self.config.connection_string, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def create_schema(self) -> None:
        """Create the documents table and indexes."""
        if not self._conn:
            self.connect()

        # Create table
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.config.table_name} (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                markers TEXT[],
                embedding vector({self.config.embedding_dim})
            )
        """
        )

        # Create HNSW index for fast similarity search
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.config.table_name}_embedding_idx
            ON {self.config.table_name}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
        )

        # Create GIN index for marker filtering
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {self.config.table_name}_markers_idx
            ON {self.config.table_name}
            USING GIN (markers)
        """
        )

    def insert_document(self, doc: Document) -> None:
        """Insert a document with its embedding."""
        if not self._conn:
            self.connect()

        # Generate embedding if not provided
        if doc.embedding is None:
            doc.embedding = self._embeddings.embed(f"{doc.title}\n{doc.content}")

        self._conn.execute(
            f"""
            INSERT INTO {self.config.table_name} (id, title, content, markers, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                markers = EXCLUDED.markers,
                embedding = EXCLUDED.embedding
            """,
            (doc.id, doc.title, doc.content, doc.markers, doc.embedding),
        )

    def insert_documents_batch(self, docs: list[Document]) -> None:
        """Batch insert documents."""
        # Generate embeddings for docs without them
        texts_to_embed = []
        indices_to_embed = []

        for i, doc in enumerate(docs):
            if doc.embedding is None:
                texts_to_embed.append(f"{doc.title}\n{doc.content}")
                indices_to_embed.append(i)

        if texts_to_embed:
            embeddings = self._embeddings.embed_batch(texts_to_embed)
            for idx, emb in zip(indices_to_embed, embeddings):
                docs[idx].embedding = emb

        # Insert all
        for doc in docs:
            self.insert_document(doc)

    def search(
        self,
        query: str,
        limit: int = 5,
        marker_filter: list[str] | None = None,
    ) -> list[DocumentResult]:
        """Search for similar documents."""
        if not self._conn:
            self.connect()

        # Generate query embedding
        query_embedding = self._embeddings.embed(query)

        # Build query with optional filter
        if marker_filter:
            results = self._conn.execute(
                f"""
                SELECT id, title, content, markers,
                       embedding <=> %s AS distance
                FROM {self.config.table_name}
                WHERE markers && %s
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, marker_filter, limit),
            ).fetchall()
        else:
            results = self._conn.execute(
                f"""
                SELECT id, title, content, markers,
                       embedding <=> %s AS distance
                FROM {self.config.table_name}
                ORDER BY distance
                LIMIT %s
                """,
                (query_embedding, limit),
            ).fetchall()

        # Convert to DocumentResult objects
        return [
            DocumentResult(
                id=row[0],
                title=row[1],
                content=row[2],
                markers=row[3] or [],
                score=1 - row[4],  # Convert distance to similarity
            )
            for row in results
        ]

    def search_by_markers(
        self,
        markers: list[str],
        limit: int = 5,
    ) -> list[DocumentResult]:
        """Search for documents related to specific lab markers."""
        query = f"Medical information about: {', '.join(markers)}"
        return self.search(query, limit=limit, marker_filter=markers)


# ---------------------------------------------------------------------------
# IN-MEMORY STORE (Testing/Development)
# ---------------------------------------------------------------------------


class InMemoryVectorStore:
    """
    In-memory vector store for development/testing.

    Implements the same interface as PgVectorStore but doesn't require Postgres.
    Uses cosine similarity for searching.
    """

    def __init__(self, embeddings: EmbeddingProvider):
        """
        Initialize with injected embedding provider.

        Args:
            embeddings: Embedding provider for generating vectors
        """
        self._embeddings = embeddings
        self._documents: dict[str, Document] = {}

    def connect(self) -> None:
        """No-op for in-memory store."""
        pass

    def close(self) -> None:
        """No-op for in-memory store."""
        pass

    def create_schema(self) -> None:
        """No-op for in-memory store."""
        pass

    def insert_document(self, doc: Document) -> None:
        """Insert document into memory."""
        if doc.embedding is None:
            doc.embedding = self._embeddings.embed(f"{doc.title}\n{doc.content}")
        self._documents[doc.id] = doc

    def insert_documents_batch(self, docs: list[Document]) -> None:
        """Batch insert."""
        # Generate embeddings in batch for efficiency
        texts_to_embed = []
        indices_to_embed = []

        for i, doc in enumerate(docs):
            if doc.embedding is None:
                texts_to_embed.append(f"{doc.title}\n{doc.content}")
                indices_to_embed.append(i)

        if texts_to_embed:
            embeddings = self._embeddings.embed_batch(texts_to_embed)
            for idx, emb in zip(indices_to_embed, embeddings):
                docs[idx].embedding = emb

        for doc in docs:
            self._documents[doc.id] = doc

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(
        self,
        query: str,
        limit: int = 5,
        marker_filter: list[str] | None = None,
    ) -> list[DocumentResult]:
        """Search using cosine similarity."""
        query_emb = self._embeddings.embed(query)

        scored = []
        for doc in self._documents.values():
            # Apply marker filter if specified
            if marker_filter:
                if not any(m in doc.markers for m in marker_filter):
                    continue

            if doc.embedding is not None:
                score = self._cosine_similarity(query_emb, doc.embedding)
                scored.append((doc, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        return [
            DocumentResult(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                markers=doc.markers,
                score=score,
            )
            for doc, score in scored[:limit]
        ]

    def search_by_markers(
        self,
        markers: list[str],
        limit: int = 5,
    ) -> list[DocumentResult]:
        """Search by markers."""
        query = f"Medical information about: {', '.join(markers)}"
        return self.search(query, limit=limit, marker_filter=markers)


# ---------------------------------------------------------------------------
# FACTORY FUNCTION
# ---------------------------------------------------------------------------


def get_vector_store(
    use_postgres: bool = False,
    embeddings: EmbeddingProvider | None = None,
    config: VectorStoreConfig | None = None,
) -> PgVectorStore | InMemoryVectorStore:
    """
    Factory function to get the appropriate vector store.

    Follows the gold standard pattern from embeddings module.

    Args:
        use_postgres: Use PostgreSQL store (default: False for dev)
        embeddings: Embedding provider (will create one if not provided)
        config: Store configuration (uses defaults if not provided)

    Returns:
        VectorStore implementation
    """
    # Get or create embeddings
    if embeddings is None:
        from agent_eval_pipeline.embeddings import get_embedding_provider

        use_mock = os.environ.get("USE_MOCK_EMBEDDINGS", "true").lower() == "true"
        embeddings = get_embedding_provider(use_mock=use_mock and not use_postgres)

    # Create appropriate store
    if use_postgres and PGVECTOR_AVAILABLE:
        config = config or VectorStoreConfig(
            connection_string=os.environ.get(
                "DATABASE_URL", "postgresql://localhost/eval_pipeline"
            )
        )
        return PgVectorStore(config, embeddings)
    else:
        return InMemoryVectorStore(embeddings)
