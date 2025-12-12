"""
Document model for the retrieval system.

Single responsibility: Define the structure of documents
stored in vector stores.

"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Document:
    """
    A document with embedding for retrieval.

    This is the internal representation used by vector stores.
    For external APIs, we convert to DocumentResult 
    (defined in core.protocols).
    """
    id: str
    title: str
    content: str
    markers: list[str]  # Related lab markers for filtering
    embedding: np.ndarray | None = None
    score: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "markers": self.markers,
            "score": self.score,
        }
