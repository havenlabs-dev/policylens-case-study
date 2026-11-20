# Excerpt from PolicyLens (private repo): apps/backend/adapters/base.py — published unchanged.
"""Abstract adapter interfaces for PolicyLens's pluggable service layer.

All external service calls (LLM, embeddings, storage) go through these
interfaces. Concrete implementations live in this package; application code
imports the interface only — never the concrete class directly.

This design ensures that swapping providers (e.g., OpenAI → Ollama for the
Offline tier) requires only a configuration change, not a code change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMAdapter(ABC):
    """Complete a prompt and return the model's response."""

    @abstractmethod
    async def complete(self, system: str, user: str) -> LLMResponse:
        """Send a system + user message and return the model's reply."""
        ...


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingAdapter(ABC):
    """Convert text into embedding vectors."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a float vector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns one vector per input."""
        ...


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class StorageAdapter(ABC):
    """Read and write binary objects (PDF files) using opaque storage keys.

    Callers are responsible for encoding tenant_id into the key to enforce
    isolation (e.g. "tenants/{tenant_id}/documents/{doc_id}/{filename}").
    This matches how S3 and Azure Blob Storage work natively.
    """

    @abstractmethod
    async def write(self, key: str, data: bytes) -> None:
        """Write data to storage at the given key."""
        ...

    @abstractmethod
    async def read(self, key: str) -> bytes:
        """Read data from storage. Raises FileNotFoundError if absent."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a stored object. No-op if it doesn't exist."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if the object exists in storage."""
        ...

    @abstractmethod
    async def delete_prefix(self, prefix: str) -> None:
        """Delete all objects whose key starts with prefix. No-op if none match."""
        ...
