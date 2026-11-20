# Excerpt from PolicyLens (private repo): apps/backend/adapters/factory.py — published unchanged.
"""Adapter factory — selects concrete implementations from configuration.

This is the only place in application code that references the LLM_BACKEND,
EMBEDDING_BACKEND, and STORAGE_BACKEND settings. All other code imports the
abstract interfaces (LLMAdapter, EmbeddingAdapter, StorageAdapter) and
receives concrete instances through dependency injection.

Usage:
    from adapters.factory import get_llm, get_embeddings, get_storage
    llm = get_llm()
    embeddings = get_embeddings()
    storage = get_storage()
"""

from functools import lru_cache

from adapters.base import EmbeddingAdapter, LLMAdapter, StorageAdapter
from config import settings


@lru_cache(maxsize=1)
def get_llm() -> LLMAdapter:
    """Return the configured LLM adapter (singleton)."""
    backend = settings.llm_backend
    if backend == "openai":
        from adapters.llm_openai import OpenAILLMAdapter
        return OpenAILLMAdapter()
    if backend == "anthropic":
        from adapters.llm_anthropic import AnthropicLLMAdapter
        return AnthropicLLMAdapter()
    if backend == "ollama":
        from adapters.llm_ollama import OllamaLLMAdapter
        return OllamaLLMAdapter()
    raise ValueError(f"Unknown LLM backend: {backend!r}. Expected: openai | anthropic | ollama")


@lru_cache(maxsize=1)
def get_embeddings() -> EmbeddingAdapter:
    """Return the configured embedding adapter (singleton)."""
    backend = settings.embedding_backend
    if backend == "openai":
        from adapters.embeddings_openai import OpenAIEmbeddingAdapter
        return OpenAIEmbeddingAdapter()
    if backend == "ollama":
        from adapters.embeddings_ollama import OllamaEmbeddingAdapter
        return OllamaEmbeddingAdapter()
    raise ValueError(f"Unknown embedding backend: {backend!r}. Expected: openai | ollama")


@lru_cache(maxsize=1)
def get_storage() -> StorageAdapter:
    """Return the configured storage adapter (singleton)."""
    backend = settings.storage_backend
    if backend == "local":
        from adapters.storage_local import LocalStorageAdapter
        return LocalStorageAdapter()
    if backend == "s3":
        from adapters.storage_s3 import S3StorageAdapter
        return S3StorageAdapter()
    if backend == "azure":
        from adapters.storage_azure import AzureStorageAdapter
        return AzureStorageAdapter()
    raise ValueError(f"Unknown storage backend: {backend!r}. Expected: local | s3 | azure")
