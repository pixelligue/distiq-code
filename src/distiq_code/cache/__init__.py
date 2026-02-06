"""Semantic caching module (requires `distiq-code[ml]`)."""

__all__ = ["SemanticCache"]


def __getattr__(name: str):
    if name == "SemanticCache":
        from distiq_code.cache.semantic_cache import SemanticCache
        return SemanticCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
