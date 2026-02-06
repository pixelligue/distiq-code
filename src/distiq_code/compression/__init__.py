"""Prompt compression module (requires `distiq-code[compression]`)."""

__all__ = ["PromptCompressor"]


def __getattr__(name: str):
    if name == "PromptCompressor":
        from distiq_code.compression.compressor import PromptCompressor
        return PromptCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
