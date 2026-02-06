"""Prompt compression module (requires `distiq-code[compression]`)."""

__all__ = ["PromptCompressor", "compress_tools", "decompress_tool_use"]


def __getattr__(name: str):
    if name == "PromptCompressor":
        from distiq_code.compression.compressor import PromptCompressor
        return PromptCompressor
    if name == "compress_tools":
        from distiq_code.compression.tool_compressor import compress_tools
        return compress_tools
    if name == "decompress_tool_use":
        from distiq_code.compression.tool_compressor import decompress_tool_use
        return decompress_tool_use
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
