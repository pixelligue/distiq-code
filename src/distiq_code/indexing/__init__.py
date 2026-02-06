"""
Code Indexing Module for Distiq Code

Provides intelligent code understanding through:
- Tree-sitter parsing for semantic chunking
- Code embeddings for similarity search
- FAISS vector index for fast retrieval
- Smart context building for LLM prompts

Usage:
    from distiq_code.indexing import Indexer, ContextBuilder
    
    # Index a project
    indexer = Indexer()
    await indexer.index_directory("./src")
    
    # Build context for a query
    builder = ContextBuilder(indexer)
    context = await builder.build("How does authentication work?")
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .indexer import Indexer
    from .context_builder import ContextBuilder
    from .embedder import CodeEmbedder

__all__ = [
    "Indexer",
    "ContextBuilder",
    "CodeEmbedder",
]


def get_indexer(project_dir: Path | str | None = None) -> "Indexer":
    """
    Get or create indexer for a project.
    
    Args:
        project_dir: Project directory (default: current directory)
        
    Returns:
        Indexer instance
    """
    from .indexer import Indexer
    
    if project_dir is None:
        project_dir = Path.cwd()
    elif isinstance(project_dir, str):
        project_dir = Path(project_dir)
    
    return Indexer(project_dir)


def get_context_builder(project_dir: Path | str | None = None) -> "ContextBuilder":
    """
    Get context builder for a project.
    
    Args:
        project_dir: Project directory
        
    Returns:
        ContextBuilder instance
    """
    from .context_builder import ContextBuilder
    
    indexer = get_indexer(project_dir)
    return ContextBuilder(indexer)
