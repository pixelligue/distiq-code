"""
Smart Context Builder

Builds optimal context for LLM prompts by:
1. Semantic search for relevant code
2. Adding dependencies and related code
3. Respecting token budgets
4. Prioritizing recent/important files

This is the key component for reducing token usage by 90%+
while maintaining high quality responses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class ContextChunk:
    """A chunk of context to include in the prompt."""
    
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str
    name: str
    score: float
    reason: str  # Why this chunk was included
    
    def to_prompt_format(self) -> str:
        """Format for inclusion in LLM prompt."""
        return f"""### {self.file_path} (lines {self.start_line}-{self.end_line})
```{self._get_language()}
{self.content}
```"""
    
    def _get_language(self) -> str:
        """Get language from file extension."""
        ext = Path(self.file_path).suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
        }.get(ext, "")


@dataclass
class BuiltContext:
    """The built context ready for LLM."""
    
    chunks: list[ContextChunk] = field(default_factory=list)
    total_tokens: int = 0
    query: str = ""
    
    def to_prompt(self) -> str:
        """Convert to prompt text."""
        if not self.chunks:
            return ""
        
        parts = ["## Relevant Code\n"]
        
        for chunk in self.chunks:
            parts.append(chunk.to_prompt_format())
            parts.append("")
        
        return "\n".join(parts)
    
    def __str__(self) -> str:
        return self.to_prompt()


class ContextBuilder:
    """
    Build optimal context for LLM prompts.
    
    Strategy:
    1. Search for chunks matching the query
    2. Add dependencies of top matches
    3. Add recently edited files (optional)
    4. Truncate to fit token budget
    5. Format for LLM consumption
    """
    
    # Default token budget for context
    DEFAULT_MAX_TOKENS = 8000
    
    # Approximate chars per token (for estimation)
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        indexer: Any,  # Indexer instance
        max_tokens: int | None = None,
    ):
        """
        Initialize context builder.
        
        Args:
            indexer: Indexer instance with search capability
            max_tokens: Maximum tokens for context
        """
        self.indexer = indexer
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        
        # Recent files boost
        self._recent_files: list[str] = []
    
    def build(
        self,
        query: str,
        k: int = 10,
        include_dependencies: bool = True,
        recent_files: list[str] | None = None,
    ) -> BuiltContext:
        """
        Build context for a query.
        
        Args:
            query: User's query/request
            k: Number of chunks to retrieve
            include_dependencies: Include related code
            recent_files: Recently edited files to boost
            
        Returns:
            Built context ready for LLM
        """
        context = BuiltContext(query=query)
        
        # 1. Search for relevant chunks
        search_results = self.indexer.search(query, k=k * 2, min_score=0.3)
        
        if not search_results:
            logger.debug(f"No search results for: {query[:50]}...")
            return context
        
        logger.debug(f"Found {len(search_results)} relevant chunks")
        
        # 2. Score and rank results
        ranked = self._rank_results(search_results, recent_files or [])
        
        # 3. Add chunks until budget is reached
        used_tokens = 0
        seen_files = set()
        
        for result in ranked:
            # Estimate tokens
            chunk_tokens = len(result["content"]) // self.CHARS_PER_TOKEN
            
            # Check budget
            if used_tokens + chunk_tokens > self.max_tokens:
                # Try to fit a truncated version
                remaining_tokens = self.max_tokens - used_tokens
                if remaining_tokens > 200:  # Worth including
                    truncated_content = result["content"][:remaining_tokens * self.CHARS_PER_TOKEN]
                    result["content"] = truncated_content + "\n# ... (truncated)"
                    chunk_tokens = remaining_tokens
                else:
                    continue
            
            # Create context chunk
            chunk = ContextChunk(
                file_path=result["file_path"],
                start_line=result["start_line"],
                end_line=result["end_line"],
                content=result["content"],
                chunk_type=result["chunk_type"],
                name=result["name"],
                score=result["score"],
                reason=self._get_inclusion_reason(result),
            )
            
            context.chunks.append(chunk)
            used_tokens += chunk_tokens
            seen_files.add(result["file_path"])
            
            if used_tokens >= self.max_tokens:
                break
        
        # 4. Optionally add dependencies
        if include_dependencies and used_tokens < self.max_tokens * 0.8:
            deps = self._find_dependencies(context.chunks)
            
            for dep in deps:
                if dep["file_path"] in seen_files:
                    continue
                
                chunk_tokens = len(dep["content"]) // self.CHARS_PER_TOKEN
                if used_tokens + chunk_tokens > self.max_tokens:
                    continue
                
                chunk = ContextChunk(
                    file_path=dep["file_path"],
                    start_line=dep["start_line"],
                    end_line=dep["end_line"],
                    content=dep["content"],
                    chunk_type=dep["chunk_type"],
                    name=dep["name"],
                    score=0.5,  # Lower score for dependencies
                    reason="dependency",
                )
                
                context.chunks.append(chunk)
                used_tokens += chunk_tokens
        
        context.total_tokens = used_tokens
        
        logger.info(
            f"Built context: {len(context.chunks)} chunks, ~{used_tokens} tokens"
        )
        
        return context
    
    def _rank_results(
        self,
        results: list[dict],
        recent_files: list[str],
    ) -> list[dict]:
        """Rank search results with boosting."""
        for result in results:
            boost = 0.0
            
            # Boost recent files
            if result["file_path"] in recent_files:
                boost += 0.2
            
            # Boost classes (more context)
            if result["chunk_type"] == "class":
                boost += 0.1
            
            # Boost items with docstrings (better documented)
            if result.get("docstring"):
                boost += 0.05
            
            result["adjusted_score"] = result["score"] + boost
        
        # Sort by adjusted score
        return sorted(results, key=lambda x: x["adjusted_score"], reverse=True)
    
    def _find_dependencies(
        self,
        chunks: list[ContextChunk],
    ) -> list[dict]:
        """Find dependencies of the selected chunks."""
        dependencies = []
        
        # Extract imported names from chunks
        imported_names = set()
        for chunk in chunks:
            # Simple heuristic: look for import statements
            content = chunk.content
            
            # Python imports
            if "import " in content:
                import re
                # from X import Y
                for match in re.finditer(r'from\s+([\w.]+)\s+import\s+([\w,\s]+)', content):
                    imported_names.update(match.group(2).replace(" ", "").split(","))
                # import X
                for match in re.finditer(r'^import\s+([\w.]+)', content, re.MULTILINE):
                    imported_names.add(match.group(1).split(".")[-1])
        
        # Search for imported names
        for name in imported_names:
            if len(name) < 3:  # Skip short names
                continue
            
            results = self.indexer.search(name, k=2, min_score=0.7)
            dependencies.extend(results)
        
        # Deduplicate
        seen = set()
        unique_deps = []
        for dep in dependencies:
            if dep["chunk_id"] not in seen:
                seen.add(dep["chunk_id"])
                unique_deps.append(dep)
        
        return unique_deps[:5]  # Limit to top 5 dependencies
    
    def _get_inclusion_reason(self, result: dict) -> str:
        """Get reason why chunk was included."""
        score = result["score"]
        if score > 0.8:
            return "highly_relevant"
        elif score > 0.6:
            return "relevant"
        elif score > 0.4:
            return "possibly_relevant"
        else:
            return "contextual"
    
    def set_recent_files(self, files: list[str]) -> None:
        """Set list of recently edited files for boosting."""
        self._recent_files = files
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN


def build_context_for_prompt(
    query: str,
    project_dir: Path | str,
    max_tokens: int = 8000,
) -> str:
    """
    Convenience function to build context string.
    
    Args:
        query: User's query
        project_dir: Project directory
        max_tokens: Maximum tokens
        
    Returns:
        Context string ready for prompt
    """
    from .indexer import Indexer
    
    indexer = Indexer(project_dir)
    builder = ContextBuilder(indexer, max_tokens=max_tokens)
    context = builder.build(query)
    
    return context.to_prompt()
