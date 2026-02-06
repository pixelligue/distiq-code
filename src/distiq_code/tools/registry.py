"""
Tool Registry

Centralized registry for all available tools.
Supports dynamic registration and discovery.

Tools are functions that the LLM can call to:
- Read/write files
- Execute commands
- Search code
- Access web
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from loguru import logger


class ToolCategory(str, Enum):
    """Tool categories."""
    FILE = "file"           # File operations
    SEARCH = "search"       # Code/file search
    EXECUTE = "execute"     # Command execution
    WEB = "web"             # Web access
    ANALYSIS = "analysis"   # Code analysis
    OTHER = "other"


@dataclass
class ToolDefinition:
    """Definition of an available tool."""
    
    name: str
    description: str
    category: ToolCategory
    
    # JSON schema for parameters
    parameters: dict = field(default_factory=dict)
    
    # Function to call
    handler: Callable | None = None
    
    # Is async
    is_async: bool = False
    
    # Requires confirmation before execution
    requires_confirmation: bool = False
    
    # Token cost estimate (for routing decisions)
    token_cost_estimate: int = 0
    
    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    """
    Registry of available tools.
    
    Usage:
        registry = ToolRegistry()
        registry.register(read_file_tool)
        
        # Get tools for LLM
        tools = registry.get_openai_tools()
        
        # Execute tool
        result = await registry.execute("Read", {"file_path": "main.py"})
    """
    
    def __init__(self):
        """Initialize registry with default tools."""
        self._tools: dict[str, ToolDefinition] = {}
        self._register_defaults()
    
    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
    
    def get(self, name: str) -> ToolDefinition | None:
        """Get tool by name."""
        return self._tools.get(name)
    
    def list_tools(
        self,
        category: ToolCategory | None = None,
    ) -> list[ToolDefinition]:
        """List all registered tools."""
        tools = list(self._tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        return tools
    
    def get_openai_tools(self) -> list[dict]:
        """Get tools in OpenAI format."""
        return [t.to_openai_schema() for t in self._tools.values()]
    
    def get_anthropic_tools(self) -> list[dict]:
        """Get tools in Anthropic format."""
        return [t.to_anthropic_schema() for t in self._tools.values()]
    
    async def execute(
        self,
        name: str,
        parameters: dict,
    ) -> Any:
        """
        Execute a tool.
        
        Args:
            name: Tool name
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        
        if not tool.handler:
            raise ValueError(f"Tool {name} has no handler")
        
        logger.debug(f"Executing tool: {name}")
        
        if tool.is_async:
            return await tool.handler(**parameters)
        else:
            return tool.handler(**parameters)
    
    def _register_defaults(self) -> None:
        """Register default tools."""
        # Read file
        self.register(ToolDefinition(
            name="Read",
            description="Read the contents of a file",
            category=ToolCategory.FILE,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line (1-indexed, optional)",
                    },
                    "end_line": {
                        "type": "integer", 
                        "description": "End line (1-indexed, optional)",
                    },
                },
                "required": ["file_path"],
            },
            handler=self._read_file,
            is_async=False,
        ))
        
        # Write file
        self.register(ToolDefinition(
            name="Write",
            description="Write content to a file (creates or overwrites)",
            category=ToolCategory.FILE,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["file_path", "content"],
            },
            handler=self._write_file,
            requires_confirmation=True,
        ))
        
        # Glob search
        self.register(ToolDefinition(
            name="Glob",
            description="Find files matching a glob pattern",
            category=ToolCategory.SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py')",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in (default: current)",
                    },
                },
                "required": ["pattern"],
            },
            handler=self._glob_search,
        ))
        
        # Grep search
        self.register(ToolDefinition(
            name="Grep",
            description="Search for text pattern in files",
            category=ToolCategory.SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in",
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include (e.g., '*.py')",
                    },
                },
                "required": ["pattern"],
            },
            handler=self._grep_search,
        ))
        
        # Bash command
        self.register(ToolDefinition(
            name="Bash",
            description="Execute a shell command",
            category=ToolCategory.EXECUTE,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 30)",
                    },
                },
                "required": ["command"],
            },
            handler=self._bash_command,
            requires_confirmation=True,
        ))
        
        # Semantic search (uses index)
        self.register(ToolDefinition(
            name="Search",
            description="Semantic search through codebase (requires indexing)",
            category=ToolCategory.SEARCH,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
            handler=self._semantic_search,
            is_async=True,
        ))
        
        # Web search (Jina AI)
        self.register(ToolDefinition(
            name="WebSearch",
            description="Search the internet for documentation, tutorials, Stack Overflow answers. Uses Jina AI.",
            category=ToolCategory.WEB,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'Python async best practices')",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (1-5, default: 3)",
                    },
                },
                "required": ["query"],
            },
            handler=self._web_search,
            is_async=True,
        ))
        
        # Read URL (Jina Reader)
        self.register(ToolDefinition(
            name="ReadURL",
            description="Read a webpage and convert to Markdown. Useful for documentation pages.",
            category=ToolCategory.WEB,
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to read (e.g., 'https://docs.python.org/3/library/asyncio.html')",
                    },
                },
                "required": ["url"],
            },
            handler=self._read_url,
            is_async=True,
        ))
    
    # Tool implementations
    def _read_file(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """Read file contents."""
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")
        
        lines = content.split("\n")
        
        if start_line is not None or end_line is not None:
            start = (start_line or 1) - 1
            end = end_line or len(lines)
            lines = lines[start:end]
            content = "\n".join(lines)
        
        return content
    
    def _write_file(self, file_path: str, content: str) -> str:
        """Write to file."""
        from pathlib import Path
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    
    def _glob_search(
        self,
        pattern: str,
        directory: str = ".",
    ) -> str:
        """Glob file search."""
        from pathlib import Path
        
        results = list(Path(directory).glob(pattern))
        
        if not results:
            return "No files found matching pattern"
        
        # Limit results
        if len(results) > 50:
            results = results[:50]
            extra = f"\n... and {len(results) - 50} more"
        else:
            extra = ""
        
        return "\n".join(str(r) for r in results) + extra
    
    def _grep_search(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
    ) -> str:
        """Grep text search."""
        import subprocess
        
        cmd = ["grep", "-rn", pattern, path]
        if include:
            cmd.extend(["--include", include])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            output = result.stdout
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            
            return output or "No matches found"
            
        except subprocess.TimeoutExpired:
            return "Search timed out"
        except FileNotFoundError:
            # grep not available, use Python
            return self._python_grep(pattern, path, include)
    
    def _python_grep(
        self,
        pattern: str,
        path: str,
        include: str | None,
    ) -> str:
        """Python-based grep fallback."""
        import re
        from pathlib import Path
        
        results = []
        root = Path(path)
        
        glob_pattern = include or "**/*"
        for file_path in root.glob(glob_pattern):
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.split("\n"), 1):
                    if re.search(pattern, line):
                        results.append(f"{file_path}:{i}: {line.strip()}")
                        
                        if len(results) >= 50:
                            return "\n".join(results) + "\n... (truncated)"
            except Exception:
                continue
        
        return "\n".join(results) if results else "No matches found"
    
    def _bash_command(
        self,
        command: str,
        timeout: int = 30,
    ) -> str:
        """Execute shell command."""
        import subprocess
        import sys
        
        # Use appropriate shell
        if sys.platform == "win32":
            shell = True
        else:
            shell = True
        
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            output = result.stdout + result.stderr
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            
            return output or "(no output)"
            
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"
    
    async def _semantic_search(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """Semantic code search."""
        try:
            from distiq_code.indexing import get_indexer
            
            indexer = get_indexer()
            results = indexer.search(query, k=limit)
            
            if not results:
                return "No relevant code found. Try indexing with: distiq-code index"
            
            output_parts = []
            for r in results:
                output_parts.append(
                    f"## {r['file_path']}:{r['start_line']}-{r['end_line']}\n"
                    f"```\n{r['content'][:500]}\n```"
                )
            
            return "\n\n".join(output_parts)
            
        except Exception as e:
            return f"Search error: {e}. Make sure project is indexed."
    
    async def _web_search(
        self,
        query: str,
        num_results: int = 3,
    ) -> str:
        """Web search using Jina AI or Tavily with automatic fallback."""
        try:
            from distiq_code.tools.web_search import smart_search
            
            response = await smart_search(query, min(num_results, 5))
            
            if not response.results:
                return f"No results found for: {query}. Set JINA_API_KEY or TAVILY_API_KEY."
            
            return response.to_context(max_results=num_results)
            
        except Exception as e:
            return f"Web search error: {e}. Set JINA_API_KEY or TAVILY_API_KEY."
    
    async def _read_url(self, url: str) -> str:
        """Read URL and convert to Markdown using Jina Reader."""
        try:
            from distiq_code.tools.web_search import read_url
            
            content = await read_url(url)
            
            # Truncate if too long
            if len(content) > 10000:
                content = content[:10000] + "\n\n... (truncated)"
            
            return content
            
        except Exception as e:
            return f"Error reading URL: {e}"


# Global registry
_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def get_available_tools(
    category: ToolCategory | None = None,
) -> list[str]:
    """Get list of available tool names."""
    registry = get_tool_registry()
    return [t.name for t in registry.list_tools(category)]
