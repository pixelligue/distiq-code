"""
Tree-sitter Code Parser

Parses source code into semantic chunks using Tree-sitter AST.
Supports: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++

Each chunk contains:
- Type (function, class, method, import_block)
- Name and signature
- Content
- Location (file, lines)
- Dependencies
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class ChunkType(str, Enum):
    """Types of code chunks."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    IMPORT_BLOCK = "import_block"
    CONSTANT = "constant"
    CONFIG = "config"
    COMMENT = "comment"
    OTHER = "other"


@dataclass
class CodeChunk:
    """A semantic chunk of code."""
    
    # Identification
    chunk_id: str  # Unique hash
    chunk_type: ChunkType
    name: str  # Function/class name
    
    # Location
    file_path: str
    start_line: int
    end_line: int
    
    # Content
    content: str
    signature: str = ""  # e.g., "def foo(x: int) -> str"
    docstring: str = ""
    
    # Dependencies
    dependencies: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    
    # Metadata
    language: str = ""
    file_hash: str = ""
    
    def __post_init__(self):
        if not self.chunk_id:
            # Generate unique ID from content
            content_hash = hashlib.sha256(
                f"{self.file_path}:{self.name}:{self.content}".encode()
            ).hexdigest()[:16]
            self.chunk_id = content_hash
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "signature": self.signature,
            "docstring": self.docstring,
            "dependencies": self.dependencies,
            "references": self.references,
            "language": self.language,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CodeChunk":
        """Create from dictionary."""
        data = data.copy()
        data["chunk_type"] = ChunkType(data["chunk_type"])
        return cls(**data)


class CodeParser:
    """
    Parse code into semantic chunks using Tree-sitter.
    
    Falls back to regex-based parsing if tree-sitter is not available.
    """
    
    # Supported languages and their extensions
    LANGUAGE_MAP = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".rb": "ruby",
        ".php": "php",
    }
    
    def __init__(self):
        """Initialize parser."""
        self._ts_parser = None
        self._ts_languages = {}
        self._init_treesitter()
    
    def _init_treesitter(self):
        """Initialize tree-sitter if available."""
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            from tree_sitter import Language, Parser
            
            self._ts_parser = Parser()
            
            # Load languages
            self._ts_languages["python"] = Language(tree_sitter_python.language())
            self._ts_languages["javascript"] = Language(tree_sitter_javascript.language())
            
            logger.info("Tree-sitter initialized with Python, JavaScript")
            
        except ImportError:
            logger.warning(
                "Tree-sitter not available. Install with: "
                "pip install tree-sitter tree-sitter-python tree-sitter-javascript"
            )
            self._ts_parser = None
    
    def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """
        Parse a file into code chunks.
        
        Args:
            file_path: Path to source file
            
        Returns:
            List of code chunks
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return []
        
        # Get language
        ext = file_path.suffix.lower()
        language = self.LANGUAGE_MAP.get(ext)
        
        if not language:
            logger.debug(f"Unsupported file type: {ext}")
            return []
        
        # Read file
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return []
        
        # Calculate file hash for delta updates
        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Parse based on language
        if self._ts_parser and language in self._ts_languages:
            chunks = self._parse_with_treesitter(content, language, str(file_path))
        else:
            chunks = self._parse_with_regex(content, language, str(file_path))
        
        # Add metadata
        for chunk in chunks:
            chunk.language = language
            chunk.file_hash = file_hash
        
        return chunks
    
    def _parse_with_treesitter(
        self,
        content: str,
        language: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Parse using tree-sitter AST."""
        from tree_sitter import Parser
        
        chunks = []
        
        # Set language
        self._ts_parser.language = self._ts_languages[language]
        
        # Parse
        tree = self._ts_parser.parse(content.encode())
        root = tree.root_node
        
        # Walk AST and extract chunks
        chunks.extend(self._extract_chunks_from_node(
            root, content, file_path, language
        ))
        
        return chunks
    
    def _extract_chunks_from_node(
        self,
        node,
        content: str,
        file_path: str,
        language: str,
    ) -> list[CodeChunk]:
        """Extract chunks from tree-sitter node recursively."""
        chunks = []
        
        # Node types to extract
        if language == "python":
            function_types = ["function_definition"]
            class_types = ["class_definition"]
        elif language in ("javascript", "typescript"):
            function_types = ["function_declaration", "arrow_function", "method_definition"]
            class_types = ["class_declaration"]
        else:
            function_types = []
            class_types = []
        
        # Check current node
        if node.type in function_types:
            chunk = self._node_to_chunk(node, content, file_path, ChunkType.FUNCTION)
            if chunk:
                chunks.append(chunk)
        elif node.type in class_types:
            chunk = self._node_to_chunk(node, content, file_path, ChunkType.CLASS)
            if chunk:
                chunks.append(chunk)
        
        # Recurse into children
        for child in node.children:
            chunks.extend(self._extract_chunks_from_node(
                child, content, file_path, language
            ))
        
        return chunks
    
    def _node_to_chunk(
        self,
        node,
        content: str,
        file_path: str,
        chunk_type: ChunkType,
    ) -> CodeChunk | None:
        """Convert tree-sitter node to CodeChunk."""
        lines = content.split("\n")
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        # Extract content
        chunk_content = "\n".join(lines[start_line - 1:end_line])
        
        # Extract name
        name = "unknown"
        for child in node.children:
            if child.type == "identifier" or child.type == "name":
                name = content[child.start_byte:child.end_byte]
                break
        
        # Extract signature (first line)
        signature = lines[start_line - 1].strip() if lines else ""
        
        # Extract docstring
        docstring = ""
        for child in node.children:
            if child.type in ("string", "expression_statement"):
                doc_text = content[child.start_byte:child.end_byte].strip()
                if doc_text.startswith('"""') or doc_text.startswith("'''"):
                    docstring = doc_text.strip("\"'").strip()
                    break
        
        return CodeChunk(
            chunk_id="",
            chunk_type=chunk_type,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            content=chunk_content,
            signature=signature,
            docstring=docstring,
        )
    
    def _parse_with_regex(
        self,
        content: str,
        language: str,
        file_path: str,
    ) -> list[CodeChunk]:
        """Fallback regex-based parsing."""
        import re
        
        chunks = []
        lines = content.split("\n")
        
        if language == "python":
            # Python function pattern
            func_pattern = re.compile(
                r'^(\s*)(async\s+)?def\s+(\w+)\s*\([^)]*\)',
                re.MULTILINE
            )
            
            # Python class pattern
            class_pattern = re.compile(
                r'^(\s*)class\s+(\w+)',
                re.MULTILINE
            )
            
            # Find functions
            for match in func_pattern.finditer(content):
                indent = len(match.group(1))
                name = match.group(3)
                start_line = content[:match.start()].count("\n") + 1
                
                # Find end of function (next line with same or less indent)
                end_line = start_line
                for i, line in enumerate(lines[start_line:], start=start_line + 1):
                    if line.strip() and not line.startswith(" " * (indent + 1)) and not line.startswith("\t"):
                        if not line.strip().startswith("#"):
                            break
                    end_line = i
                
                chunk_content = "\n".join(lines[start_line - 1:end_line])
                
                chunks.append(CodeChunk(
                    chunk_id="",
                    chunk_type=ChunkType.FUNCTION,
                    name=name,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content,
                    signature=lines[start_line - 1].strip(),
                ))
            
            # Find classes
            for match in class_pattern.finditer(content):
                indent = len(match.group(1))
                name = match.group(2)
                start_line = content[:match.start()].count("\n") + 1
                
                end_line = start_line
                for i, line in enumerate(lines[start_line:], start=start_line + 1):
                    if line.strip() and not line.startswith(" " * (indent + 1)) and not line.startswith("\t"):
                        if not line.strip().startswith("#"):
                            break
                    end_line = i
                
                chunk_content = "\n".join(lines[start_line - 1:end_line])
                
                chunks.append(CodeChunk(
                    chunk_id="",
                    chunk_type=ChunkType.CLASS,
                    name=name,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content,
                    signature=lines[start_line - 1].strip(),
                ))
        
        elif language in ("javascript", "typescript"):
            # JS function patterns
            func_patterns = [
                re.compile(r'(async\s+)?function\s+(\w+)\s*\([^)]*\)', re.MULTILINE),
                re.compile(r'(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\([^)]*\)\s*=>', re.MULTILINE),
            ]
            
            for pattern in func_patterns:
                for match in pattern.finditer(content):
                    start_line = content[:match.start()].count("\n") + 1
                    name = match.group(2) if match.lastindex >= 2 else "anonymous"
                    
                    # Simple heuristic: find matching braces
                    end_line = self._find_block_end(lines, start_line - 1)
                    
                    chunk_content = "\n".join(lines[start_line - 1:end_line])
                    
                    chunks.append(CodeChunk(
                        chunk_id="",
                        chunk_type=ChunkType.FUNCTION,
                        name=name,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=chunk_content,
                        signature=lines[start_line - 1].strip()[:100],
                    ))
        
        return chunks
    
    def _find_block_end(self, lines: list[str], start: int) -> int:
        """Find end of a code block using brace counting."""
        brace_count = 0
        started = False
        
        for i, line in enumerate(lines[start:], start=start):
            brace_count += line.count("{") - line.count("}")
            if brace_count > 0:
                started = True
            if started and brace_count <= 0:
                return i + 1
        
        return len(lines)
    
    def parse_directory(
        self,
        directory: Path,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[CodeChunk]:
        """
        Parse all files in a directory.
        
        Args:
            directory: Directory to parse
            extensions: File extensions to include (default: all supported)
            exclude_patterns: Patterns to exclude (e.g., node_modules)
            
        Returns:
            List of all code chunks
        """
        if extensions is None:
            extensions = list(self.LANGUAGE_MAP.keys())
        
        exclude_patterns = exclude_patterns or [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
        ]
        
        chunks = []
        
        for ext in extensions:
            for file_path in directory.rglob(f"*{ext}"):
                # Check exclusions - match path parts, not substrings
                path_parts = file_path.parts
                if any(pattern in path_parts for pattern in exclude_patterns):
                    continue
                
                file_chunks = self.parse_file(file_path)
                chunks.extend(file_chunks)
        
        logger.info(f"Parsed {len(chunks)} chunks from {directory}")
        return chunks
