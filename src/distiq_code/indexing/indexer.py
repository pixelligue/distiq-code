"""
Main Indexer

Orchestrates the indexing pipeline:
1. Parse files into chunks (parser.py)
2. Generate embeddings (embedder.py)  
3. Store in vector index (vector_store.py)

Supports incremental updates via file hash tracking.
"""

import hashlib
from pathlib import Path

from loguru import logger

from .parser import CodeParser, CodeChunk
from .embedder import CodeEmbedder
from .vector_store import VectorStore


class Indexer:
    """
    Main indexer for code projects.
    
    Usage:
        indexer = Indexer(project_dir)
        await indexer.index()  # Full index
        await indexer.update()  # Incremental update
        results = indexer.search("authentication")
    """
    
    # Default directory name for index
    INDEX_DIR_NAME = ".distiq-code"
    
    def __init__(
        self,
        project_dir: Path | str,
        embedding_model: str | None = None,
    ):
        """
        Initialize indexer.
        
        Args:
            project_dir: Root directory of the project
            embedding_model: Optional embedding model name
        """
        self.project_dir = Path(project_dir).resolve()
        self.index_dir = self.project_dir / self.INDEX_DIR_NAME
        
        # Components
        self.parser = CodeParser()
        self.embedder = CodeEmbedder(
            model_name=embedding_model,
            cache_dir=self.index_dir,
        )
        self.vector_store: VectorStore | None = None
        
        # Stats
        self._indexed_files = 0
        self._indexed_chunks = 0
    
    def _ensure_vector_store(self):
        """Ensure vector store is initialized."""
        if self.vector_store is None:
            self.vector_store = VectorStore(
                index_dir=self.index_dir,
                embedding_dim=self.embedder.embedding_dim,
            )
    
    def index(
        self,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        show_progress: bool = True,
    ) -> dict:
        """
        Index the entire project.
        
        Args:
            extensions: File extensions to include
            exclude_patterns: Patterns to exclude
            show_progress: Show progress output
            
        Returns:
            Indexing statistics
        """
        logger.info(f"Indexing project: {self.project_dir}")
        
        self._ensure_vector_store()
        
        # Parse all files
        chunks = self.parser.parse_directory(
            self.project_dir,
            extensions=extensions,
            exclude_patterns=exclude_patterns,
        )
        
        if not chunks:
            logger.warning("No code chunks found")
            return {"files": 0, "chunks": 0}
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            # Build embedding text
            text = self._chunk_to_embedding_text(chunk)
            texts.append(text)
        
        embeddings = self.embedder.embed_batch(texts, show_progress=show_progress)
        
        # Add to vector store
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]
        
        self.vector_store.add_batch(chunk_ids, embeddings, metadatas)
        
        # Track file hashes
        file_hashes = {}
        for chunk in chunks:
            if chunk.file_path not in file_hashes:
                file_hashes[chunk.file_path] = chunk.file_hash
        
        for file_path, file_hash in file_hashes.items():
            self.vector_store.mark_updated(file_path, file_hash)
        
        # Save
        self.vector_store.save()
        self.embedder.save_cache()
        
        self._indexed_files = len(file_hashes)
        self._indexed_chunks = len(chunks)
        
        logger.success(
            f"Indexed {self._indexed_chunks} chunks from {self._indexed_files} files"
        )
        
        return self.vector_store.get_stats()
    
    def update(
        self,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict:
        """
        Incrementally update the index.
        
        Only re-indexes files that have changed.
        
        Args:
            extensions: File extensions to include
            exclude_patterns: Patterns to exclude
            
        Returns:
            Update statistics
        """
        logger.info("Checking for file changes...")
        
        self._ensure_vector_store()
        
        if extensions is None:
            extensions = list(self.parser.LANGUAGE_MAP.keys())
        
        exclude_patterns = exclude_patterns or [
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", self.INDEX_DIR_NAME,
        ]
        
        updated_files = 0
        updated_chunks = 0
        
        # Scan all files
        for ext in extensions:
            for file_path in self.project_dir.rglob(f"*{ext}"):
                # Check exclusions
                path_str = str(file_path)
                if any(pattern in path_str for pattern in exclude_patterns):
                    continue
                
                # Calculate file hash
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception:
                    continue
                
                file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                rel_path = str(file_path.relative_to(self.project_dir))
                
                # Check if needs update
                if not self.vector_store.needs_update(rel_path, file_hash):
                    continue
                
                # Delete old chunks
                self.vector_store.delete_by_file(rel_path)
                
                # Parse and index new chunks
                chunks = self.parser.parse_file(file_path)
                
                if chunks:
                    texts = [self._chunk_to_embedding_text(c) for c in chunks]
                    embeddings = self.embedder.embed_batch(texts, show_progress=False)
                    
                    chunk_ids = [c.chunk_id for c in chunks]
                    metadatas = [c.to_dict() for c in chunks]
                    
                    self.vector_store.add_batch(chunk_ids, embeddings, metadatas)
                    updated_chunks += len(chunks)
                
                self.vector_store.mark_updated(rel_path, file_hash)
                updated_files += 1
        
        if updated_files > 0:
            self.vector_store.save()
            self.embedder.save_cache()
        
        logger.info(f"Updated {updated_files} files, {updated_chunks} chunks")
        
        return {
            "updated_files": updated_files,
            "updated_chunks": updated_chunks,
            **self.vector_store.get_stats(),
        }
    
    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.3,
    ) -> list[dict]:
        """
        Search for relevant code chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of matching chunks with scores
        """
        self._ensure_vector_store()
        
        if self.vector_store.size == 0:
            logger.warning("Index is empty. Run index() first.")
            return []
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Search
        results = self.vector_store.search(
            query_embedding,
            k=k,
            min_score=min_score,
        )
        
        return results
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        self._ensure_vector_store()
        return self.vector_store.get_stats()
    
    def _chunk_to_embedding_text(self, chunk: CodeChunk) -> str:
        """Convert chunk to text for embedding."""
        parts = []
        
        # Add type and name
        parts.append(f"[{chunk.chunk_type.value.upper()}] {chunk.name}")
        
        # Add signature
        if chunk.signature:
            parts.append(chunk.signature)
        
        # Add docstring
        if chunk.docstring:
            parts.append(chunk.docstring)
        
        # Add content (truncated)
        content = chunk.content
        if len(content) > 2000:
            content = content[:2000] + "..."
        parts.append(content)
        
        return "\n".join(parts)
    
    def close(self):
        """Close all resources."""
        if self.vector_store:
            self.vector_store.close()
        self.embedder.save_cache()
