"""
Vector Index for Code Chunks

Uses FAISS for efficient similarity search.
Stores metadata in SQLite for persistence.
Supports incremental updates.

Structure:
    .distiq-code/
    ├── index.faiss       # Vector index
    ├── metadata.db       # SQLite metadata
    └── file_hashes.json  # For delta updates
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


class VectorStore:
    """
    FAISS-based vector store for code chunks.
    
    Features:
    - Fast similarity search
    - Persistence to disk
    - Incremental updates
    - Metadata storage
    """
    
    def __init__(
        self,
        index_dir: Path,
        embedding_dim: int = 768,
    ):
        """
        Initialize vector store.
        
        Args:
            index_dir: Directory to store index files
            embedding_dim: Dimension of embeddings
        """
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        
        # Create directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.faiss_path = index_dir / "index.faiss"
        self.metadata_path = index_dir / "metadata.db"
        self.hashes_path = index_dir / "file_hashes.json"
        
        # FAISS index
        self._index = None
        self._chunk_ids: list[str] = []  # Map index position to chunk_id
        
        # Metadata database
        self._db: sqlite3.Connection | None = None
        
        # File hashes for delta updates
        self._file_hashes: dict[str, str] = {}
        
        # Initialize
        self._init_faiss()
        self._init_database()
        self._load_hashes()
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            
            # Try to load existing index
            if self.faiss_path.exists():
                self._index = faiss.read_index(str(self.faiss_path))
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
            else:
                # Create new index
                # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                logger.info(f"Created new FAISS index (dim={self.embedding_dim})")
                
        except ImportError:
            logger.error(
                "FAISS not available. Install with: pip install faiss-cpu "
                "or pip install faiss-gpu (for GPU support)"
            )
            raise
    
    def _init_database(self):
        """Initialize SQLite metadata database."""
        self._db = sqlite3.connect(str(self.metadata_path))
        self._db.row_factory = sqlite3.Row
        
        # Create tables
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                index_position INTEGER,
                chunk_type TEXT,
                name TEXT,
                file_path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                content TEXT,
                signature TEXT,
                docstring TEXT,
                language TEXT,
                file_hash TEXT,
                created_at REAL
            )
        """)
        
        # Create indexes
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_name ON chunks(name)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)")
        
        self._db.commit()
        
        # Load chunk IDs
        cursor = self._db.execute(
            "SELECT chunk_id FROM chunks ORDER BY index_position"
        )
        self._chunk_ids = [row["chunk_id"] for row in cursor]
        
        logger.debug(f"Loaded {len(self._chunk_ids)} chunk IDs from database")
    
    def _load_hashes(self):
        """Load file hashes for delta updates."""
        if self.hashes_path.exists():
            try:
                self._file_hashes = json.loads(self.hashes_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load file hashes: {e}")
                self._file_hashes = {}
    
    def _save_hashes(self):
        """Save file hashes."""
        self.hashes_path.write_text(json.dumps(self._file_hashes, indent=2))
    
    def add(
        self,
        chunk_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """
        Add a single chunk to the index.
        
        Args:
            chunk_id: Unique chunk identifier
            embedding: Embedding vector
            metadata: Chunk metadata (type, name, content, etc.)
        """
        import time
        
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Add to FAISS
        self._index.add(embedding.astype(np.float32))
        position = self._index.ntotal - 1
        
        # Track ID
        self._chunk_ids.append(chunk_id)
        
        # Add to database
        self._db.execute("""
            INSERT OR REPLACE INTO chunks 
            (chunk_id, index_position, chunk_type, name, file_path, 
             start_line, end_line, content, signature, docstring, 
             language, file_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            position,
            metadata.get("chunk_type", ""),
            metadata.get("name", ""),
            metadata.get("file_path", ""),
            metadata.get("start_line", 0),
            metadata.get("end_line", 0),
            metadata.get("content", ""),
            metadata.get("signature", ""),
            metadata.get("docstring", ""),
            metadata.get("language", ""),
            metadata.get("file_hash", ""),
            time.time(),
        ))
        
        self._db.commit()
    
    def add_batch(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add multiple chunks to the index.
        
        Args:
            chunk_ids: List of chunk IDs
            embeddings: Array of embeddings (n_chunks, embedding_dim)
            metadatas: List of metadata dicts
        """
        import time
        
        if len(chunk_ids) == 0:
            return
        
        # Add to FAISS
        start_position = self._index.ntotal
        self._index.add(embeddings.astype(np.float32))
        
        # Track IDs
        self._chunk_ids.extend(chunk_ids)
        
        # Add to database
        now = time.time()
        for i, (chunk_id, metadata) in enumerate(zip(chunk_ids, metadatas)):
            self._db.execute("""
                INSERT OR REPLACE INTO chunks 
                (chunk_id, index_position, chunk_type, name, file_path, 
                 start_line, end_line, content, signature, docstring, 
                 language, file_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_id,
                start_position + i,
                metadata.get("chunk_type", ""),
                metadata.get("name", ""),
                metadata.get("file_path", ""),
                metadata.get("start_line", 0),
                metadata.get("end_line", 0),
                metadata.get("content", ""),
                metadata.get("signature", ""),
                metadata.get("docstring", ""),
                metadata.get("language", ""),
                metadata.get("file_hash", ""),
                now,
            ))
        
        self._db.commit()
        logger.debug(f"Added {len(chunk_ids)} chunks to index")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of matching chunks with scores
        """
        if self._index.ntotal == 0:
            return []
        
        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            
            # Get metadata from database
            chunk_id = self._chunk_ids[idx] if idx < len(self._chunk_ids) else None
            if not chunk_id:
                continue
            
            cursor = self._db.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            )
            row = cursor.fetchone()
            
            if row:
                results.append({
                    "chunk_id": row["chunk_id"],
                    "score": float(score),
                    "chunk_type": row["chunk_type"],
                    "name": row["name"],
                    "file_path": row["file_path"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "content": row["content"],
                    "signature": row["signature"],
                    "docstring": row["docstring"],
                    "language": row["language"],
                })
        
        return results
    
    def get_by_file(self, file_path: str) -> list[dict[str, Any]]:
        """Get all chunks from a specific file."""
        cursor = self._db.execute(
            "SELECT * FROM chunks WHERE file_path = ?", (file_path,)
        )
        return [dict(row) for row in cursor]
    
    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all chunks from a file.
        
        Note: This marks chunks as deleted but doesn't remove from FAISS
        (would require rebuilding index). Use rebuild_index() periodically.
        
        Args:
            file_path: File path to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        cursor = self._db.execute(
            "SELECT chunk_id FROM chunks WHERE file_path = ?", (file_path,)
        )
        chunk_ids = [row["chunk_id"] for row in cursor]
        
        # Remove from chunk_ids list (mark as deleted)
        for chunk_id in chunk_ids:
            if chunk_id in self._chunk_ids:
                idx = self._chunk_ids.index(chunk_id)
                self._chunk_ids[idx] = None  # Mark as deleted
        
        # Remove from database
        self._db.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self._db.commit()
        
        return len(chunk_ids)
    
    def needs_update(self, file_path: str, file_hash: str) -> bool:
        """Check if a file needs re-indexing."""
        old_hash = self._file_hashes.get(file_path)
        return old_hash != file_hash
    
    def mark_updated(self, file_path: str, file_hash: str) -> None:
        """Mark a file as updated."""
        self._file_hashes[file_path] = file_hash
    
    def save(self) -> None:
        """Save index to disk."""
        import faiss
        
        faiss.write_index(self._index, str(self.faiss_path))
        self._save_hashes()
        logger.info(f"Saved index with {self._index.ntotal} vectors")
    
    def rebuild_index(self) -> None:
        """
        Rebuild FAISS index removing deleted entries.
        
        Call periodically after many deletions.
        """
        import faiss
        
        # Get all valid chunks
        cursor = self._db.execute(
            "SELECT chunk_id FROM chunks ORDER BY index_position"
        )
        valid_ids = {row["chunk_id"] for row in cursor}
        
        # Get valid embeddings
        # This requires re-embedding, so we just log for now
        logger.warning("Index rebuild not implemented - requires re-embedding chunks")
    
    @property
    def size(self) -> int:
        """Number of vectors in index."""
        return self._index.ntotal if self._index else 0
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        cursor = self._db.execute("SELECT COUNT(*) as count FROM chunks")
        db_count = cursor.fetchone()["count"]
        
        cursor = self._db.execute(
            "SELECT file_path, COUNT(*) as count FROM chunks GROUP BY file_path"
        )
        files = {row["file_path"]: row["count"] for row in cursor}
        
        return {
            "total_vectors": self._index.ntotal if self._index else 0,
            "total_chunks": db_count,
            "total_files": len(files),
            "embedding_dim": self.embedding_dim,
        }
    
    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None
