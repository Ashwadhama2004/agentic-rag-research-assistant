"""
Long-Term Memory
Manages persistent memory with summaries and key insights.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from sqlalchemy.orm import Session

from backend.app.rag.embedder import get_embedder
from backend.app.rag.generator import get_generator
from backend.app.endee_client.multi_tenant import get_tenant_manager
from backend.app.database.schema import MemorySummary


@dataclass
class MemoryEntry:
    """Represents a long-term memory entry."""
    id: str
    summary_type: str  # 'conversation', 'research', 'document'
    content: str
    source_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class LongTermMemory:
    """
    Manages long-term memory with summaries and compressed knowledge.
    """
    
    def __init__(self, user_id: int, db: Session = None):
        self.user_id = user_id
        self.db = db
        self.embedder = get_embedder()
        self.generator = get_generator()
        self.tenant_manager = get_tenant_manager(user_id)
    
    def store_summary(
        self,
        content: str,
        summary_type: str,
        source_ids: List[str] = None
    ) -> MemoryEntry:
        """
        Store a summary in long-term memory.
        
        Args:
            content: Summary content
            summary_type: Type of summary ('conversation', 'research', 'document')
            source_ids: IDs of source items that were summarized
        
        Returns:
            MemoryEntry object
        """
        entry_id = f"summary_{uuid.uuid4().hex[:12]}"
        
        # Generate embedding
        embedding = self.embedder.encode_single(content)
        
        # Store in vector database
        self.tenant_manager.add_summary(
            summary_id=entry_id,
            embedding=embedding,
            content=content,
            metadata={
                "summary_type": summary_type,
                "source_ids": source_ids or [],
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Store in SQL database if available
        if self.db:
            db_entry = MemorySummary(
                user_id=self.user_id,
                summary_type=summary_type,
                content=content,
                source_ids=source_ids or []
            )
            self.db.add(db_entry)
            self.db.commit()
        
        return MemoryEntry(
            id=entry_id,
            summary_type=summary_type,
            content=content,
            source_ids=source_ids or []
        )
    
    def summarize_and_store(
        self,
        texts: List[str],
        summary_type: str,
        source_ids: List[str] = None,
        max_length: int = 200
    ) -> MemoryEntry:
        """
        Generate a summary of texts and store it.
        
        Args:
            texts: List of texts to summarize
            summary_type: Type of summary
            source_ids: Source item IDs
            max_length: Maximum summary length in words
        """
        # Combine texts
        combined = "\n\n---\n\n".join(texts)
        
        # Generate summary using LLM
        summary = self.generator.summarize(combined, max_length)
        
        return self.store_summary(summary, summary_type, source_ids)
    
    def search_memories(
        self,
        query: str,
        top_k: int = 3,
        summary_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search long-term memory for relevant summaries.
        
        Args:
            query: Search query
            top_k: Number of results
            summary_type: Filter by summary type
        """
        query_embedding = self.embedder.encode_query(query)
        
        results = self.tenant_manager.search_summaries(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Filter by type if specified
        if summary_type:
            results = [
                r for r in results
                if r.get("metadata", {}).get("summary_type") == summary_type
            ]
        
        return results
    
    def get_relevant_context(
        self,
        query: str,
        max_entries: int = 3
    ) -> str:
        """
        Get relevant long-term memory as context string.
        """
        memories = self.search_memories(query, top_k=max_entries)
        
        if not memories:
            return ""
        
        parts = []
        for mem in memories:
            summary_type = mem.get("metadata", {}).get("summary_type", "general")
            content = mem.get("content", "")
            parts.append(f"[{summary_type.title()} Memory]\n{content}")
        
        return "\n\n".join(parts)


def create_long_term_memory(user_id: int, db: Session = None) -> LongTermMemory:
    """Create a long-term memory instance for a user."""
    return LongTermMemory(user_id, db)
