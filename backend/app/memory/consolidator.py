"""
Memory Consolidation
Consolidates and compresses memories over time.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from backend.app.memory.short_term import ShortTermMemory
from backend.app.memory.long_term import LongTermMemory, create_long_term_memory
from backend.app.rag.generator import get_generator
from backend.app.database.schema import Conversation, Message


class MemoryConsolidator:
    """
    Consolidates short-term memories into long-term storage.
    Implements memory compression and cleanup.
    """
    
    def __init__(self, user_id: int, db: Session):
        self.user_id = user_id
        self.db = db
        self.generator = get_generator()
        self.long_term = create_long_term_memory(user_id, db)
    
    def consolidate_conversation(
        self,
        conversation_id: int,
        force: bool = False
    ) -> Optional[str]:
        """
        Consolidate a conversation into a summary.
        
        Args:
            conversation_id: ID of conversation to consolidate
            force: Force consolidation even if recent
        
        Returns:
            Summary content if created, None otherwise
        """
        # Get conversation
        conversation = self.db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == self.user_id
        ).first()
        
        if not conversation:
            return None
        
        # Check if recent (don't consolidate fresh conversations)
        if not force:
            age = datetime.utcnow() - conversation.updated_at
            if age < timedelta(hours=24):
                return None
        
        # Get messages
        messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at).all()
        
        if len(messages) < 5:  # Don't consolidate short conversations
            return None
        
        # Format messages for summarization
        texts = [
            f"{msg.role.title()}: {msg.content}"
            for msg in messages
        ]
        
        # Generate and store summary
        entry = self.long_term.summarize_and_store(
            texts=texts,
            summary_type="conversation",
            source_ids=[str(conversation_id)],
            max_length=150
        )
        
        return entry.content
    
    def consolidate_old_conversations(
        self,
        older_than_days: int = 7,
        max_consolidations: int = 10
    ) -> int:
        """
        Consolidate conversations older than specified days.
        
        Returns:
            Number of conversations consolidated
        """
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Find old conversations that haven't been consolidated
        old_conversations = self.db.query(Conversation).filter(
            Conversation.user_id == self.user_id,
            Conversation.updated_at < cutoff
        ).limit(max_consolidations).all()
        
        count = 0
        for conv in old_conversations:
            result = self.consolidate_conversation(conv.id, force=True)
            if result:
                count += 1
        
        return count
    
    def consolidate_research(
        self,
        research_items: List[Dict[str, str]],
        topic: str
    ) -> str:
        """
        Consolidate research findings into a summary.
        
        Args:
            research_items: List of research findings
            topic: Research topic
        
        Returns:
            Summary content
        """
        texts = [
            f"Finding: {item.get('content', '')}"
            for item in research_items
        ]
        
        entry = self.long_term.summarize_and_store(
            texts=texts,
            summary_type="research",
            source_ids=[item.get("id", "") for item in research_items],
            max_length=200
        )
        
        return entry.content
    
    def cleanup_old_data(
        self,
        keep_days: int = 30
    ) -> Dict[str, int]:
        """
        Clean up old data while preserving consolidated summaries.
        
        Returns:
            Counts of deleted items by type
        """
        cutoff = datetime.utcnow() - timedelta(days=keep_days)
        deleted = {"conversations": 0, "messages": 0}
        
        # Find old conversations
        old_convs = self.db.query(Conversation).filter(
            Conversation.user_id == self.user_id,
            Conversation.updated_at < cutoff
        ).all()
        
        for conv in old_convs:
            # Consolidate before deleting
            self.consolidate_conversation(conv.id, force=True)
            
            # Count messages
            msg_count = self.db.query(Message).filter(
                Message.conversation_id == conv.id
            ).count()
            deleted["messages"] += msg_count
            
            # Delete conversation (cascade deletes messages)
            self.db.delete(conv)
            deleted["conversations"] += 1
        
        self.db.commit()
        
        return deleted


def create_consolidator(user_id: int, db: Session) -> MemoryConsolidator:
    """Create a memory consolidator for a user."""
    return MemoryConsolidator(user_id, db)
