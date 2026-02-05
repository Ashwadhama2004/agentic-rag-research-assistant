"""
Short-Term Memory
Manages conversation history and recent context.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from backend.app.rag.embedder import get_embedder
from backend.app.endee_client.multi_tenant import get_tenant_manager


@dataclass
class ConversationMessage:
    """Represents a single conversation message."""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """
    Manages short-term conversation memory.
    Stores recent messages and retrieves relevant context.
    """
    
    def __init__(self, user_id: int, max_messages: int = 20):
        self.user_id = user_id
        self.max_messages = max_messages
        self.messages: List[ConversationMessage] = []
        self.embedder = get_embedder()
        self.tenant_manager = get_tenant_manager(user_id)
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None,
        store_embedding: bool = True
    ) -> ConversationMessage:
        """
        Add a message to short-term memory.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (sources, latency, etc.)
            store_embedding: Whether to store in vector DB
        
        Returns:
            The created message object
        """
        message = ConversationMessage(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        
        # Trim to max messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Store in vector database for semantic retrieval
        if store_embedding:
            embedding = self.embedder.encode_single(content)
            self.tenant_manager.add_conversation_message(
                message_id=message.id,
                embedding=embedding,
                content=content,
                metadata={
                    "role": role,
                    "timestamp": message.timestamp.isoformat(),
                    "user_id": self.user_id,
                    **(metadata or {})
                }
            )
        
        return message
    
    def get_recent_messages(self, n: int = 10) -> List[ConversationMessage]:
        """Get the n most recent messages."""
        return self.messages[-n:]
    
    def get_context_window(self, n: int = 10) -> str:
        """Get recent messages formatted as context string."""
        recent = self.get_recent_messages(n)
        
        parts = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{role_label}: {msg.content}")
        
        return "\n\n".join(parts)
    
    def get_context_for_llm(self, n: int = 10) -> List[Dict[str, str]]:
        """Get recent messages in LLM chat format."""
        recent = self.get_recent_messages(n)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent
        ]
    
    def search_relevant_context(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant past messages using semantic similarity.
        """
        query_embedding = self.embedder.encode_query(query)
        return self.tenant_manager.search_conversations(
            query_embedding=query_embedding,
            top_k=top_k
        )
    
    def clear(self):
        """Clear all messages from short-term memory."""
        self.messages.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory state to dictionary."""
        return {
            "user_id": self.user_id,
            "message_count": len(self.messages),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in self.messages
            ]
        }


def create_short_term_memory(user_id: int, max_messages: int = 20) -> ShortTermMemory:
    """Create a short-term memory instance for a user."""
    return ShortTermMemory(user_id, max_messages)
