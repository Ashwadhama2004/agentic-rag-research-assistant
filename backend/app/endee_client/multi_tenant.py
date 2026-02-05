"""
Multi-Tenant Helpers for Endee
Provides utilities for managing multi-tenant data isolation.
"""
from typing import List, Dict, Any, Optional
import numpy as np

from backend.app.endee_client.client import get_endee_client
from backend.app.endee_client.collections import (
    get_collection_name, 
    CollectionType,
    create_user_collections,
    delete_user_collections
)


class MultiTenantManager:
    """Manages multi-tenant operations for Endee."""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.client = get_endee_client()
        self._ensure_collections()
    
    def _ensure_collections(self):
        """Ensure all user collections exist."""
        create_user_collections(self.user_id)
    
    def _get_collection_name(self, collection_type: str) -> str:
        """Get the full collection name for this user."""
        return get_collection_name(self.user_id, collection_type)
    
    # Document Operations
    def add_document_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[np.ndarray],
        contents: List[str],
        metadatas: List[Dict[str, Any]] = None
    ):
        """Add document chunks to the user's document collection."""
        collection_name = self._get_collection_name(CollectionType.DOCUMENTS)
        self.client.add_documents(
            collection_name=collection_name,
            ids=chunk_ids,
            embeddings=embeddings,
            contents=contents,
            metadatas=metadatas
        )
    
    def search_documents(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_id: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in user's documents."""
        collection_name = self._get_collection_name(CollectionType.DOCUMENTS)
        filter_metadata = {"document_id": document_id} if document_id else None
        return self.client.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
    
    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a specific document."""
        collection_name = self._get_collection_name(CollectionType.DOCUMENTS)
        collection = self.client.get_collection(collection_name)
        
        if not collection:
            return 0
        
        # Find all chunks with this document_id
        chunk_ids = [
            doc_id for doc_id, doc in collection.documents.items()
            if doc.metadata.get("document_id") == document_id
        ]
        
        return self.client.delete_documents(collection_name, chunk_ids)
    
    # Conversation Operations
    def add_conversation_message(
        self,
        message_id: str,
        embedding: np.ndarray,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Add a conversation message to memory."""
        collection_name = self._get_collection_name(CollectionType.CONVERSATION)
        self.client.add_documents(
            collection_name=collection_name,
            ids=[message_id],
            embeddings=[embedding],
            contents=[content],
            metadatas=[metadata or {}]
        )
    
    def search_conversations(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search past conversations for relevant context."""
        collection_name = self._get_collection_name(CollectionType.CONVERSATION)
        return self.client.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )
    
    # Research Operations
    def add_research_result(
        self,
        result_id: str,
        embedding: np.ndarray,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Add a research result to the cache."""
        collection_name = self._get_collection_name(CollectionType.RESEARCH)
        self.client.add_documents(
            collection_name=collection_name,
            ids=[result_id],
            embeddings=[embedding],
            contents=[content],
            metadatas=[metadata or {}]
        )
    
    def search_research(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search cached research results."""
        collection_name = self._get_collection_name(CollectionType.RESEARCH)
        return self.client.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )
    
    # Agent Operations
    def add_agent_step(
        self,
        step_id: str,
        embedding: np.ndarray,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Add an agent execution step."""
        collection_name = self._get_collection_name(CollectionType.AGENT_STEPS)
        self.client.add_documents(
            collection_name=collection_name,
            ids=[step_id],
            embeddings=[embedding],
            contents=[content],
            metadatas=[metadata or {}]
        )
    
    # Summary Operations
    def add_summary(
        self,
        summary_id: str,
        embedding: np.ndarray,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Add a memory summary."""
        collection_name = self._get_collection_name(CollectionType.SUMMARIES)
        self.client.add_documents(
            collection_name=collection_name,
            ids=[summary_id],
            embeddings=[embedding],
            contents=[content],
            metadatas=[metadata or {}]
        )
    
    def search_summaries(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Search long-term memory summaries."""
        collection_name = self._get_collection_name(CollectionType.SUMMARIES)
        return self.client.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )
    
    # Cleanup
    def cleanup_all_data(self):
        """Delete all user data from Endee."""
        delete_user_collections(self.user_id)


def get_tenant_manager(user_id: int) -> MultiTenantManager:
    """Get a multi-tenant manager for a specific user."""
    return MultiTenantManager(user_id)
