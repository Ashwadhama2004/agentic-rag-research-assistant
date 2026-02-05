"""
Vector Retrieval
Retrieves relevant documents from Endee using semantic search.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from backend.app.endee_client.multi_tenant import get_tenant_manager
from backend.app.rag.embedder import get_embedder


@dataclass
class RetrievalResult:
    """Represents a retrieval result with metadata."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    @property
    def document_name(self) -> str:
        return self.metadata.get("document_name", "Unknown")
    
    @property
    def page_number(self) -> Optional[int]:
        return self.metadata.get("page_number")
    
    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)


class DocumentRetriever:
    """
    Retrieves relevant documents using semantic search.
    """
    
    def __init__(self, user_id: int, top_k: int = 5):
        self.user_id = user_id
        self.top_k = top_k
        self.embedder = get_embedder()
        self.tenant_manager = get_tenant_manager(user_id)
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        document_id: str = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return (overrides default)
            document_id: Filter to specific document
        
        Returns:
            List of RetrievalResult objects
        """
        k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)
        
        # Search in user's documents
        results = self.tenant_manager.search_documents(
            query_embedding=query_embedding,
            top_k=k,
            document_id=document_id
        )
        
        # Convert to RetrievalResult objects
        return [
            RetrievalResult(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"]
            )
            for r in results
        ]
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = None,
        include_conversation: bool = True,
        include_summaries: bool = True
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve documents along with relevant conversation history and summaries.
        
        Returns:
            Dict with 'documents', 'conversations', and 'summaries' keys
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.encode_query(query)
        
        results = {
            "documents": [],
            "conversations": [],
            "summaries": []
        }
        
        # Search documents
        doc_results = self.tenant_manager.search_documents(
            query_embedding=query_embedding,
            top_k=k
        )
        results["documents"] = [
            RetrievalResult(
                id=r["id"],
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"]
            )
            for r in doc_results
        ]
        
        # Search conversation history
        if include_conversation:
            conv_results = self.tenant_manager.search_conversations(
                query_embedding=query_embedding,
                top_k=min(k, 5)
            )
            results["conversations"] = [
                RetrievalResult(
                    id=r["id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in conv_results
            ]
        
        # Search long-term summaries
        if include_summaries:
            summary_results = self.tenant_manager.search_summaries(
                query_embedding=query_embedding,
                top_k=min(k, 3)
            )
            results["summaries"] = [
                RetrievalResult(
                    id=r["id"],
                    content=r["content"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in summary_results
            ]
        
        return results
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining semantic and keyword search.
        Currently uses only semantic search; extension point for BM25.
        """
        # For now, just use semantic search
        # In future, could add BM25 or other keyword methods
        return self.retrieve(query, top_k)


def create_retriever(user_id: int, top_k: int = 5) -> DocumentRetriever:
    """Create a document retriever for a user."""
    return DocumentRetriever(user_id, top_k)
