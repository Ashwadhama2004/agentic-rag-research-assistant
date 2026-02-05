"""
Result Reranking
Reranks retrieval results for improved relevance.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from backend.app.rag.retriever import RetrievalResult


class SimpleReranker:
    """
    Simple reranking based on various scoring factors.
    Can be extended with cross-encoder models for better accuracy.
    """
    
    def __init__(
        self,
        length_penalty: float = 0.1,
        recency_boost: float = 0.05,
        source_diversity: bool = True
    ):
        self.length_penalty = length_penalty
        self.recency_boost = recency_boost
        self.source_diversity = source_diversity
    
    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        Rerank results based on multiple factors.
        
        Args:
            results: Initial retrieval results
            query: Original query for relevance computation
            top_k: Return top K results after reranking
        
        Returns:
            Reranked list of results
        """
        if not results:
            return []
        
        scored_results = []
        
        for result in results:
            # Start with original score
            score = result.score
            
            # Apply length penalty (prefer medium-length chunks)
            content_len = len(result.content)
            if content_len < 100:
                score -= self.length_penalty * 0.5
            elif content_len > 2000:
                score -= self.length_penalty * 0.3
            
            # Boost if query terms appear in content
            query_terms = set(query.lower().split())
            content_lower = result.content.lower()
            term_matches = sum(1 for term in query_terms if term in content_lower)
            score += 0.02 * term_matches
            
            scored_results.append((result, score))
        
        # Sort by adjusted score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply source diversity if enabled
        if self.source_diversity and len(scored_results) > 1:
            scored_results = self._apply_diversity(scored_results)
        
        # Extract reranked results
        reranked = [r for r, _ in scored_results]
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def _apply_diversity(
        self,
        scored_results: List[tuple]
    ) -> List[tuple]:
        """
        Apply source diversity to avoid too many results from same document.
        """
        seen_docs = {}
        diverse_results = []
        remaining = []
        
        for result, score in scored_results:
            doc_id = result.metadata.get("document_id", "unknown")
            
            if doc_id not in seen_docs:
                seen_docs[doc_id] = 1
                diverse_results.append((result, score))
            else:
                if seen_docs[doc_id] < 2:  # Allow max 2 from same doc in top results
                    seen_docs[doc_id] += 1
                    diverse_results.append((result, score * 0.95))  # Slight penalty
                else:
                    remaining.append((result, score * 0.9))
        
        # Add remaining results at the end
        diverse_results.extend(remaining)
        
        return diverse_results


class CrossEncoderReranker:
    """
    Reranker using a cross-encoder model for more accurate relevance scoring.
    Requires sentence-transformers with cross-encoder support.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "CrossEncoder requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        top_k: int = None
    ) -> List[RetrievalResult]:
        """
        Rerank using cross-encoder scores.
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, r.content) for r in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Combine results with scores
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        reranked = [r for r, _ in scored_results]
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked


def get_reranker(use_cross_encoder: bool = False) -> SimpleReranker:
    """Get a reranker instance."""
    if use_cross_encoder:
        return CrossEncoderReranker()
    return SimpleReranker()
