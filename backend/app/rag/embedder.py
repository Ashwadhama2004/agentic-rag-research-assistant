"""
Embedding Generator
Generates embeddings using SentenceTransformers.
"""
from typing import List, Union, Optional
import numpy as np
from functools import lru_cache

from backend.app.config import config


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers models.
    Supports batch processing and caching.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingGenerator._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
            EmbeddingGenerator._model = SentenceTransformer(config.EMBEDDING_MODEL)
            print("Embedding model loaded successfully!")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
    
    @property
    def model(self):
        """Get the loaded model."""
        if EmbeddingGenerator._model is None:
            self._load_model()
        return EmbeddingGenerator._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.encode([text], normalize=normalize)[0]
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query for retrieval.
        Some models have specific query prefixes.
        """
        return self.encode_single(query.strip())
    
    def encode_documents(
        self, 
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple documents for storage.
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings (len(documents) x dimension)
        """
        return self.encode(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        ))


# Global embedding generator instance
_embedder: Optional[EmbeddingGenerator] = None


def get_embedder() -> EmbeddingGenerator:
    """Get the global embedding generator instance."""
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingGenerator()
    return _embedder


def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a single text."""
    return get_embedder().encode_single(text)


def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for multiple texts."""
    return get_embedder().encode_documents(texts, batch_size=batch_size)
