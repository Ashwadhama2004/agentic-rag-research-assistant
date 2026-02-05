"""
Endee Vector Database Client
HTTP Client for Endee vector database (https://github.com/EndeeLabs/endee).

Endee is a high-performance vector database. This client connects to 
the Endee server via HTTP REST API.

For local development without Endee server, uses an in-memory fallback.
"""
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from pathlib import Path
import requests
from requests.exceptions import ConnectionError, Timeout

from backend.app.config import config


# Endee Server Configuration
ENDEE_SERVER_URL = os.getenv("ENDEE_SERVER_URL", "http://localhost:8080")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")


@dataclass
class VectorDocument:
    """Represents a document stored in the vector database."""
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: str = ""


class EndeeCollection:
    """A collection of vectors in the Endee database (in-memory fallback)."""
    
    def __init__(self, name: str, dimension: int = 384):
        self.name = name
        self.dimension = dimension
        self.documents: Dict[str, VectorDocument] = {}
    
    def add(self, doc_id: str, embedding: np.ndarray, content: str = "", metadata: Dict[str, Any] = None):
        """Add a document to the collection."""
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} doesn't match collection dimension {self.dimension}")
        
        self.documents[doc_id] = VectorDocument(
            id=doc_id,
            embedding=np.array(embedding),
            content=content,
            metadata=metadata or {}
        )
    
    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, filter_metadata: Dict[str, Any] = None) -> List[Tuple[VectorDocument, float]]:
        """Search for similar documents using cosine similarity."""
        if len(self.documents) == 0:
            return []
        
        query_embedding = np.array(query_embedding)
        results = []
        
        for doc in self.documents.values():
            # Apply metadata filter if provided
            if filter_metadata:
                match = all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding) + 1e-8
            )
            results.append((doc, float(similarity)))
        
        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return len(self.documents)
    
    def clear(self):
        """Clear all documents from the collection."""
        self.documents.clear()


class EndeeHTTPClient:
    """
    HTTP Client for Endee Vector Database Server.
    Connects to the Endee REST API at the configured server URL.
    """
    
    def __init__(self, base_url: str = None, auth_token: str = None):
        self.base_url = base_url or ENDEE_SERVER_URL
        self.auth_token = auth_token or ENDEE_AUTH_TOKEN
        self._is_available = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication if configured."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers
    
    def is_available(self) -> bool:
        """Check if Endee server is available."""
        if self._is_available is not None:
            return self._is_available
        
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/index/list",
                headers=self._get_headers(),
                timeout=2
            )
            self._is_available = response.status_code == 200
        except (ConnectionError, Timeout):
            self._is_available = False
        
        return self._is_available
    
    def create_index(self, name: str, dimension: int = 384) -> bool:
        """Create a new vector index (collection)."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/index/create",
                headers=self._get_headers(),
                json={"name": name, "dimension": dimension},
                timeout=5
            )
            return response.status_code == 200
        except (ConnectionError, Timeout):
            return False
    
    def insert_vectors(self, index_name: str, ids: List[str], vectors: List[List[float]], 
                       contents: List[str] = None, metadatas: List[Dict] = None) -> bool:
        """Insert vectors into an index."""
        try:
            data = {
                "index": index_name,
                "ids": ids,
                "vectors": vectors,
                "metadata": metadatas or [{} for _ in ids]
            }
            if contents:
                data["contents"] = contents
            
            response = requests.post(
                f"{self.base_url}/api/v1/vector/insert",
                headers=self._get_headers(),
                json=data,
                timeout=30
            )
            return response.status_code == 200
        except (ConnectionError, Timeout):
            return False
    
    def search_vectors(self, index_name: str, query_vector: List[float], 
                       top_k: int = 5) -> List[Dict]:
        """Search for similar vectors."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/vector/search",
                headers=self._get_headers(),
                json={
                    "index": index_name,
                    "vector": query_vector,
                    "top_k": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except (ConnectionError, Timeout):
            return []
    
    def delete_vectors(self, index_name: str, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/vector/delete",
                headers=self._get_headers(),
                json={"index": index_name, "ids": ids},
                timeout=5
            )
            return response.status_code == 200
        except (ConnectionError, Timeout):
            return False


class EndeeClient:
    """
    Main client for interacting with Endee vector database.
    
    Uses HTTP API when Endee server is running (from GitHub: EndeeLabs/endee).
    Falls back to local file persistence when server is unavailable.
    
    The Endee server can be run via:
    - Docker: docker run -p 8080:8080 -v endee-data:/data endeeio/endee-server:latest
    - Local build: See endee_vdb/README.md for instructions
    """
    
    def __init__(self, path: str = None):
        self.path = Path(path or config.ENDEE_PATH)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collections: Dict[str, EndeeCollection] = {}
        
        # Try to connect to Endee server
        self.http_client = EndeeHTTPClient()
        self.use_server = self.http_client.is_available()
        
        if self.use_server:
            print(f"Connected to Endee server at {ENDEE_SERVER_URL}")
        else:
            print(f"Endee server not available, using local file storage at {self.path}")
            self._load_collections()
    
    def _get_collection_path(self, name: str) -> Path:
        """Get the file path for a collection."""
        return self.path / f"{name}.pkl"
    
    def _load_collections(self):
        """Load all collections from disk."""
        for file in self.path.glob("*.pkl"):
            try:
                with open(file, "rb") as f:
                    collection = pickle.load(f)
                    self.collections[collection.name] = collection
            except Exception as e:
                print(f"Error loading collection {file}: {e}")
    
    def _save_collection(self, name: str):
        """Save a collection to disk."""
        if name in self.collections:
            collection_path = self._get_collection_path(name)
            with open(collection_path, "wb") as f:
                pickle.dump(self.collections[name], f)
    
    def create_collection(self, name: str, dimension: int = 384) -> EndeeCollection:
        """Create a new collection."""
        if self.use_server:
            self.http_client.create_index(name, dimension)
        
        if name in self.collections:
            return self.collections[name]
        
        collection = EndeeCollection(name=name, dimension=dimension)
        self.collections[name] = collection
        self._save_collection(name)
        return collection
    
    def get_collection(self, name: str) -> Optional[EndeeCollection]:
        """Get a collection by name."""
        return self.collections.get(name)
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if name in self.collections:
            del self.collections[name]
            collection_path = self._get_collection_path(name)
            if collection_path.exists():
                collection_path.unlink()
            return True
        return False
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        return list(self.collections.keys())
    
    def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[np.ndarray],
        contents: List[str] = None,
        metadatas: List[Dict[str, Any]] = None
    ):
        """Add multiple documents to a collection."""
        collection = self.get_collection(collection_name)
        if not collection:
            collection = self.create_collection(collection_name)
        
        contents = contents or [""] * len(ids)
        metadatas = metadatas or [{}] * len(ids)
        
        # Use Endee server if available
        if self.use_server:
            vectors = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
            self.http_client.insert_vectors(collection_name, ids, vectors, contents, metadatas)
        
        # Also store locally for hybrid access
        for doc_id, embedding, content, metadata in zip(ids, embeddings, contents, metadatas):
            collection.add(doc_id, embedding, content, metadata)
        
        self._save_collection(collection_name)
    
    def search(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in a collection."""
        # Try Endee server first
        if self.use_server:
            query_vec = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            server_results = self.http_client.search_vectors(collection_name, query_vec, top_k)
            if server_results:
                return server_results
        
        # Fall back to local search
        collection = self.get_collection(collection_name)
        if not collection:
            return []
        
        results = collection.search(query_embedding, top_k, filter_metadata)
        
        return [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def delete_documents(self, collection_name: str, ids: List[str]) -> int:
        """Delete documents from a collection."""
        if self.use_server:
            self.http_client.delete_vectors(collection_name, ids)
        
        collection = self.get_collection(collection_name)
        if not collection:
            return 0
        
        deleted = 0
        for doc_id in ids:
            if collection.delete(doc_id):
                deleted += 1
        
        self._save_collection(collection_name)
        return deleted
    
    def get_document(self, collection_name: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document from a collection."""
        collection = self.get_collection(collection_name)
        if not collection:
            return None
        
        doc = collection.get(doc_id)
        if not doc:
            return None
        
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata
        }


# Global client instance
_client: Optional[EndeeClient] = None


def get_endee_client() -> EndeeClient:
    """Get the global Endee client instance."""
    global _client
    if _client is None:
        _client = EndeeClient()
    return _client
