"""
Document Chunking
Splits documents into smaller chunks for embedding and retrieval.
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re
import uuid


@dataclass
class Chunk:
    """Represents a document chunk."""
    id: str
    content: str
    metadata: Dict[str, Any]
    
    def __len__(self):
        return len(self.content)


class RecursiveCharacterSplitter:
    """
    Recursively splits text by different separators to create chunks
    of approximately the target size with overlap.
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",   # Paragraphs
        "\n",     # Lines
        ". ",     # Sentences
        ", ",     # Clauses
        " ",      # Words
        ""        # Characters
    ]
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self._split_recursive(text, self.separators)
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchical separators."""
        final_chunks = []
        
        # Find the appropriate separator
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        # Process splits
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_length + split_len + len(separator) > self.chunk_size:
                if current_chunk:
                    merged = separator.join(current_chunk)
                    
                    if len(merged) > self.chunk_size and new_separators:
                        # Recursively split large chunks
                        final_chunks.extend(self._split_recursive(merged, new_separators))
                    else:
                        final_chunks.append(merged)
                    
                    # Keep some content for overlap
                    overlap_start = max(0, len(current_chunk) - 2)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(c) + len(separator) for c in current_chunk)
            
            current_chunk.append(split)
            current_length += split_len + len(separator)
        
        # Add remaining content
        if current_chunk:
            merged = separator.join(current_chunk)
            final_chunks.append(merged)
        
        return final_chunks
    
    def create_chunks(
        self, 
        text: str, 
        document_id: str,
        document_name: str,
        page_number: int = None,
        extra_metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """Create chunk objects with metadata."""
        text_chunks = self.split_text(text)
        chunks = []
        
        for i, content in enumerate(text_chunks):
            if not content.strip():
                continue
            
            chunk_id = f"{document_id}_chunk_{i}"
            
            metadata = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "char_count": len(content)
            }
            
            if page_number is not None:
                metadata["page_number"] = page_number
            
            if extra_metadata:
                metadata.update(extra_metadata)
            
            chunks.append(Chunk(
                id=chunk_id,
                content=content.strip(),
                metadata=metadata
            ))
        
        return chunks


class DocumentChunker:
    """High-level interface for document chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(
        self,
        text: str,
        document_id: str,
        document_name: str,
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """Chunk a single text document."""
        return self.splitter.create_chunks(
            text=text,
            document_id=document_id,
            document_name=document_name,
            extra_metadata=metadata
        )
    
    def chunk_pages(
        self,
        pages: List[str],
        document_id: str,
        document_name: str,
        metadata: Dict[str, Any] = None
    ) -> List[Chunk]:
        """Chunk a multi-page document."""
        all_chunks = []
        
        for page_num, page_text in enumerate(pages, start=1):
            chunks = self.splitter.create_chunks(
                text=page_text,
                document_id=document_id,
                document_name=document_name,
                page_number=page_num,
                extra_metadata=metadata
            )
            
            # Update chunk IDs to be globally unique
            for chunk in chunks:
                chunk.id = f"{document_id}_p{page_num}_{chunk.id.split('_')[-1]}"
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def estimate_chunks(self, text: str) -> int:
        """Estimate the number of chunks for a text."""
        return max(1, len(text) // (self.splitter.chunk_size - self.splitter.chunk_overlap))
