"""
Document Management API Endpoints
Handles document upload, listing, and deletion.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime

from backend.app.database.connection import get_db
from backend.app.database.schema import Document
from backend.app.auth.security import decode_access_token
from backend.app.utils.parsers import get_document_parser
from backend.app.rag.chunker import DocumentChunker
from backend.app.rag.embedder import get_embedder
from backend.app.endee_client.multi_tenant import get_tenant_manager
from backend.app.config import config

router = APIRouter(prefix="/api/documents", tags=["Documents"])


def get_user_from_token(token: str, db: Session) -> int:
    """Extract and validate user ID from token."""
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    return payload.get("user_id")


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    token: str = Form(...),
    tags: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process a document."""
    user_id = get_user_from_token(token, db)
    
    # Validate file type
    parser = get_document_parser()
    if not parser.get_parser(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported: {parser.supported_extensions}"
        )
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Check file size
    if file_size > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {config.MAX_UPLOAD_SIZE_MB}MB"
        )
    
    try:
        # Parse document
        pages, metadata = parser.parse_pages(content, file.filename)
        
        # Create document record
        doc = Document(
            user_id=user_id,
            filename=file.filename,
            file_type=metadata.get("file_type", "unknown"),
            file_size=file_size,
            metadata={
                **metadata,
                "tags": tags.split(",") if tags else []
            },
            status="processing"
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        # Chunk document
        chunker = DocumentChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = chunker.chunk_pages(
            pages=pages,
            document_id=str(doc.id),
            document_name=file.filename
        )
        
        # Generate embeddings
        embedder = get_embedder()
        chunk_contents = [c.content for c in chunks]
        embeddings = embedder.encode_documents(chunk_contents)
        
        # Store in Endee
        tenant_manager = get_tenant_manager(user_id)
        tenant_manager.add_document_chunks(
            chunk_ids=[c.id for c in chunks],
            embeddings=list(embeddings),
            contents=chunk_contents,
            metadatas=[c.metadata for c in chunks]
        )
        
        # Update document status
        doc.chunk_count = len(chunks)
        doc.status = "ready"
        db.commit()
        
        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "chunks_created": len(chunks),
            "file_size": file_size,
            "status": "success"
        }
        
    except Exception as e:
        # Update status to failed
        if 'doc' in locals():
            doc.status = "failed"
            db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/list")
async def list_documents(
    token: str,
    db: Session = Depends(get_db)
):
    """List all documents for the user."""
    user_id = get_user_from_token(token, db)
    
    documents = db.query(Document).filter(
        Document.user_id == user_id
    ).order_by(Document.created_at.desc()).all()
    
    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "chunk_count": doc.chunk_count,
                "status": doc.status,
                "created_at": doc.created_at.isoformat(),
                "tags": doc.metadata.get("tags", [])
            }
            for doc in documents
        ],
        "total": len(documents)
    }


@router.get("/{doc_id}")
async def get_document(
    doc_id: int,
    token: str,
    db: Session = Depends(get_db)
):
    """Get document details."""
    user_id = get_user_from_token(token, db)
    
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "file_size": doc.file_size,
        "chunk_count": doc.chunk_count,
        "status": doc.status,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat()
    }


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    token: str,
    db: Session = Depends(get_db)
):
    """Delete a document and its chunks."""
    user_id = get_user_from_token(token, db)
    
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == user_id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete chunks from Endee
    tenant_manager = get_tenant_manager(user_id)
    deleted_chunks = tenant_manager.delete_document_chunks(str(doc_id))
    
    # Delete from database
    db.delete(doc)
    db.commit()
    
    return {
        "message": "Document deleted successfully",
        "chunks_deleted": deleted_chunks
    }
