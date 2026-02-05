"""
Chat API Endpoints
Handles chat queries and conversation management.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import time

from backend.app.database.connection import get_db
from backend.app.database.schema import Conversation, Message
from backend.app.auth.security import decode_access_token
from backend.app.rag.retriever import create_retriever
from backend.app.rag.reranker import get_reranker
from backend.app.rag.generator import get_generator
from backend.app.memory.short_term import create_short_term_memory
from backend.app.utils.validators import ChatQuery

router = APIRouter(prefix="/api/chat", tags=["Chat"])


def get_user_from_token(token: str) -> int:
    """Extract and validate user ID from token."""
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    return payload.get("user_id")


@router.post("/query")
async def query(
    request: ChatQuery,
    token: str,
    db: Session = Depends(get_db)
):
    """Process a RAG query."""
    user_id = get_user_from_token(token)
    start_time = time.time()
    
    try:
        # Create or get conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation = Conversation(
                user_id=user_id,
                title=request.query[:50] + "..." if len(request.query) > 50 else request.query
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            conversation_id = conversation.id
        
        # Save user message
        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=request.query
        )
        db.add(user_message)
        db.commit()
        
        # Retrieve relevant documents
        retriever = create_retriever(user_id, top_k=request.top_k)
        results = retriever.retrieve(request.query)
        
        # Rerank results
        reranker = get_reranker()
        reranked = reranker.rerank(results, request.query, top_k=request.top_k)
        
        # Generate answer
        generator = get_generator()
        answer = generator.generate(request.query, reranked)
        
        latency = time.time() - start_time
        
        # Save assistant message
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer.answer,
            sources=answer.sources,
            latency=latency
        )
        db.add(assistant_message)
        db.commit()
        
        return {
            "answer": answer.answer,
            "sources": answer.sources,
            "latency": round(latency, 3),
            "model": answer.model,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/conversation")
async def conversation_chat(
    message: str,
    conversation_id: Optional[int] = None,
    token: str = "",
    db: Session = Depends(get_db)
):
    """Continue a conversation."""
    user_id = get_user_from_token(token)
    start_time = time.time()
    
    # Get or create conversation
    if conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    else:
        conversation = Conversation(user_id=user_id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    
    # Get conversation history
    history = db.query(Message).filter(
        Message.conversation_id == conversation.id
    ).order_by(Message.created_at).all()
    
    history_list = [
        {"role": m.role, "content": m.content}
        for m in history[-10:]  # Last 10 messages
    ]
    
    # Save user message
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=message
    )
    db.add(user_msg)
    db.commit()
    
    # Generate response with context
    generator = get_generator()
    
    # Also retrieve relevant documents
    retriever = create_retriever(user_id)
    results = retriever.retrieve(message, top_k=3)
    
    if results:
        # Use RAG mode with context
        answer = generator.generate(message, results)
    else:
        # Pure conversation mode
        answer = generator.generate_conversation(message, history_list)
    
    latency = time.time() - start_time
    
    # Save assistant message
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer.answer,
        sources=answer.sources,
        latency=latency
    )
    db.add(assistant_msg)
    db.commit()
    
    return {
        "response": answer.answer,
        "sources": answer.sources,
        "conversation_id": conversation.id,
        "latency": round(latency, 3)
    }


@router.get("/history")
async def get_history(
    token: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get conversation history."""
    user_id = get_user_from_token(token)
    
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).limit(limit).all()
    
    result = []
    for conv in conversations:
        messages = db.query(Message).filter(
            Message.conversation_id == conv.id
        ).order_by(Message.created_at).all()
        
        result.append({
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": len(messages),
            "last_message": messages[-1].content[:100] if messages else None
        })
    
    return {"conversations": result, "total": len(result)}


@router.get("/conversation/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    token: str,
    db: Session = Depends(get_db)
):
    """Get full conversation with messages."""
    user_id = get_user_from_token(token)
    
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at).all()
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "created_at": m.created_at.isoformat()
            }
            for m in messages
        ]
    }
