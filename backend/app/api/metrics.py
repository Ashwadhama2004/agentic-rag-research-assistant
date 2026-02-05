"""
Metrics API Endpoints
Provides system and user performance metrics.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from datetime import datetime, timedelta

from backend.app.database.connection import get_db
from backend.app.database.schema import Metric, Message, Document, AgentTask, Conversation
from backend.app.auth.security import decode_access_token
from backend.app.endee_client.collections import get_collection_stats

router = APIRouter(prefix="/api/metrics", tags=["Metrics"])


def get_user_from_token(token: str) -> int:
    """Extract and validate user ID from token."""
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    return payload.get("user_id")


@router.get("/system")
async def get_system_metrics(db: Session = Depends(get_db)):
    """Get system-wide metrics."""
    # Get counts
    total_documents = db.query(func.count(Document.id)).scalar()
    total_messages = db.query(func.count(Message.id)).scalar()
    total_tasks = db.query(func.count(AgentTask.id)).scalar()
    
    # Get average latency from recent messages
    recent_messages = db.query(Message).filter(
        Message.latency.isnot(None),
        Message.created_at >= datetime.utcnow() - timedelta(days=1)
    ).all()
    
    avg_latency = 0
    if recent_messages:
        avg_latency = sum(m.latency for m in recent_messages) / len(recent_messages)
    
    # Task success rate
    completed_tasks = db.query(func.count(AgentTask.id)).filter(
        AgentTask.status == "completed"
    ).scalar()
    
    task_success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
    
    return {
        "total_documents": total_documents,
        "total_messages": total_messages,
        "total_tasks": total_tasks,
        "avg_latency_24h": round(avg_latency, 3),
        "task_success_rate": round(task_success_rate, 3),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/user")
async def get_user_metrics(
    token: str,
    db: Session = Depends(get_db)
):
    """Get metrics for current user."""
    user_id = get_user_from_token(token)
    
    # Document count
    doc_count = db.query(func.count(Document.id)).filter(
        Document.user_id == user_id
    ).scalar()
    
    # Conversation count
    conv_count = db.query(func.count(Conversation.id)).filter(
        Conversation.user_id == user_id
    ).scalar()
    
    # Message count
    msg_count = db.query(func.count(Message.id)).join(Conversation).filter(
        Conversation.user_id == user_id
    ).scalar()
    
    # Task count
    task_count = db.query(func.count(AgentTask.id)).filter(
        AgentTask.user_id == user_id
    ).scalar()
    
    # Vector collection stats
    collection_stats = get_collection_stats(user_id)
    
    # Calculate latency stats
    user_messages = db.query(Message).join(Conversation).filter(
        Conversation.user_id == user_id,
        Message.latency.isnot(None)
    ).order_by(Message.created_at.desc()).limit(100).all()
    
    latencies = [m.latency for m in user_messages]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    return {
        "documents": doc_count,
        "conversations": conv_count,
        "messages": msg_count,
        "tasks": task_count,
        "vector_chunks": collection_stats,
        "avg_latency": round(avg_latency, 3),
        "recent_queries": len(user_messages)
    }


@router.get("/performance")
async def get_performance_metrics(
    token: str,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get detailed performance metrics."""
    user_id = get_user_from_token(token)
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get messages with latency
    messages = db.query(Message).join(Conversation).filter(
        Conversation.user_id == user_id,
        Message.latency.isnot(None),
        Message.created_at >= start_date
    ).all()
    
    latencies = [m.latency for m in messages]
    
    # Calculate percentiles
    if latencies:
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        p50 = sorted_latencies[int(n * 0.5)]
        p95 = sorted_latencies[int(n * 0.95)]
        p99 = sorted_latencies[min(int(n * 0.99), n - 1)]
        avg = sum(latencies) / n
        max_lat = max(latencies)
    else:
        p50 = p95 = p99 = avg = max_lat = 0
    
    # Get task metrics
    tasks = db.query(AgentTask).filter(
        AgentTask.user_id == user_id,
        AgentTask.created_at >= start_date
    ).all()
    
    completed = sum(1 for t in tasks if t.status == "completed")
    failed = sum(1 for t in tasks if t.status == "failed")
    
    # Daily query counts
    daily_counts = {}
    for msg in messages:
        date_key = msg.created_at.strftime("%Y-%m-%d")
        daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
    
    return {
        "latency": {
            "avg": round(avg, 3),
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "max": round(max_lat, 3)
        },
        "queries": {
            "total": len(messages),
            "daily": daily_counts
        },
        "tasks": {
            "total": len(tasks),
            "completed": completed,
            "failed": failed,
            "success_rate": round(completed / len(tasks), 3) if tasks else 0
        },
        "period_days": days
    }
