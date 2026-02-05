"""
Agent API Endpoints
Handles agent task creation and monitoring.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import asyncio

from backend.app.database.connection import get_db
from backend.app.database.schema import AgentTask as AgentTaskModel, AgentStep as AgentStepModel
from backend.app.auth.security import decode_access_token
from backend.app.agents.executor import ResearchAgent
from backend.app.utils.validators import AgentTask

router = APIRouter(prefix="/api/agent", tags=["Agent"])


def get_user_from_token(token: str) -> int:
    """Extract and validate user ID from token."""
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    return payload.get("user_id")


async def run_agent_task(task_id: int, user_id: int, description: str, db_url: str):
    """Background task to run agent."""
    from backend.app.database.connection import SessionLocal
    
    db = SessionLocal()
    try:
        # Get task
        task = db.query(AgentTaskModel).filter(AgentTaskModel.id == task_id).first()
        if not task:
            return
        
        task.status = "running"
        db.commit()
        
        # Run agent
        agent = ResearchAgent(user_id)
        result = await agent.research(description)
        
        # Update task with results
        task.status = "completed"
        task.result = str(result.get("findings", []))
        task.steps_completed = len(result.get("results", []))
        task.total_steps = len(result.get("plan", []))
        task.completed_at = datetime.utcnow()
        
        # Save steps
        for i, step_result in enumerate(result.get("results", [])):
            step = AgentStepModel(
                task_id=task_id,
                step_number=i + 1,
                action="execute",
                status="completed" if step_result.get("success") else "failed",
                output_data=step_result
            )
            db.add(step)
        
        db.commit()
        
    except Exception as e:
        task = db.query(AgentTaskModel).filter(AgentTaskModel.id == task_id).first()
        if task:
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/research")
async def create_research_task(
    request: AgentTask,
    token: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new research task."""
    user_id = get_user_from_token(token)
    
    # Create task record
    task = AgentTaskModel(
        user_id=user_id,
        task_description=request.task,
        status="pending",
        total_steps=request.max_steps
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # Start background task
    from backend.app.config import config
    background_tasks.add_task(
        run_agent_task,
        task.id,
        user_id,
        request.task,
        config.DATABASE_URL
    )
    
    return {
        "task_id": task.id,
        "status": "pending",
        "message": "Research task created and queued"
    }


@router.get("/tasks")
async def list_tasks(
    token: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """List user's agent tasks."""
    user_id = get_user_from_token(token)
    
    tasks = db.query(AgentTaskModel).filter(
        AgentTaskModel.user_id == user_id
    ).order_by(AgentTaskModel.created_at.desc()).limit(limit).all()
    
    return {
        "tasks": [
            {
                "id": t.id,
                "description": t.task_description[:100],
                "status": t.status,
                "steps_completed": t.steps_completed,
                "total_steps": t.total_steps,
                "created_at": t.created_at.isoformat(),
                "completed_at": t.completed_at.isoformat() if t.completed_at else None
            }
            for t in tasks
        ]
    }


@router.get("/tasks/{task_id}/status")
async def get_task_status(
    task_id: int,
    token: str,
    db: Session = Depends(get_db)
):
    """Get task status."""
    user_id = get_user_from_token(token)
    
    task = db.query(AgentTaskModel).filter(
        AgentTaskModel.id == task_id,
        AgentTaskModel.user_id == user_id
    ).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return {
        "task_id": task.id,
        "status": task.status,
        "steps_completed": task.steps_completed,
        "total_steps": task.total_steps,
        "result": task.result,
        "error": task.error
    }


@router.get("/tasks/{task_id}/trace")
async def get_task_trace(
    task_id: int,
    token: str,
    db: Session = Depends(get_db)
):
    """Get task execution trace."""
    user_id = get_user_from_token(token)
    
    task = db.query(AgentTaskModel).filter(
        AgentTaskModel.id == task_id,
        AgentTaskModel.user_id == user_id
    ).first()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    steps = db.query(AgentStepModel).filter(
        AgentStepModel.task_id == task_id
    ).order_by(AgentStepModel.step_number).all()
    
    return {
        "task_id": task.id,
        "description": task.task_description,
        "status": task.status,
        "steps": [
            {
                "step_number": s.step_number,
                "action": s.action,
                "tool_used": s.tool_used,
                "status": s.status,
                "output": s.output_data
            }
            for s in steps
        ],
        "result": task.result
    }
