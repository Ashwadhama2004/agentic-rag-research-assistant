"""
Database Schema
SQLAlchemy models for users, documents, conversations, and metrics.
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship

from backend.app.database.base import Base


class User(Base):
    """User model for multi-tenant authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    agent_tasks = relationship("AgentTask", back_populates="user", cascade="all, delete-orphan")
    
    @property
    def namespace(self) -> str:
        """Get user-specific namespace for Endee collections."""
        return f"user_{self.id}"


class Document(Base):
    """Document model for uploaded files."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    chunk_count = Column(Integer, default=0)
    extra_data = Column(JSON, default=dict)
    status = Column(String(50), default="processing")  # processing, ready, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="documents")


class Conversation(Base):
    """Conversation model for chat history."""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Message model for conversation messages."""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    sources = Column(JSON, default=list)  # List of source citations
    latency = Column(Float, nullable=True)  # Response time in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


class AgentTask(Base):
    """Agent task model for research tasks."""
    __tablename__ = "agent_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    task_description = Column(Text, nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    steps_completed = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="agent_tasks")
    steps = relationship("AgentStep", back_populates="task", cascade="all, delete-orphan")


class AgentStep(Base):
    """Agent step model for tracking agent execution."""
    __tablename__ = "agent_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("agent_tasks.id"), nullable=False)
    step_number = Column(Integer, nullable=False)
    action = Column(String(100), nullable=False)  # think, plan, act, reflect, respond
    tool_used = Column(String(100), nullable=True)
    input_data = Column(JSON, default=dict)
    output_data = Column(JSON, default=dict)
    status = Column(String(50), default="running")  # running, completed, failed
    latency = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("AgentTask", back_populates="steps")


class Metric(Base):
    """Metric model for performance tracking."""
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null for system metrics
    metric_type = Column(String(100), nullable=False)  # query_latency, retrieval_precision, etc.
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    extra_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class MemorySummary(Base):
    """Long-term memory summaries."""
    __tablename__ = "memory_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    summary_type = Column(String(50), nullable=False)  # conversation, research, document
    content = Column(Text, nullable=False)
    source_ids = Column(JSON, default=list)  # IDs of source conversations/documents
    created_at = Column(DateTime, default=datetime.utcnow)
