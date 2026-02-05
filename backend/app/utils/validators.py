"""
Input Validators
Validates user input and request data.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, EmailStr
import re


class UserRegister(BaseModel):
    """User registration schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username must contain only alphanumeric characters and underscores')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """User login schema."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class DocumentUpload(BaseModel):
    """Document upload metadata schema."""
    filename: str
    tags: Optional[List[str]] = []
    category: Optional[str] = None
    description: Optional[str] = None


class ChatQuery(BaseModel):
    """Chat query schema."""
    query: str = Field(..., min_length=1, max_length=5000)
    mode: str = Field(default="rag", pattern="^(rag|chat|agent)$")
    top_k: int = Field(default=5, ge=1, le=20)
    conversation_id: Optional[int] = None
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class AgentTask(BaseModel):
    """Agent task schema."""
    task: str = Field(..., min_length=10, max_length=2000)
    include_external: bool = Field(default=False)
    max_steps: int = Field(default=10, ge=1, le=20)
    timeout_seconds: int = Field(default=300, ge=60, le=600)


class MemoryConsolidation(BaseModel):
    """Memory consolidation request schema."""
    older_than_days: int = Field(default=7, ge=1, le=90)
    max_consolidations: int = Field(default=10, ge=1, le=50)


# Response schemas

class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str


class DocumentResponse(BaseModel):
    """Document response schema."""
    id: int
    filename: str
    file_type: str
    chunk_count: int
    status: str
    created_at: str


class MessageResponse(BaseModel):
    """Chat message response."""
    role: str
    content: str
    sources: List[Dict[str, Any]] = []
    latency: Optional[float] = None


class QueryResponse(BaseModel):
    """Query response schema."""
    answer: str
    sources: List[Dict[str, Any]]
    latency: float
    model: str


class AgentTaskResponse(BaseModel):
    """Agent task response schema."""
    task_id: str
    status: str
    steps_completed: int
    total_steps: int
    result: Optional[str] = None


class MetricsResponse(BaseModel):
    """Metrics response schema."""
    avg_latency: float
    total_queries: int
    success_rate: float
    active_users: int


# Validation utilities

def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if file extension is allowed."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return f'.{ext}' in allowed_extensions or ext in [e.lstrip('.') for e in allowed_extensions]


def validate_file_size(size_bytes: int, max_size_mb: int) -> bool:
    """Check if file size is within limits."""
    return size_bytes <= max_size_mb * 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')
    # Remove null bytes
    filename = filename.replace('\x00', '')
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = f"{name[:250]}.{ext}" if ext else name[:255]
    return filename
