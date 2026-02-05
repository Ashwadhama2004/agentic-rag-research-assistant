"""
Database Connection Management
Handles SQLite database connections and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os

from backend.app.config import config
from backend.app.database.base import Base

# Ensure data directory exists
db_path = config.DATABASE_URL.replace("sqlite:///", "")
if db_path.startswith("./"):
    db_path = db_path[2:]
db_dir = os.path.dirname(db_path) if os.path.dirname(db_path) else "data"
os.makedirs(db_dir, exist_ok=True)

# Create SQLAlchemy engine
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_database():
    """Initialize database by creating all tables."""
    from backend.app.database import schema
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


def drop_database():
    """Drop all tables from database."""
    from backend.app.database import schema
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped!")


# Alias for compatibility
get_db_session = get_db_context
