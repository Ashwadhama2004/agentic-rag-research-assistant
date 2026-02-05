"""
Database Base Module
Centralized SQLAlchemy Base class to prevent circular imports.
"""
from sqlalchemy.orm import declarative_base

# Create a single Base class for all models
Base = declarative_base()
