"""
User Management
Handles user registration, authentication, and profile management.
"""
from typing import Optional
from sqlalchemy.orm import Session

from backend.app.database.schema import User
from backend.app.auth.security import get_password_hash, verify_password, create_access_token


def create_user(db: Session, username: str, email: str, password: str) -> User:
    """Create a new user."""
    hashed_password = get_password_hash(password)
    
    user = User(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user = get_user_by_username(db, username)
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user


def login_user(db: Session, username: str, password: str) -> Optional[dict]:
    """Authenticate user and return access token."""
    user = authenticate_user(db, username, password)
    
    if not user:
        return None
    
    access_token = create_access_token(data={"user_id": user.id, "username": user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }


def update_user(db: Session, user_id: int, **kwargs) -> Optional[User]:
    """Update user profile."""
    user = get_user_by_id(db, user_id)
    
    if not user:
        return None
    
    for key, value in kwargs.items():
        if hasattr(user, key) and key != "hashed_password":
            setattr(user, key, value)
    
    db.commit()
    db.refresh(user)
    
    return user


def delete_user(db: Session, user_id: int) -> bool:
    """Delete user and all associated data."""
    user = get_user_by_id(db, user_id)
    
    if not user:
        return False
    
    db.delete(user)
    db.commit()
    
    return True


def change_password(db: Session, user_id: int, old_password: str, new_password: str) -> bool:
    """Change user password."""
    user = get_user_by_id(db, user_id)
    
    if not user:
        return False
    
    if not verify_password(old_password, user.hashed_password):
        return False
    
    user.hashed_password = get_password_hash(new_password)
    db.commit()
    
    return True
