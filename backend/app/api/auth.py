"""
Authentication API Endpoints
Handles user registration, login, and session management.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional

from backend.app.database.connection import get_db
from backend.app.auth.users import (
    create_user, 
    get_user_by_username, 
    get_user_by_email,
    login_user,
    get_user_by_id,
    change_password
)
from backend.app.auth.security import decode_access_token
from backend.app.utils.validators import UserRegister, UserLogin, TokenResponse
from backend.app.endee_client.collections import create_user_collections

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if username exists
    if get_user_by_username(db, user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    if get_user_by_email(db, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = create_user(db, user_data.username, user_data.email, user_data.password)
    
    # Create user's Endee collections
    create_user_collections(user.id)
    
    # Log in user
    login_result = login_user(db, user_data.username, user_data.password)
    
    return TokenResponse(**login_result)


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with username and password."""
    result = login_user(db, credentials.username, credentials.password)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return TokenResponse(**result)


@router.post("/logout")
async def logout():
    """Logout (client should discard token)."""
    return {"message": "Successfully logged out"}


@router.get("/session")
async def get_session(token: str, db: Session = Depends(get_db)):
    """Get current session information."""
    payload = decode_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    user_id = payload.get("user_id")
    user = get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "user_id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat()
    }


@router.post("/change-password")
async def update_password(
    old_password: str,
    new_password: str,
    token: str,
    db: Session = Depends(get_db)
):
    """Change user password."""
    payload = decode_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    user_id = payload.get("user_id")
    
    success = change_password(db, user_id, old_password, new_password)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Old password is incorrect"
        )
    
    return {"message": "Password changed successfully"}
