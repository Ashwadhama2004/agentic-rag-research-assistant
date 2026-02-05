"""
Session Management
Handles user sessions using Streamlit session state.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


@dataclass
class UserSession:
    """User session data."""
    user_id: int
    username: str
    access_token: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Manages user sessions using Streamlit session state."""
    
    @staticmethod
    def init_session():
        """Initialize session state variables."""
        if not HAS_STREAMLIT:
            return
        
        if "user_session" not in st.session_state:
            st.session_state.user_session = None
        
        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id = None
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    @staticmethod
    def login(user_id: int, username: str, access_token: str):
        """Set user session after login."""
        if not HAS_STREAMLIT:
            return
        
        st.session_state.user_session = UserSession(
            user_id=user_id,
            username=username,
            access_token=access_token
        )
    
    @staticmethod
    def logout():
        """Clear user session."""
        if not HAS_STREAMLIT:
            return
        
        st.session_state.user_session = None
        st.session_state.current_conversation_id = None
        st.session_state.messages = []
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated."""
        if not HAS_STREAMLIT:
            return False
        
        return st.session_state.get("user_session") is not None
    
    @staticmethod
    def get_current_user() -> Optional[UserSession]:
        """Get current user session."""
        if not HAS_STREAMLIT:
            return None
        
        return st.session_state.get("user_session")
    
    @staticmethod
    def get_user_id() -> Optional[int]:
        """Get current user ID."""
        session = SessionManager.get_current_user()
        return session.user_id if session else None
    
    @staticmethod
    def get_access_token() -> Optional[str]:
        """Get current access token."""
        session = SessionManager.get_current_user()
        return session.access_token if session else None
    
    @staticmethod
    def set_conversation(conversation_id: int):
        """Set current conversation."""
        if HAS_STREAMLIT:
            st.session_state.current_conversation_id = conversation_id
    
    @staticmethod
    def get_conversation_id() -> Optional[int]:
        """Get current conversation ID."""
        if not HAS_STREAMLIT:
            return None
        
        return st.session_state.get("current_conversation_id")
    
    @staticmethod
    def add_message(role: str, content: str, sources: list = None):
        """Add message to current conversation."""
        if HAS_STREAMLIT:
            st.session_state.messages.append({
                "role": role,
                "content": content,
                "sources": sources or []
            })
    
    @staticmethod
    def get_messages() -> list:
        """Get all messages in current conversation."""
        if not HAS_STREAMLIT:
            return []
        
        return st.session_state.get("messages", [])
    
    @staticmethod
    def clear_messages():
        """Clear all messages."""
        if HAS_STREAMLIT:
            st.session_state.messages = []


# Helper functions for easier import
def is_logged_in() -> bool:
    """Check if user is logged in."""
    SessionManager.init_session()
    return SessionManager.is_authenticated()


def get_current_session() -> Optional[UserSession]:
    """Get current user session."""
    SessionManager.init_session()
    return SessionManager.get_current_user()


def require_login():
    """Redirect to login if not authenticated."""
    if not HAS_STREAMLIT:
        return
    
    SessionManager.init_session()
    if not SessionManager.is_authenticated():
        st.warning("Please log in to access this page.")
        st.stop()


def login_user(user_id: int, username: str, access_token: str):
    """Log in a user."""
    SessionManager.init_session()
    SessionManager.login(user_id, username, access_token)


def logout_user():
    """Log out the current user."""
    SessionManager.logout()


def get_user_id() -> Optional[int]:
    """Get the current user ID."""
    return SessionManager.get_user_id()


def set_session(user_id: int, username: str, token: str = ""):
    """Set user session (alias for login_user)."""
    SessionManager.init_session()
    SessionManager.login(user_id, username, token or f"token_{user_id}")


# Alias for logout
logout = logout_user



