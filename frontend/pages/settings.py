"""
Settings Page
User settings and configuration.
"""
import streamlit as st

from backend.app.auth.sessions import get_current_session, require_login, logout
from backend.app.database.connection import get_db_session
from backend.app.auth.users import change_password, get_user_by_id
from backend.app.config import config


def render_settings():
    """Render the settings page."""
    require_login()
    session = get_current_session()
    
    st.title(" Settings")
    st.markdown("---")
    
    # User profile section
    st.subheader(" Profile")
    
    with get_db_session() as db:
        user = get_user_by_id(db, session.user_id)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Username", value=user.username, disabled=True)
            st.text_input("Email", value=user.email, disabled=True)
        
        with col2:
            st.text_input("User ID", value=str(user.id), disabled=True)
            st.text_input("Created", value=user.created_at.strftime("%Y-%m-%d"), disabled=True)
    
    st.markdown("---")
    
    # Password change section
    st.subheader(" Change Password")
    
    with st.form("password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        submitted = st.form_submit_button("Update Password")
        
        if submitted:
            if not all([current_password, new_password, confirm_password]):
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("New passwords do not match")
            elif len(new_password) < 8:
                st.error("New password must be at least 8 characters")
            else:
                with get_db_session() as db:
                    success = change_password(db, session.user_id, current_password, new_password)
                    if success:
                        st.success("Password updated successfully!")
                    else:
                        st.error("Current password is incorrect")
    
    st.markdown("---")
    
    # LLM Configuration
    st.subheader(" LLM Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Provider", value=config.LLM_PROVIDER, disabled=True)
        st.text_input("Model", value=config.LLM_MODEL, disabled=True)
    
    with col2:
        st.text_input("Embedding Model", value=config.EMBEDDING_MODEL, disabled=True)
        st.text_input("Chunk Size", value=str(config.CHUNK_SIZE), disabled=True)
    
    st.info("To change LLM settings, update the .env file and restart the application.")
    
    st.markdown("---")
    
    # Data management
    st.subheader("📦 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Export your data")
        if st.button("📥 Export Data"):
            st.info("Data export feature coming soon!")
    
    with col2:
        st.caption("Clear conversation history")
        if st.button(" Clear History", type="secondary"):
            with get_db_session() as db:
                from backend.app.database.schema import Conversation, Message
                
                # Get user's conversations
                convs = db.query(Conversation).filter(
                    Conversation.user_id == session.user_id
                ).all()
                
                for conv in convs:
                    db.delete(conv)
                
                db.commit()
            
            st.success("Conversation history cleared!")
            st.rerun()
    
    st.markdown("---")
    
    # About section
    st.subheader("ℹ️ About")
    
    st.markdown("""
    **Agentic RAG Research Assistant v1.0.0**
    
    A multi-tenant research assistant that combines:
    -  Document processing and semantic search
    - 🔍 RAG-powered question answering
    -  Autonomous research agent
    -  Performance analytics
    
    Built with Streamlit, FastAPI, and Endee Vector Database.
    """)
    
    st.markdown("---")
    
    # Logout
    st.subheader("🚪 Session")
    
    if st.button("Logout", type="primary"):
        logout()
        st.success("Logged out successfully!")
        st.rerun()


if __name__ == "__main__":
    render_settings()
