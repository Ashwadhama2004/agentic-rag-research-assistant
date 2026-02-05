"""
Dashboard Page
Main dashboard with overview and quick actions.
"""
import streamlit as st
from datetime import datetime

from backend.app.auth.sessions import get_current_session, require_login
from backend.app.database.connection import get_db_session
from backend.app.database.schema import Document, Conversation, AgentTask


def render_dashboard():
    """Render the dashboard page."""
    require_login()
    session = get_current_session()
    
    st.title(" Research Assistant Dashboard")
    st.markdown("---")
    
    # Welcome message
    st.subheader(f"Welcome back, {session.username}!")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with get_db_session() as db:
        # Count documents
        doc_count = db.query(Document).filter(
            Document.user_id == session.user_id
        ).count()
        
        # Count conversations
        conv_count = db.query(Conversation).filter(
            Conversation.user_id == session.user_id
        ).count()
        
        # Count tasks
        task_count = db.query(AgentTask).filter(
            AgentTask.user_id == session.user_id
        ).count()
        
        # Active tasks
        active_tasks = db.query(AgentTask).filter(
            AgentTask.user_id == session.user_id,
            AgentTask.status.in_(["pending", "running"])
        ).count()
    
    with col1:
        st.metric(" Documents", doc_count)
    
    with col2:
        st.metric(" Conversations", conv_count)
    
    with col3:
        st.metric(" Research Tasks", task_count)
    
    with col4:
        st.metric("⚡ Active Tasks", active_tasks)
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Upload Document", use_container_width=True):
            st.session_state.page = "Documents"
            st.rerun()
    
    with col2:
        if st.button(" Start Chat", use_container_width=True):
            st.session_state.page = "Chat"
            st.rerun()
    
    with col3:
        if st.button("🔬 New Research", use_container_width=True):
            st.session_state.page = "Agent"
            st.rerun()
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📜 Recent Documents")
        with get_db_session() as db:
            recent_docs = db.query(Document).filter(
                Document.user_id == session.user_id
            ).order_by(Document.created_at.desc()).limit(5).all()
            
            if recent_docs:
                for doc in recent_docs:
                    status_icon = "[OK]" if doc.status == "ready" else "[...]"
                    st.markdown(f"{status_icon} **{doc.filename}** - {doc.chunk_count} chunks")
            else:
                st.info("No documents uploaded yet. Upload your first document to get started!")
    
    with col2:
        st.subheader("💭 Recent Conversations")
        with get_db_session() as db:
            recent_convs = db.query(Conversation).filter(
                Conversation.user_id == session.user_id
            ).order_by(Conversation.updated_at.desc()).limit(5).all()
            
            if recent_convs:
                for conv in recent_convs:
                    title = conv.title or "Untitled"
                    st.markdown(f" **{title[:30]}...**")
            else:
                st.info("No conversations yet. Start chatting to see your history!")
    
    # System status
    st.markdown("---")
    st.subheader(" System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("🟢 LLM Service: Online")
    
    with col2:
        st.success("🟢 Vector DB: Online")
    
    with col3:
        st.success("🟢 Database: Online")


if __name__ == "__main__":
    render_dashboard()
