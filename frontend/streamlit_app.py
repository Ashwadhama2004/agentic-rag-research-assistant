"""
Agentic RAG Research Assistant - Main Streamlit Application
Multi-Tenant Agentic RAG Research & Knowledge Assistant
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.config import config
from backend.app.database.connection import init_database
from backend.app.auth.sessions import is_logged_in, get_current_session

# Import pages
from frontend.pages.login import render_login
from frontend.pages.dashboard import render_dashboard
from frontend.pages.documents import render_documents
from frontend.pages.chat import render_chat
from frontend.pages.agent import render_agent
from frontend.pages.metrics import render_metrics
from frontend.pages.settings import render_settings

# Page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding: 2rem;
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
    }
    
    .stMetric label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    h1 {
        color: #667eea;
    }
    
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Initialize database
    config.ensure_directories()
    init_database()
    
    # Check authentication
    if not is_logged_in():
        render_login()
        return
    
    # Get current session
    session = get_current_session()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("#  Research Assistant")
        st.markdown(f"*Logged in as: {session.username}*")
        st.markdown("---")
        
        # Navigation
        if "page" not in st.session_state:
            st.session_state.page = "Dashboard"
        
        pages = {
            " Dashboard": "Dashboard",
            " Documents": "Documents",
            " Chat": "Chat",
            " Agent": "Agent",
            " Metrics": "Metrics",
            " Settings": "Settings"
        }
        
        for label, page in pages.items():
            if st.button(label, use_container_width=True, 
                        type="primary" if st.session_state.page == page else "secondary"):
                st.session_state.page = page
                st.rerun()
        
        st.markdown("---")
        st.caption("v1.0.0 | Built with ")
    
    # Render selected page
    page = st.session_state.page
    
    if page == "Dashboard":
        render_dashboard()
    elif page == "Documents":
        render_documents()
    elif page == "Chat":
        render_chat()
    elif page == "Agent":
        render_agent()
    elif page == "Metrics":
        render_metrics()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
