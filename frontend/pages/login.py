"""
Login Page
User authentication interface.
"""
import streamlit as st

from backend.app.database.connection import get_db_session
from backend.app.auth.users import create_user, get_user_by_username, get_user_by_email, login_user
from backend.app.auth.sessions import set_session
from backend.app.endee_client.collections import create_user_collections


def render_login():
    """Render the login page."""
    st.title(" Research Assistant")
    st.markdown("#### Multi-Tenant Agentic RAG Knowledge Assistant")
    st.markdown("---")
    
    tab1, tab2 = st.tabs([" Login", "📝 Register"])
    
    with tab1:
        st.subheader("Welcome Back")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    with get_db_session() as db:
                        result = login_user(db, username, password)
                        
                        if result:
                            set_session(
                                user_id=result["user_id"],
                                username=result["username"],
                                token=result["access_token"]
                            )
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Create Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            submitted = st.form_submit_button("Register", type="primary", use_container_width=True)
            
            if submitted:
                # Validation
                errors = []
                
                if not all([new_username, new_email, new_password, confirm_password]):
                    errors.append("Please fill in all fields")
                
                if len(new_username) < 3:
                    errors.append("Username must be at least 3 characters")
                
                if "@" not in new_email:
                    errors.append("Please enter a valid email")
                
                if len(new_password) < 8:
                    errors.append("Password must be at least 8 characters")
                
                if new_password != confirm_password:
                    errors.append("Passwords do not match")
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    with get_db_session() as db:
                        # Check if username exists
                        if get_user_by_username(db, new_username):
                            st.error("Username already exists")
                        elif get_user_by_email(db, new_email):
                            st.error("Email already registered")
                        else:
                            # Create user
                            user = create_user(db, new_username, new_email, new_password)
                            
                            # Create user's Endee collections
                            create_user_collections(user.id)
                            
                            # Login
                            result = login_user(db, new_username, new_password)
                            
                            if result:
                                set_session(
                                    user_id=result["user_id"],
                                    username=result["username"],
                                    token=result["access_token"]
                                )
                                st.success("Account created successfully!")
                                st.rerun()


if __name__ == "__main__":
    render_login()
