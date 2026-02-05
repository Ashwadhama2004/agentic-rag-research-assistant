"""
Chat Page
RAG-powered chat interface.
"""
import streamlit as st
import time

from backend.app.auth.sessions import get_current_session, require_login
from backend.app.database.connection import get_db_session
from backend.app.database.schema import Conversation, Message
from backend.app.rag.retriever import create_retriever
from backend.app.rag.reranker import get_reranker
from backend.app.rag.generator import get_generator
from backend.app.memory.short_term import create_short_term_memory


def render_chat():
    """Render the chat page."""
    require_login()
    session = get_current_session()
    
    st.title(" Research Chat")
    
    # Initialize session state
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for conversation history
    with st.sidebar:
        st.subheader(" Conversations")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # List recent conversations
        with get_db_session() as db:
            conversations = db.query(Conversation).filter(
                Conversation.user_id == session.user_id
            ).order_by(Conversation.updated_at.desc()).limit(10).all()
            
            for conv in conversations:
                title = conv.title or "Untitled"
                title = title[:25] + "..." if len(title) > 25 else title
                
                if st.button(f" {title}", key=f"conv_{conv.id}", use_container_width=True):
                    # Load conversation
                    st.session_state.conversation_id = conv.id
                    messages = db.query(Message).filter(
                        Message.conversation_id == conv.id
                    ).order_by(Message.created_at).all()
                    st.session_state.messages = [
                        {"role": m.role, "content": m.content, "sources": m.sources}
                        for m in messages
                    ]
                    st.rerun()
    
    # Chat settings
    with st.expander(" Chat Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Retrieved Documents", 1, 10, 5)
        with col2:
            use_reranker = st.checkbox("Use Reranker", value=True)
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for source in msg["sources"]:
                        st.markdown(f"- **{source.get('document', 'Unknown')}** (Score: {source.get('relevance_score', 0):.2f})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                try:
                    # Retrieve relevant documents
                    retriever = create_retriever(session.user_id, top_k=top_k)
                    results = retriever.retrieve(prompt)
                    
                    # Rerank if enabled
                    if use_reranker and results:
                        reranker = get_reranker()
                        results = reranker.rerank(results, prompt, top_k=top_k)
                    
                    # Generate answer
                    generator = get_generator()
                    answer = generator.generate(prompt, results)
                    
                    latency = time.time() - start_time
                    
                    # Display answer
                    st.markdown(answer.answer)
                    
                    # Show sources
                    if answer.sources:
                        with st.expander("📚 Sources", expanded=False):
                            for source in answer.sources:
                                st.markdown(f"- **{source.get('document', 'Unknown')}** (Score: {source.get('relevance_score', 0):.2f})")
                    
                    # Show latency
                    st.caption(f"⏱️ {latency:.2f}s | Model: {answer.model}")
                    
                    # Save to database
                    with get_db_session() as db:
                        # Create or update conversation
                        if st.session_state.conversation_id:
                            conv = db.query(Conversation).filter(
                                Conversation.id == st.session_state.conversation_id
                            ).first()
                        else:
                            conv = Conversation(
                                user_id=session.user_id,
                                title=prompt[:50] + "..." if len(prompt) > 50 else prompt
                            )
                            db.add(conv)
                            db.commit()
                            db.refresh(conv)
                            st.session_state.conversation_id = conv.id
                        
                        # Save messages
                        user_msg = Message(
                            conversation_id=conv.id,
                            role="user",
                            content=prompt
                        )
                        assistant_msg = Message(
                            conversation_id=conv.id,
                            role="assistant",
                            content=answer.answer,
                            sources=answer.sources,
                            latency=latency
                        )
                        db.add(user_msg)
                        db.add(assistant_msg)
                        db.commit()
                    
                    # Update session messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer.answer,
                        "sources": answer.sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


if __name__ == "__main__":
    render_chat()
