"""
Documents Page
Document upload and management interface.
"""
import streamlit as st
import time

from backend.app.auth.sessions import get_current_session, require_login
from backend.app.database.connection import get_db_session
from backend.app.database.schema import Document
from backend.app.utils.parsers import get_document_parser
from backend.app.rag.chunker import DocumentChunker
from backend.app.rag.embedder import get_embedder
from backend.app.endee_client.multi_tenant import get_tenant_manager
from backend.app.config import config
from backend.app.utils.helpers import format_bytes


def render_documents():
    """Render the documents page."""
    require_login()
    session = get_current_session()
    
    st.title(" Document Management")
    st.markdown("---")
    
    # Document upload section
    st.subheader(" Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md"],
        help="Supported formats: PDF, DOCX, TXT, Markdown"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        tags = st.text_input("Tags (comma-separated)", placeholder="research, notes, important")
    
    with col2:
        category = st.selectbox("Category", ["General", "Research", "Notes", "Reference", "Other"])
    
    if uploaded_file and st.button(" Process Document", type="primary"):
        with st.spinner("Processing document..."):
            try:
                # Read file content
                content = uploaded_file.read()
                file_size = len(content)
                
                # Check file size
                if file_size > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    st.error(f"File too large. Maximum size: {config.MAX_UPLOAD_SIZE_MB}MB")
                    return
                
                # Parse document
                parser = get_document_parser()
                pages, metadata = parser.parse_pages(content, uploaded_file.name)
                
                with get_db_session() as db:
                    # Create document record
                    doc = Document(
                        user_id=session.user_id,
                        filename=uploaded_file.name,
                        file_type=metadata.get("file_type", "unknown"),
                        file_size=file_size,
                        extra_data={
                            **metadata,
                            "tags": [t.strip() for t in tags.split(",") if t.strip()],
                            "category": category
                        },
                        status="processing"
                    )
                    db.add(doc)
                    db.commit()
                    db.refresh(doc)
                    
                    # Progress bar
                    progress = st.progress(0, text="Chunking document...")
                    
                    # Chunk document
                    chunker = DocumentChunker()
                    chunks = chunker.chunk_pages(
                        pages=pages,
                        document_id=str(doc.id),
                        document_name=uploaded_file.name
                    )
                    
                    progress.progress(30, text="Generating embeddings...")
                    
                    # Generate embeddings
                    embedder = get_embedder()
                    chunk_contents = [c.content for c in chunks]
                    embeddings = embedder.encode_documents(chunk_contents)
                    
                    progress.progress(70, text="Storing in vector database...")
                    
                    # Store in Endee
                    tenant_manager = get_tenant_manager(session.user_id)
                    tenant_manager.add_document_chunks(
                        chunk_ids=[c.id for c in chunks],
                        embeddings=list(embeddings),
                        contents=chunk_contents,
                        metadatas=[c.metadata for c in chunks]
                    )
                    
                    # Update document status
                    doc.chunk_count = len(chunks)
                    doc.status = "ready"
                    db.commit()
                    
                    progress.progress(100, text="Complete!")
                    time.sleep(0.5)
                    progress.empty()
                
                st.success(f"✅ Successfully processed '{uploaded_file.name}' - {len(chunks)} chunks created")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
    
    st.markdown("---")
    
    # Document list
    st.subheader(" Your Documents")
    
    # Filters
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(" Search documents", placeholder="Search by filename...")
    with col2:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Name", "Size"])
    
    with get_db_session() as db:
        # Query documents
        query = db.query(Document).filter(Document.user_id == session.user_id)
        
        if search_query:
            query = query.filter(Document.filename.ilike(f"%{search_query}%"))
        
        # Apply sorting
        if sort_by == "Newest":
            query = query.order_by(Document.created_at.desc())
        elif sort_by == "Oldest":
            query = query.order_by(Document.created_at.asc())
        elif sort_by == "Name":
            query = query.order_by(Document.filename.asc())
        elif sort_by == "Size":
            query = query.order_by(Document.file_size.desc())
        
        documents = query.all()
        
        if documents:
            for doc in documents:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        status_icon = "[OK]" if doc.status == "ready" else "[...]" if doc.status == "processing" else "[X]"
                        st.markdown(f"**{status_icon} {doc.filename}**")
                        tags_list = []
                        if doc.extra_data and isinstance(doc.extra_data, dict):
                            tags_list = doc.extra_data.get("tags", [])
                        tags_str = ", ".join(tags_list) if tags_list else ""
                        if tags_str:
                            st.caption(f"Tags: {tags_str}")
                    
                    with col2:
                        st.metric("Chunks", doc.chunk_count)
                    
                    with col3:
                        st.caption(format_bytes(doc.file_size))
                        st.caption(doc.created_at.strftime("%Y-%m-%d"))
                    
                    with col4:
                        if st.button("Delete", key=f"delete_{doc.id}", help="Delete document"):
                            # Delete chunks from Endee
                            tenant_manager = get_tenant_manager(session.user_id)
                            tenant_manager.delete_document_chunks(str(doc.id))
                            # Delete from database
                            db.delete(doc)
                            db.commit()
                            st.success("Document deleted!")
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No documents found. Upload your first document above!")


if __name__ == "__main__":
    render_documents()
