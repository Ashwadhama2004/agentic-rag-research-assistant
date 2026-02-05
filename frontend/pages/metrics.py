"""
Metrics Page
Performance metrics and analytics dashboard.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

from backend.app.auth.sessions import get_current_session, require_login
from backend.app.database.connection import get_db_session
from backend.app.database.schema import Message, Document, AgentTask, Conversation
from sqlalchemy import func


def render_metrics():
    """Render the metrics page."""
    require_login()
    session = get_current_session()
    
    st.title(" Performance Metrics")
    st.markdown("---")
    
    # Time period selector
    period = st.selectbox("Time Period", ["Last 7 days", "Last 30 days", "All time"])
    
    if period == "Last 7 days":
        start_date = datetime.utcnow() - timedelta(days=7)
    elif period == "Last 30 days":
        start_date = datetime.utcnow() - timedelta(days=30)
    else:
        start_date = datetime(2020, 1, 1)
    
    # Initialize variables
    latencies = []
    message_data = []
    doc_count = 0
    conv_count = 0
    task_count = 0
    completed_tasks = 0
    total_chunks = 0
    messages_count = 0
    
    with get_db_session() as db:
        # Query latency data - extract values inside context
        messages = db.query(Message).join(Conversation).filter(
            Conversation.user_id == session.user_id,
            Message.latency.isnot(None),
            Message.created_at >= start_date
        ).all()
        
        # Extract data inside the session context
        for m in messages:
            latencies.append(m.latency)
            message_data.append({
                "date": m.created_at.date(),
                "latency": m.latency
            })
        messages_count = len(messages)
        
        # Query usage data
        doc_count = db.query(func.count(Document.id)).filter(
            Document.user_id == session.user_id
        ).scalar() or 0
        
        conv_count = db.query(func.count(Conversation.id)).filter(
            Conversation.user_id == session.user_id
        ).scalar() or 0
        
        task_count = db.query(func.count(AgentTask.id)).filter(
            AgentTask.user_id == session.user_id
        ).scalar() or 0
        
        completed_tasks = db.query(func.count(AgentTask.id)).filter(
            AgentTask.user_id == session.user_id,
            AgentTask.status == "completed"
        ).scalar() or 0
        
        total_chunks = db.query(func.sum(Document.chunk_count)).filter(
            Document.user_id == session.user_id
        ).scalar() or 0
    
    # Summary metrics
    st.subheader(" Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        st.metric("Avg Latency", f"{avg_latency:.2f}s")
    
    with col2:
        st.metric("Total Queries", messages_count)
    
    with col3:
        success_rate = (completed_tasks / task_count * 100) if task_count > 0 else 0
        st.metric("Task Success Rate", f"{success_rate:.1f}%")
    
    with col4:
        st.metric("Vector Chunks", total_chunks)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Latency Distribution")
        if latencies:
            fig = px.histogram(
                x=latencies,
                nbins=20,
                labels={"x": "Latency (seconds)", "y": "Count"},
                color_discrete_sequence=["#667eea"]
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available")
    
    with col2:
        st.subheader(" Usage Breakdown")
        usage_data = {
            "Category": ["Documents", "Conversations", "Agent Tasks"],
            "Count": [doc_count, conv_count, task_count]
        }
        fig = px.pie(
            usage_data,
            values="Count",
            names="Category",
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"]
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Query trends
    st.subheader(" Query Trends")
    
    if message_data:
        # Group by date
        df = pd.DataFrame(message_data)
        
        daily_stats = df.groupby("date").agg({
            "latency": ["count", "mean"]
        }).reset_index()
        daily_stats.columns = ["date", "count", "avg_latency"]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_stats["date"],
            y=daily_stats["count"],
            name="Query Count",
            marker_color="#667eea"
        ))
        fig.add_trace(go.Scatter(
            x=daily_stats["date"],
            y=daily_stats["avg_latency"],
            name="Avg Latency",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#f093fb")
        ))
        
        fig.update_layout(
            yaxis=dict(title="Query Count"),
            yaxis2=dict(title="Avg Latency (s)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No query data available for the selected period")
    
    # Latency percentiles
    st.subheader(" Latency Percentiles")
    
    if latencies:
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Min", f"{min(latencies):.2f}s")
        with col2:
            st.metric("P50", f"{sorted_lat[int(n * 0.5)]:.2f}s")
        with col3:
            st.metric("P90", f"{sorted_lat[int(n * 0.9)]:.2f}s")
        with col4:
            st.metric("P99", f"{sorted_lat[min(int(n * 0.99), n - 1)]:.2f}s")
        with col5:
            st.metric("Max", f"{max(latencies):.2f}s")
    else:
        st.info("No latency data available")


if __name__ == "__main__":
    render_metrics()
