"""
Agent Page
Autonomous research agent interface.
"""
import streamlit as st
import asyncio
import time

from backend.app.auth.sessions import get_current_session, require_login
from backend.app.database.connection import get_db_session
from backend.app.database.schema import AgentTask, AgentStep
from backend.app.agents.executor import ResearchAgent


def format_finding(finding):
    """Format a finding into human-readable text."""
    if isinstance(finding, str):
        return finding
    
    if isinstance(finding, dict):
        # Handle different types of findings
        if "analysis" in finding:
            return finding["analysis"]
        elif "comparison" in finding:
            return finding["comparison"]
        elif "documents" in finding:
            # Format document search results
            docs = finding.get("documents", [])
            if not docs:
                return "No relevant documents found."
            
            formatted = []
            for i, doc in enumerate(docs, 1):
                content = doc.get("content", "")
                # Clean up OCR artifacts
                content = content.replace("\n", " ").replace("  ", " ").strip()
                doc_name = doc.get("document", "Unknown")
                score = doc.get("score", 0)
                formatted.append(f"**Source {i}** ({doc_name}, relevance: {score:.1%}):\n{content[:300]}...")
            
            return "\n\n".join(formatted)
        elif "results" in finding:
            # Vector search results
            results = finding.get("results", [])
            if not results:
                return "No relevant results found."
            
            formatted = []
            for i, r in enumerate(results, 1):
                content = r.get("content", "").replace("\n", " ").strip()
                doc = r.get("document", "Unknown")
                formatted.append(f"**Result {i}** ({doc}):\n{content[:250]}...")
            
            return "\n\n".join(formatted)
        elif "summary" in finding:
            return finding["summary"]
        else:
            # Generic dict - extract meaningful content
            if "output" in finding:
                return format_finding(finding["output"])
            return str(finding)
    
    return str(finding)


def generate_summary(findings, query):
    """Generate a human-readable summary from findings."""
    if not findings:
        return "No findings were collected during the research."
    
    # Collect all text content
    all_text = []
    for f in findings:
        formatted = format_finding(f)
        if formatted and len(formatted) > 20:
            all_text.append(formatted)
    
    if not all_text:
        return "The research completed but no substantial content was found."
    
    # Create summary header
    summary = f"### Research Summary for: {query}\n\n"
    summary += f"Found {len(all_text)} relevant pieces of information:\n\n"
    
    for i, text in enumerate(all_text[:5], 1):  # Limit to top 5
        summary += f"---\n\n**Finding {i}:**\n{text[:500]}\n\n"
    
    if len(all_text) > 5:
        summary += f"\n*...and {len(all_text) - 5} more findings in the detailed view below.*"
    
    return summary


def render_agent():
    """Render the agent page."""
    require_login()
    session = get_current_session()
    
    st.title("Research Agent")
    st.markdown("Let the AI autonomously research topics using your documents.")
    st.markdown("---")
    
    # New research task
    st.subheader("New Research Task")
    
    research_query = st.text_area(
        "Research Task",
        placeholder="Describe what you want to research. The agent will search your documents, analyze findings, and provide a comprehensive response.",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_steps = st.slider("Max Steps", 3, 15, 10)
    with col2:
        timeout = st.slider("Timeout (seconds)", 60, 300, 180)
    
    if st.button("Start Research", type="primary", disabled=not research_query):
        with st.spinner("Running research agent..."):
            start_time = time.time()
            
            # Create task record
            with get_db_session() as db:
                task = AgentTask(
                    user_id=session.user_id,
                    task_description=research_query,
                    status="running",
                    total_steps=max_steps
                )
                db.add(task)
                db.commit()
                db.refresh(task)
                task_id = task.id
            
            try:
                # Run agent
                agent = ResearchAgent(session.user_id)
                result = asyncio.run(agent.research(research_query, max_steps=max_steps))
                
                elapsed = time.time() - start_time
                
                # Update task
                with get_db_session() as db:
                    task = db.query(AgentTask).filter(AgentTask.id == task_id).first()
                    task.status = "completed"
                    task.result = str(result.get("findings", []))
                    task.steps_completed = len(result.get("results", []))
                    db.commit()
                
                # Display results
                st.success(f"Research completed in {elapsed:.2f}s")
                
                # Show summary (human-readable)
                findings = result.get("findings", [])
                st.subheader("Summary")
                summary = generate_summary(findings, research_query)
                st.markdown(summary)
                
                # Show detailed findings in expander
                st.subheader("Detailed Findings")
                if findings:
                    for i, finding in enumerate(findings, 1):
                        with st.expander(f"Finding {i}", expanded=False):
                            formatted = format_finding(finding)
                            st.markdown(formatted)
                else:
                    st.info("No specific findings recorded.")
                
                # Show execution trace in expander
                with st.expander("Execution Trace (Technical Details)"):
                    results = result.get("results", [])
                    for i, step_result in enumerate(results, 1):
                        status = "OK" if step_result.get("success") else "Failed"
                        st.markdown(f"**Step {i}** [{status}]")
                        if step_result.get("error"):
                            st.error(step_result["error"])
                        st.json(step_result)
                
            except Exception as e:
                # Update task status
                with get_db_session() as db:
                    task = db.query(AgentTask).filter(AgentTask.id == task_id).first()
                    task.status = "failed"
                    task.error = str(e)
                    db.commit()
                
                st.error(f"Research failed: {str(e)}")
    
    st.markdown("---")
    
    # Past research tasks
    st.subheader("Past Research Tasks")
    
    with get_db_session() as db:
        tasks = db.query(AgentTask).filter(
            AgentTask.user_id == session.user_id
        ).order_by(AgentTask.created_at.desc()).limit(10).all()
        
        # Extract data inside session
        task_data = []
        for t in tasks:
            task_data.append({
                "id": t.id,
                "description": t.task_description,
                "status": t.status,
                "steps_completed": t.steps_completed,
                "total_steps": t.total_steps,
                "result": t.result,
                "error": t.error,
                "created_at": t.created_at.strftime('%Y-%m-%d %H:%M') if t.created_at else ""
            })
    
    if task_data:
        for task in task_data:
            status_icon = "[OK]" if task["status"] == "completed" else "[...]" if task["status"] in ["pending", "running"] else "[X]"
            
            with st.expander(f"{status_icon} {task['description'][:50]}..."):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Status", task["status"].title())
                with col2:
                    st.metric("Steps", f"{task['steps_completed']}/{task['total_steps']}")
                with col3:
                    st.caption(f"Created: {task['created_at']}")
                
                if task["result"]:
                    st.markdown("**Result:**")
                    result_text = task["result"]
                    st.write(result_text[:500] + "..." if len(result_text) > 500 else result_text)
                
                if task["error"]:
                    st.error(f"Error: {task['error']}")
    else:
        st.info("No research tasks yet. Start your first research above!")


if __name__ == "__main__":
    render_agent()
