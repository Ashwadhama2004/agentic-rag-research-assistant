"""
Agent Memory Manager
Manages agent-specific memory for reasoning chains and task history.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from backend.app.rag.embedder import get_embedder
from backend.app.endee_client.multi_tenant import get_tenant_manager


class AgentMemoryManager:
    """
    Manages memory specific to agent operations.
    Stores reasoning chains, task history, and learned patterns.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.embedder = get_embedder()
        self.tenant_manager = get_tenant_manager(user_id)
        
        # In-memory cache for current session
        self.current_session: Dict[str, Any] = {
            "reasoning_chain": [],
            "tool_usage": [],
            "findings": []
        }
    
    def add_reasoning_step(
        self,
        step_type: str,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Add a step to the current reasoning chain.
        
        Args:
            step_type: Type of reasoning (think, plan, act, reflect)
            content: Content of the reasoning step
            metadata: Additional metadata
        """
        step = {
            "id": f"reason_{uuid.uuid4().hex[:8]}",
            "type": step_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.current_session["reasoning_chain"].append(step)
        
        # Store in vector database for future reference
        embedding = self.embedder.encode_single(content)
        self.tenant_manager.add_agent_step(
            step_id=step["id"],
            embedding=embedding,
            content=content,
            metadata={
                "step_type": step_type,
                "user_id": self.user_id,
                **step
            }
        )
    
    def record_tool_usage(
        self,
        tool_name: str,
        input_data: Any,
        output_data: Any,
        success: bool
    ):
        """Record a tool invocation."""
        usage = {
            "tool": tool_name,
            "input": str(input_data)[:200],  # Truncate
            "output": str(output_data)[:500] if output_data else None,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.current_session["tool_usage"].append(usage)
    
    def add_finding(self, finding: str, source: str = None):
        """Add a finding or insight."""
        self.current_session["findings"].append({
            "content": finding,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Get the current reasoning chain."""
        return self.current_session["reasoning_chain"]
    
    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning chain."""
        chain = self.current_session["reasoning_chain"]
        if not chain:
            return "No reasoning steps recorded."
        
        parts = []
        for step in chain:
            parts.append(f"[{step['type'].upper()}] {step['content'][:100]}...")
        
        return "\n".join(parts)
    
    def search_past_reasoning(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search past reasoning steps that are relevant to current query.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of relevant past reasoning steps
        """
        query_embedding = self.embedder.encode_query(query)
        
        # Search agent steps collection
        from backend.app.endee_client.collections import get_collection_name, CollectionType
        from backend.app.endee_client.client import get_endee_client
        
        client = get_endee_client()
        collection_name = get_collection_name(self.user_id, CollectionType.AGENT_STEPS)
        
        results = client.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        return results
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get statistics on tool usage."""
        usage = self.current_session["tool_usage"]
        
        if not usage:
            return {"total": 0, "by_tool": {}, "success_rate": 0}
        
        by_tool = {}
        success_count = 0
        
        for u in usage:
            tool = u["tool"]
            by_tool[tool] = by_tool.get(tool, 0) + 1
            if u["success"]:
                success_count += 1
        
        return {
            "total": len(usage),
            "by_tool": by_tool,
            "success_rate": success_count / len(usage),
            "successful": success_count,
            "failed": len(usage) - success_count
        }
    
    def get_findings(self) -> List[Dict[str, Any]]:
        """Get all findings from current session."""
        return self.current_session["findings"]
    
    def clear_session(self):
        """Clear the current session memory."""
        self.current_session = {
            "reasoning_chain": [],
            "tool_usage": [],
            "findings": []
        }
    
    def export_session(self) -> Dict[str, Any]:
        """Export current session data."""
        return {
            "user_id": self.user_id,
            "session_data": self.current_session,
            "tool_stats": self.get_tool_statistics(),
            "exported_at": datetime.utcnow().isoformat()
        }


def create_agent_memory(user_id: int) -> AgentMemoryManager:
    """Create an agent memory manager for a user."""
    return AgentMemoryManager(user_id)
