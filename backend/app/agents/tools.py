"""
Agent Tools
Tools available for agents to use during execution.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from backend.app.rag.retriever import create_retriever
from backend.app.rag.generator import get_generator
from backend.app.rag.embedder import get_embedder
from backend.app.endee_client.multi_tenant import get_tenant_manager


@dataclass
class ToolDefinition:
    """Definition of an agent tool."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, str] = None


class AgentToolkit:
    """
    Collection of tools available to agents.
    Provides vector search, summarization, analysis, and more.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.retriever = create_retriever(user_id)
        self.generator = get_generator()
        self.embedder = get_embedder()
        self.tenant_manager = get_tenant_manager(user_id)
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default set of tools."""
        self.register(ToolDefinition(
            name="vector_search",
            description="Search for relevant documents using semantic similarity",
            function=self._vector_search,
            parameters={"query": "Search query string"}
        ))
        
        self.register(ToolDefinition(
            name="document_retrieval",
            description="Retrieve specific documents by ID or search",
            function=self._document_retrieval,
            parameters={"query": "Document search query"}
        ))
        
        self.register(ToolDefinition(
            name="summarize",
            description="Summarize text or search results",
            function=self._summarize,
            parameters={"text": "Text to summarize"}
        ))
        
        self.register(ToolDefinition(
            name="analyze",
            description="Analyze information and extract insights",
            function=self._analyze,
            parameters={"content": "Content to analyze"}
        ))
        
        self.register(ToolDefinition(
            name="compare",
            description="Compare multiple pieces of information",
            function=self._compare,
            parameters={"items": "Items to compare"}
        ))
        
        self.register(ToolDefinition(
            name="generate_answer",
            description="Generate an answer based on context",
            function=self._generate_answer,
            parameters={"query": "Question to answer", "context": "Context for answering"}
        ))
    
    def register(self, tool: ToolDefinition):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }
            for t in self.tools.values()
        ]
    
    def use_tool(self, name: str, input_data: Any) -> Any:
        """
        Use a tool by name.
        
        Args:
            name: Tool name
            input_data: Input for the tool
        
        Returns:
            Tool output
        
        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        
        return tool.function(input_data)
    
    # Tool implementations
    
    def _vector_search(self, query: str) -> Dict[str, Any]:
        """Search for relevant documents."""
        results = self.retriever.retrieve(query, top_k=5)
        
        return {
            "query": query,
            "results": [
                {
                    "content": r.content[:500],  # Truncate for agent
                    "document": r.document_name,
                    "score": round(r.score, 3)
                }
                for r in results
            ],
            "count": len(results)
        }
    
    def _document_retrieval(self, query: str) -> Dict[str, Any]:
        """Retrieve documents with full context."""
        results = self.retriever.retrieve_with_context(query, top_k=3)
        
        return {
            "documents": [
                {
                    "content": r.content,
                    "document": r.document_name,
                    "page": r.page_number,
                    "score": round(r.score, 3)
                }
                for r in results.get("documents", [])
            ],
            "related_conversations": len(results.get("conversations", [])),
            "related_summaries": len(results.get("summaries", []))
        }
    
    def _summarize(self, text: str) -> str:
        """Summarize text."""
        # Handle case where text might be a dict
        if isinstance(text, dict):
            text = str(text)
        
        # Truncate to avoid token limits (max ~3000 chars for content)
        if len(text) > 3000:
            text = text[:3000] + "... [truncated]"
        
        return self.generator.summarize(text, max_length=150)
    
    def _analyze(self, content: str) -> Dict[str, Any]:
        """Analyze content and extract insights."""
        if isinstance(content, dict):
            content = str(content)
        
        # Truncate to avoid token limits (max ~3000 chars for content)
        original_length = len(content)
        if len(content) > 3000:
            content = content[:3000] + "... [truncated]"
        
        prompt = f"""Analyze the following content and extract key insights:

{content}

Provide:
1. Main topics covered
2. Key findings or facts
3. Any notable patterns or trends

Analysis:"""
        
        response, _ = self.generator._call_llm(prompt, max_tokens=500, temperature=0.5)
        
        return {
            "analysis": response,
            "content_length": original_length
        }
    
    def _compare(self, items: str) -> Dict[str, Any]:
        """Compare multiple items."""
        if isinstance(items, dict):
            items = str(items)
        
        # Truncate to avoid token limits (max ~3000 chars for content)
        if len(items) > 3000:
            items = items[:3000] + "... [truncated]"
        
        prompt = f"""Compare the following items and identify similarities and differences:

{items}

Comparison:
1. Similarities:
2. Differences:
3. Conclusion:"""
        
        response, _ = self.generator._call_llm(prompt, max_tokens=500, temperature=0.5)
        
        return {
            "comparison": response
        }
    
    def _generate_answer(self, query: str) -> str:
        """Generate an answer with context retrieval."""
        # Get context
        results = self.retriever.retrieve(query, top_k=5)
        
        # Generate answer
        answer = self.generator.generate(query, results)
        
        return answer.answer


def create_toolkit(user_id: int) -> AgentToolkit:
    """Create an agent toolkit for a user."""
    return AgentToolkit(user_id)
