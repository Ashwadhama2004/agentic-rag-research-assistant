"""
Task Planner
Plans and decomposes complex tasks into executable steps.
"""
from typing import List, Dict, Any, Optional

from backend.app.rag.generator import get_generator
from backend.app.config import config


class TaskPlanner:
    """
    Plans task execution by breaking complex tasks into steps.
    Uses LLM for intelligent task decomposition.
    """
    
    PLANNING_PROMPT = """You are a research assistant that breaks down complex tasks into simple, actionable steps.

TASK: {task}

AVAILABLE TOOLS:
{tools}

Break down this task into a numbered list of steps. Each step should:
1. Be a single, clear action
2. Use one of the available tools when applicable
3. Build on previous steps

Output format (JSON list):
[
  {{"step": 1, "action": "tool_name", "description": "What to do", "input": "specific input"}},
  ...
]

PLAN:"""

    def __init__(self, available_tools: List[str] = None):
        self.generator = get_generator()
        self.available_tools = available_tools or [
            "vector_search",
            "document_retrieval",
            "summarize",
            "compare",
            "analyze"
        ]
    
    def plan(
        self,
        task: str,
        context: Dict[str, Any] = None,
        max_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Create an execution plan for a task.
        
        Args:
            task: Task description
            context: Optional context information
            max_steps: Maximum number of steps
        
        Returns:
            List of planned steps
        """
        # Format tools
        tools_str = "\n".join([f"- {tool}" for tool in self.available_tools])
        
        # Build prompt
        prompt = self.PLANNING_PROMPT.format(
            task=task,
            tools=tools_str
        )
        
        # Add context if available
        if context:
            prompt += f"\n\nADDITIONAL CONTEXT:\n{context}"
        
        # Generate plan
        try:
            response, _ = self.generator._call_llm(prompt, max_tokens=1024, temperature=0.3)
            plan = self._parse_plan(response)
            
            # Limit steps
            plan = plan[:max_steps]
            
            return plan
        except Exception as e:
            # Return default plan on error
            return self._default_plan(task)
    
    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured plan."""
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                plan = json.loads(json_match.group())
                return self._validate_plan(plan)
            except json.JSONDecodeError:
                pass
        
        # Fallback: Parse numbered list
        steps = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match numbered items
            match = re.match(r'^(\d+)\.\s*(.+)', line)
            if match:
                step_num = int(match.group(1))
                description = match.group(2)
                
                # Try to identify action
                action = "analyze"
                for tool in self.available_tools:
                    if tool.lower() in description.lower():
                        action = tool
                        break
                
                steps.append({
                    "step": step_num,
                    "action": action,
                    "description": description,
                    "input": ""
                })
        
        return steps if steps else self._default_plan("")
    
    def _validate_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize plan structure."""
        validated = []
        
        for i, step in enumerate(plan, 1):
            validated_step = {
                "step": step.get("step", i),
                "action": step.get("action", "analyze"),
                "description": step.get("description", "Execute step"),
                "input": step.get("input", "")
            }
            
            # Ensure action is valid
            if validated_step["action"] not in self.available_tools:
                validated_step["action"] = "analyze"
            
            validated.append(validated_step)
        
        return validated
    
    def _default_plan(self, task: str) -> List[Dict[str, Any]]:
        """Return a default plan when LLM planning fails."""
        return [
            {
                "step": 1,
                "action": "vector_search",
                "description": "Search for relevant documents",
                "input": task
            },
            {
                "step": 2,
                "action": "analyze",
                "description": "Analyze retrieved information",
                "input": ""
            },
            {
                "step": 3,
                "action": "summarize",
                "description": "Summarize findings",
                "input": ""
            }
        ]
    
    def refine_plan(
        self,
        original_plan: List[Dict[str, Any]],
        completed_steps: List[Dict[str, Any]],
        feedback: str
    ) -> List[Dict[str, Any]]:
        """
        Refine plan based on execution feedback.
        
        Args:
            original_plan: The original plan
            completed_steps: Steps completed so far
            feedback: Feedback or issues encountered
        
        Returns:
            Refined plan
        """
        # Get remaining steps
        remaining = [s for s in original_plan if s["step"] > len(completed_steps)]
        
        # Simple refinement: add re-analysis step if needed
        if "error" in feedback.lower() or "fail" in feedback.lower():
            remaining.insert(0, {
                "step": len(completed_steps) + 1,
                "action": "analyze",
                "description": f"Re-analyze due to: {feedback}",
                "input": ""
            })
        
        # Renumber steps
        for i, step in enumerate(remaining, len(completed_steps) + 1):
            step["step"] = i
        
        return remaining


def create_planner(available_tools: List[str] = None) -> TaskPlanner:
    """Create a task planner instance."""
    return TaskPlanner(available_tools)
