"""
Agent Executor
Executes agent steps using registered tools.
"""
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import time

from backend.app.agents.tools import AgentToolkit
from backend.app.agents.planner import TaskPlanner


class StepResult:
    """Result of executing an agent step."""
    
    def __init__(
        self,
        success: bool,
        output: Any = None,
        error: str = None,
        latency: float = 0.0
    ):
        self.success = success
        self.output = output
        self.error = error
        self.latency = latency
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "latency": self.latency,
            "timestamp": self.timestamp.isoformat()
        }


class AgentExecutor:
    """
    Executes agent plans step by step.
    Manages tool invocation and result collection.
    """
    
    def __init__(
        self,
        user_id: int,
        toolkit: AgentToolkit = None,
        max_retries: int = 2
    ):
        self.user_id = user_id
        self.toolkit = toolkit or AgentToolkit(user_id)
        self.max_retries = max_retries
        self.execution_history: List[StepResult] = []
    
    def execute_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> StepResult:
        """
        Execute a single step from the plan.
        
        Args:
            step: Step definition with action and input
            context: Execution context from previous steps
        
        Returns:
            StepResult with output or error
        """
        start_time = time.time()
        action = step.get("action", "")
        step_input = step.get("input", step.get("description", ""))
        
        # Add context to input if available
        if context:
            step_input = f"{step_input}\n\nContext: {context}"
        
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                # Execute using toolkit
                output = self.toolkit.use_tool(action, step_input)
                
                result = StepResult(
                    success=True,
                    output=output,
                    latency=time.time() - start_time
                )
                self.execution_history.append(result)
                return result
                
            except Exception as e:
                last_error = str(e)
                retries += 1
                time.sleep(0.5)  # Brief pause before retry
        
        # All retries failed
        result = StepResult(
            success=False,
            error=last_error,
            latency=time.time() - start_time
        )
        self.execution_history.append(result)
        return result
    
    def execute_plan(
        self,
        plan: List[Dict[str, Any]],
        stop_on_failure: bool = False
    ) -> List[StepResult]:
        """
        Execute a full plan.
        
        Args:
            plan: List of steps to execute
            stop_on_failure: Stop execution on first failure
        
        Returns:
            List of StepResults
        """
        results = []
        context = {}
        
        for step in plan:
            result = self.execute_step(step, context)
            results.append(result)
            
            # Update context with results
            if result.success and result.output:
                context[f"step_{step.get('step', len(results))}"] = result.output
            
            # Check for stop condition
            if not result.success and stop_on_failure:
                break
        
        return results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution history."""
        if not self.execution_history:
            return {"steps": 0, "success_rate": 0, "total_latency": 0}
        
        successful = sum(1 for r in self.execution_history if r.success)
        total_latency = sum(r.latency for r in self.execution_history)
        
        return {
            "steps": len(self.execution_history),
            "successful": successful,
            "failed": len(self.execution_history) - successful,
            "success_rate": successful / len(self.execution_history),
            "total_latency": round(total_latency, 3),
            "avg_latency": round(total_latency / len(self.execution_history), 3)
        }
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_history.clear()


class ResearchAgent:
    """
    High-level agent for research tasks.
    Combines planning and execution.
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.planner = TaskPlanner()
        self.executor = AgentExecutor(user_id)
    
    async def research(
        self,
        query: str,
        max_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Perform research on a topic.
        
        Args:
            query: Research query
            max_steps: Maximum steps
        
        Returns:
            Research results with steps and findings
        """
        # Plan
        plan = self.planner.plan(query, max_steps=max_steps)
        
        # Execute
        results = self.executor.execute_plan(plan)
        
        # Compile findings
        findings = []
        for result in results:
            if result.success and result.output:
                findings.append(result.output)
        
        return {
            "query": query,
            "plan": plan,
            "results": [r.to_dict() for r in results],
            "findings": findings,
            "summary": self.executor.get_execution_summary()
        }


def create_executor(user_id: int) -> AgentExecutor:
    """Create an agent executor for a user."""
    return AgentExecutor(user_id)
