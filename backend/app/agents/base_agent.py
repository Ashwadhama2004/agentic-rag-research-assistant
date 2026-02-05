"""
Base Agent Class
Foundation for agentic AI operations with planning and reasoning.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import time


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentStep:
    """Represents a single step in agent execution."""
    id: str
    action: str  # think, plan, act, reflect, respond
    tool_used: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    latency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


@dataclass
class AgentTask:
    """Represents an agent task."""
    id: str
    description: str
    status: str = "pending"
    steps: List[AgentStep] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    @property
    def duration(self) -> float:
        """Get task duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def add_step(self, step: AgentStep):
        """Add a step to the task."""
        step.id = f"step_{len(self.steps) + 1}"
        self.steps.append(step)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "steps": [
                {
                    "id": s.id,
                    "action": s.action,
                    "tool_used": s.tool_used,
                    "status": s.status,
                    "latency": s.latency
                }
                for s in self.steps
            ],
            "result": self.result,
            "error": self.error,
            "duration": self.duration
        }


class BaseAgent:
    """
    Base class for agentic AI operations.
    Implements the THINK → PLAN → ACT → REFLECT → RESPOND loop.
    """
    
    def __init__(
        self,
        user_id: int,
        max_steps: int = 10,
        timeout_seconds: int = 300
    ):
        self.user_id = user_id
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.state = AgentState.IDLE
        self.tools: Dict[str, Callable] = {}
        self.current_task: Optional[AgentTask] = None
    
    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool for the agent to use."""
        self.tools[name] = {
            "function": func,
            "description": description
        }
    
    def _create_step(self, action: str, tool: str = None) -> AgentStep:
        """Create a new agent step."""
        return AgentStep(
            id=f"step_{uuid.uuid4().hex[:8]}",
            action=action,
            tool_used=tool
        )
    
    def think(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        THINK phase: Analyze the task and understand requirements.
        Override in subclasses for custom thinking logic.
        """
        return {
            "analysis": f"Analyzing task: {task}",
            "insights": [],
            "context": context or {}
        }
    
    def plan(self, task: str, thinking_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        PLAN phase: Create a plan of steps to complete the task.
        Override in subclasses for custom planning logic.
        """
        return [
            {"step": 1, "action": "search", "description": "Search relevant documents"},
            {"step": 2, "action": "analyze", "description": "Analyze findings"},
            {"step": 3, "action": "respond", "description": "Generate response"}
        ]
    
    def act(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        ACT phase: Execute a single step using tools.
        Override in subclasses for custom action logic.
        """
        action = step.get("action", "")
        
        if action in self.tools:
            tool = self.tools[action]
            try:
                result = tool["function"](step)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": True, "result": f"Executed: {step.get('description', '')}"}
    
    def reflect(
        self,
        task: str,
        plan: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        REFLECT phase: Evaluate progress and decide next steps.
        Override in subclasses for custom reflection logic.
        """
        completed = sum(1 for r in results if r.get("success", False))
        
        return {
            "completed": completed,
            "total": len(plan),
            "is_complete": completed == len(plan),
            "needs_revision": False
        }
    
    def respond(
        self,
        task: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        RESPOND phase: Generate final response.
        Override in subclasses for custom response generation.
        """
        return f"Task completed. Processed {len(results)} steps."
    
    async def run(
        self,
        task_description: str,
        context: Dict[str, Any] = None
    ) -> AgentTask:
        """
        Run the agent on a task using the full loop.
        """
        # Create task
        self.current_task = AgentTask(
            id=f"task_{uuid.uuid4().hex[:12]}",
            description=task_description,
            status="running"
        )
        self.state = AgentState.THINKING
        
        start_time = time.time()
        results = []
        
        try:
            # THINK
            think_step = self._create_step("think")
            think_step.status = "running"
            self.current_task.add_step(think_step)
            
            thinking = self.think(task_description, context)
            think_step.output_data = thinking
            think_step.status = "completed"
            think_step.latency = time.time() - start_time
            
            # PLAN
            self.state = AgentState.PLANNING
            plan_step = self._create_step("plan")
            plan_step.status = "running"
            self.current_task.add_step(plan_step)
            
            plan = self.plan(task_description, thinking)
            plan_step.output_data = {"plan": plan}
            plan_step.status = "completed"
            plan_step.latency = time.time() - start_time - think_step.latency
            
            # ACT loop
            self.state = AgentState.EXECUTING
            for i, planned_step in enumerate(plan):
                if time.time() - start_time > self.timeout_seconds:
                    raise TimeoutError("Agent execution timed out")
                
                if i >= self.max_steps:
                    break
                
                act_step = self._create_step("act", planned_step.get("action"))
                act_step.input_data = planned_step
                act_step.status = "running"
                self.current_task.add_step(act_step)
                
                step_start = time.time()
                result = self.act(planned_step)
                results.append(result)
                
                act_step.output_data = result
                act_step.status = "completed" if result.get("success") else "failed"
                act_step.latency = time.time() - step_start
            
            # REFLECT
            self.state = AgentState.REFLECTING
            reflect_step = self._create_step("reflect")
            reflect_step.status = "running"
            self.current_task.add_step(reflect_step)
            
            reflection = self.reflect(task_description, plan, results)
            reflect_step.output_data = reflection
            reflect_step.status = "completed"
            
            # RESPOND
            response = self.respond(task_description, results)
            
            respond_step = self._create_step("respond")
            respond_step.output_data = {"response": response}
            respond_step.status = "completed"
            self.current_task.add_step(respond_step)
            
            # Complete task
            self.current_task.result = response
            self.current_task.status = "completed"
            self.current_task.completed_at = datetime.utcnow()
            self.state = AgentState.COMPLETED
            
        except Exception as e:
            self.current_task.status = "failed"
            self.current_task.error = str(e)
            self.current_task.completed_at = datetime.utcnow()
            self.state = AgentState.FAILED
        
        return self.current_task
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "state": self.state.value,
            "task": self.current_task.to_dict() if self.current_task else None,
            "available_tools": list(self.tools.keys())
        }
