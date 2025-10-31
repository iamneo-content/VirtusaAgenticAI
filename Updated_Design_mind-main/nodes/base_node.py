"""
Base Node Class for Workflow Nodes
Provides common functionality for all workflow nodes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableLambda

from state.models import HLDState


class BaseNode(ABC):
    """Abstract base class for all workflow nodes"""

    def __init__(self, name: str, description: str = "", critical: bool = False):
        """
        Initialize the base node

        Args:
            name: Node identifier
            description: Human-readable description of the node
            critical: Whether this node is critical for workflow success
        """
        self.name = name
        self.description = description
        self.critical = critical
        self.status = "pending"
        self.last_error: Optional[str] = None

    @abstractmethod
    def execute_logic(self, hld_state: HLDState) -> Any:
        """
        Execute the node's core logic

        Args:
            hld_state: Current workflow state

        Returns:
            Result of the node execution
        """
        pass

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node with error handling

        Args:
            state: Current workflow state dictionary

        Returns:
            Updated state dictionary
        """
        try:
            hld_state = HLDState(**state)
            result = self.execute_logic(hld_state)

            updated_state = hld_state.dict()
            updated_state["_node_result"] = result
            updated_state["_last_executed_node"] = self.name

            self.status = "completed"
            self.last_error = None

            return updated_state
        except Exception as e:
            self.status = "failed"
            self.last_error = str(e)
            raise

    def get_runnable(self) -> RunnableLambda:
        """Get the node as a RunnableLambda for LangGraph"""
        return RunnableLambda(self.execute)

    def get_info(self) -> Dict[str, Any]:
        """Get node information"""
        return {
            "name": self.name,
            "description": self.description,
            "critical": self.critical,
            "status": self.status,
            "last_error": self.last_error
        }
