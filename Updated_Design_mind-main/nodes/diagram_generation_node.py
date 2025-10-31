"""
Diagram Generation Node
Handles visual diagram and representation generation
"""

from typing import Any
from state.models import HLDState
from agent import DiagramAgent
from .base_node import BaseNode


class DiagramGenerationNode(BaseNode):
    """Node for diagram and visual representation generation"""

    def __init__(self):
        """Initialize diagram generation node"""
        super().__init__(
            name="diagram_generation",
            description="Generate visual diagrams and representations",
            critical=False
        )
        self.agent = DiagramAgent()

    def execute_logic(self, hld_state: HLDState) -> Any:
        """
        Execute diagram generation logic

        Args:
            hld_state: Current workflow state

        Returns:
            Generated diagrams and visual representations
        """
        return self.agent.process(hld_state)
