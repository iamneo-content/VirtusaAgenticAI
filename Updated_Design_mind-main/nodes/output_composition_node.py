"""
Output Composition Node
Handles composition and formatting of final output
"""

from typing import Any
from state.models import HLDState
from agent import OutputAgent
from .base_node import BaseNode


class OutputCompositionNode(BaseNode):
    """Node for final output composition and formatting"""

    def __init__(self):
        """Initialize output composition node"""
        super().__init__(
            name="output_composition",
            description="Compose and format final output",
            critical=False
        )
        self.agent = OutputAgent()

    def execute_logic(self, hld_state: HLDState) -> Any:
        """
        Execute output composition logic

        Args:
            hld_state: Current workflow state

        Returns:
            Composed and formatted final output
        """
        return self.agent.process(hld_state)
