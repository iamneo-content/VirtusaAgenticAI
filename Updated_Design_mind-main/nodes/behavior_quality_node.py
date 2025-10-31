"""
Behavior and Quality Analysis Node
Handles behavior pattern and quality metrics analysis
"""

from typing import Any
from state.models import HLDState
from agent import BehaviorQualityAgent
from .base_node import BaseNode


class BehaviorQualityNode(BaseNode):
    """Node for behavior pattern and quality analysis"""

    def __init__(self):
        """Initialize behavior and quality analysis node"""
        super().__init__(
            name="behavior_quality",
            description="Analyze behavior patterns and quality metrics",
            critical=False
        )
        self.agent = BehaviorQualityAgent()

    def execute_logic(self, hld_state: HLDState) -> Any:
        """
        Execute behavior and quality analysis logic

        Args:
            hld_state: Current workflow state

        Returns:
            Behavior and quality analysis results
        """
        return self.agent.process(hld_state)
