"""
Domain and API Design Node
Handles domain model and API interface design
"""

from typing import Any
from state.models import HLDState
from agent import DomainAPIAgent
from .base_node import BaseNode


class DomainAPINode(BaseNode):
    """Node for domain model and API design"""

    def __init__(self):
        """Initialize domain and API design node"""
        super().__init__(
            name="domain_api_design",
            description="Design domain models and API interfaces",
            critical=False
        )
        self.agent = DomainAPIAgent()

    def execute_logic(self, hld_state: HLDState) -> Any:
        """
        Execute domain and API design logic

        Args:
            hld_state: Current workflow state

        Returns:
            Domain model and API design results
        """
        return self.agent.process(hld_state)
