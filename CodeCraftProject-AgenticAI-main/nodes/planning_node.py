"""
Planning Node
Analyzes user stories and plans architecture through the PlanningAgent.
"""

from state import CodeCrafterState
from agents.planning_agent import planning_agent as planning_agent_func


async def planning_node(state: CodeCrafterState) -> CodeCrafterState:
    """
    Planning Node - analyzes user stories and plans architecture.

    This node is executed by the PlanningAgent through the workflow.
    It identifies features, services, and architecture requirements.
    """
    return await planning_agent_func(state)
