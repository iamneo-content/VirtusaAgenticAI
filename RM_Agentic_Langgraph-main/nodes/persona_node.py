"""
Persona Classification Node
Classifies investor behavior and personas through the PersonaAgent.
"""

from state import WorkflowState
from langraph_agents.agents.persona_agent import PersonaAgent

# Initialize agent at module level
persona_agent = PersonaAgent()


async def persona_node(state: WorkflowState) -> WorkflowState:
    """
    Persona Classification Node - classifies investor persona and behavioral patterns.

    This node is executed by the PersonaAgent through the workflow.
    It identifies investor personality types and behavioral insights.
    """
    return await persona_agent.run(state)
