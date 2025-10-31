"""
Data Analysis Node
Validates and processes prospect data through the DataAnalystAgent.
"""

from state import WorkflowState
from langraph_agents.agents.data_analyst_agent import DataAnalystAgent

# Initialize agent at module level
data_analyst_agent = DataAnalystAgent()


async def data_analysis_node(state: WorkflowState) -> WorkflowState:
    """
    Data Analysis Node - validates and processes prospect data.

    This node is executed by the DataAnalystAgent through the workflow.
    It handles data validation, cleaning, and quality assessment.
    """
    return await data_analyst_agent.run(state)
