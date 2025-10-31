"""
Product Recommendation Node
Generates product recommendations through the ProductSpecialistAgent.
"""

from state import WorkflowState
from langraph_agents.agents.product_specialist_agent import ProductSpecialistAgent

# Initialize agent at module level
product_specialist_agent = ProductSpecialistAgent()


async def product_recommendation_node(state: WorkflowState) -> WorkflowState:
    """
    Product Recommendation Node - generates personalized product recommendations.

    This node is executed by the ProductSpecialistAgent through the workflow.
    It recommends suitable investment products based on prospect profile and risk assessment.
    """
    return await product_specialist_agent.run(state)
