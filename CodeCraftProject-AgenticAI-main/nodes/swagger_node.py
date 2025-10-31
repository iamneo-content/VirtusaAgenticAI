"""
Swagger/API Documentation Node
Generates API documentation through the SwaggerAgent.
"""

from state import CodeCrafterState
from agents.swagger_agent import swagger_agent as swagger_agent_func


async def swagger_node(state: CodeCrafterState) -> CodeCrafterState:
    """
    Swagger Node - generates API documentation.

    This node is executed by the SwaggerAgent through the workflow.
    It creates OpenAPI/Swagger specifications for all services.
    """
    return await swagger_agent_func(state)
