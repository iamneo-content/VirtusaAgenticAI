"""
Code Generation Node
Generates service code through the CodeGenAgent.
"""

from state import CodeCrafterState
from agents.codegen_agent import codegen_agent as codegen_agent_func


async def codegen_node(state: CodeCrafterState) -> CodeCrafterState:
    """
    Code Generation Node - generates service code.

    This node is executed by the CodeGenAgent through the workflow.
    It creates microservice implementations based on architecture plan.
    """
    return await codegen_agent_func(state)
