"""
Test Generation Node
Generates test cases through the TestAgent.
"""

from state import CodeCrafterState
from agents.test_agent import generate_tests as generate_tests_func


async def test_node(state: CodeCrafterState) -> CodeCrafterState:
    """
    Test Node - generates test cases.

    This node is executed by the TestAgent through the workflow.
    It creates unit tests, integration tests, and test documentation.
    """
    return await generate_tests_func(state)
