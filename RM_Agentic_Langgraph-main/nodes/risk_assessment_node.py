"""
Risk Assessment Node
Performs risk profiling and assessment through the RiskAssessmentAgent.
"""

from state import WorkflowState
from langraph_agents.agents.risk_assessment_agent import RiskAssessmentAgent

# Initialize agent at module level
risk_assessment_agent = RiskAssessmentAgent()


async def risk_assessment_node(state: WorkflowState) -> WorkflowState:
    """
    Risk Assessment Node - performs risk profiling using ML models.

    This node is executed by the RiskAssessmentAgent through the workflow.
    It calculates risk scores and identifies risk factors.
    """
    return await risk_assessment_agent.run(state)
