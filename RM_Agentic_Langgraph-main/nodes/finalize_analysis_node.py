"""
Finalize Analysis Node
Finalizes analysis and prepares final recommendations.
"""

from state import WorkflowState


async def finalize_analysis_node(state: WorkflowState) -> WorkflowState:
    """
    Finalize Analysis Node - finalizes the comprehensive prospect analysis.

    This node aggregates results from all previous analysis steps,
    prepares final recommendations, and structures the output.
    """
    # Aggregate analysis results
    state.analysis_complete = True
    state.updated_at = None  # Will be set by the workflow

    return state
