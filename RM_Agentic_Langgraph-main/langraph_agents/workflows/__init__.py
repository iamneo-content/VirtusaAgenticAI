"""LangGraph workflows for orchestrating multi-agent processes."""

from .prospect_analysis_workflow import ProspectAnalysisWorkflow

# TODO: Implement these workflows in future versions
# from .product_recommendation_workflow import ProductRecommendationWorkflow
# from .interactive_chat_workflow import InteractiveChatWorkflow

__all__ = [
    "ProspectAnalysisWorkflow",
    # "ProductRecommendationWorkflow", 
    # "InteractiveChatWorkflow"
]