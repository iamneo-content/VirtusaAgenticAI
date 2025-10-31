"""
LangGraph Workflow Graph Definition
Defines the main interview workflow and evaluation workflow using LangGraph.
"""

from langgraph.graph import StateGraph, END
from state import CandidateState
from nodes.parse_resume import parse_resume
from ml.training.predict_experience import predict_experience
from ml.training.score_resume import predict_resume_score
from nodes.analyze_job_fit import analyze_job_fit
from nodes.generate_questions import generate_questions
from nodes.evaluate_answers import evaluate_answers


def create_workflow():
    """Create and configure the main LangGraph workflow

    Workflow sequence:
    1. parse_resume: Extract structured features from resume
    2. predict_experience: Predict candidate experience level (Junior/Mid/Senior)
    3. score_resume: Score resume quality (0-10)
    4. analyze_job_fit: Analyze candidate-job compatibility
    5. generate_questions: Generate personalized interview questions
    6. END: Complete main workflow

    Returns:
        Compiled LangGraph workflow
    """

    # Initialize the graph
    workflow = StateGraph(CandidateState)

    # Add nodes (execution units)
    workflow.add_node("parse_resume", parse_resume)
    workflow.add_node("predict_experience", predict_experience)
    workflow.add_node("score_resume", predict_resume_score)
    workflow.add_node("analyze_job_fit", analyze_job_fit)
    workflow.add_node("generate_questions", generate_questions)

    # Add sequential edges for main workflow
    workflow.add_edge("parse_resume", "predict_experience")
    workflow.add_edge("predict_experience", "score_resume")
    workflow.add_edge("score_resume", "analyze_job_fit")
    workflow.add_edge("analyze_job_fit", "generate_questions")
    workflow.add_edge("generate_questions", END)  # End after questions generated

    # Set entry point
    workflow.set_entry_point("parse_resume")

    return workflow.compile()


def create_evaluation_workflow():
    """Create workflow for answer evaluation

    This separate workflow evaluates answers submitted by candidates.

    Workflow sequence:
    1. evaluate_answers: Evaluate submitted answers using semantic similarity
    2. END: Complete evaluation workflow

    Returns:
        Compiled LangGraph evaluation workflow
    """

    workflow = StateGraph(CandidateState)
    workflow.add_node("evaluate_answers", evaluate_answers)
    workflow.add_edge("evaluate_answers", END)
    workflow.set_entry_point("evaluate_answers")

    return workflow.compile()


# Create the compiled workflows
app = create_workflow()
evaluation_app = create_evaluation_workflow()

__all__ = ["app", "evaluation_app", "create_workflow", "create_evaluation_workflow"]
