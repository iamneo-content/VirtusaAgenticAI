from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
import operator

class CandidateState(TypedDict):
    """State schema for candidate evaluation workflow"""
    
    # Input data
    resume_text: str
    job_description: str
    job_title: str
    candidate_name: str
    
    # Extracted features
    resume_features: Optional[Dict[str, Any]]
    experience_level: Optional[str]
    resume_score: Optional[float]
    job_fit: Optional[Dict[str, Any]]
    
    # Interview data
    questions: Annotated[List[Dict[str, Any]], operator.add]
    answers: Annotated[List[Dict[str, Any]], operator.add]
    
    # Results
    final_score: Optional[float]
    feedback: Optional[str]
    
    # Control flow
    current_step: str
    errors: Annotated[List[str], operator.add]