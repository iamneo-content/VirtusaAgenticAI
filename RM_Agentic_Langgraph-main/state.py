"""Pydantic models for LangGraph state management."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd


class ProspectData(BaseModel):
    """Individual prospect data model."""
    prospect_id: str
    name: str
    age: int
    annual_income: float
    current_savings: float
    target_goal_amount: float
    investment_horizon_years: int
    number_of_dependents: int
    investment_experience_level: str
    investment_goal: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class RiskAssessmentResult(BaseModel):
    """Risk assessment results."""
    risk_level: str  # Low, Moderate, High
    confidence_score: float
    risk_factors: List[str]
    recommendations: List[str]


class GoalPredictionResult(BaseModel):
    """Goal success prediction results."""
    goal_success: str
    probability: float
    success_factors: List[str]
    challenges: List[str]
    timeline_analysis: Dict[str, Any]


class PersonaResult(BaseModel):
    """Persona classification results."""
    persona_type: str  # Aggressive Growth, Steady Saver, Cautious Planner
    confidence_score: float
    characteristics: List[str]
    behavioral_insights: List[str]


class ProductRecommendation(BaseModel):
    """Product recommendation model."""
    product_id: str
    product_name: str
    product_type: str
    suitability_score: float
    justification: str
    risk_alignment: str
    expected_returns: Optional[str] = None
    fees: Optional[str] = None


class MeetingGuide(BaseModel):
    """Meeting guide model."""
    agenda_items: List[str]
    key_talking_points: List[str]
    questions_to_ask: List[str]
    objection_handling: Dict[str, str]
    next_steps: List[str]
    estimated_duration: int  # minutes


class ComplianceCheck(BaseModel):
    """Compliance validation results."""
    is_compliant: bool
    compliance_score: float
    violations: List[str]
    warnings: List[str]
    required_disclosures: List[str]


class AgentExecution(BaseModel):
    """Agent execution tracking."""
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class ProspectState(BaseModel):
    """State for prospect data and basic information."""
    prospect_data: Optional[ProspectData] = None
    validation_errors: List[str] = Field(default_factory=list)
    data_quality_score: Optional[float] = None
    missing_fields: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class AnalysisState(BaseModel):
    """State for analysis results."""
    risk_assessment: Optional[RiskAssessmentResult] = None
    goal_prediction: Optional[GoalPredictionResult] = None
    persona_classification: Optional[PersonaResult] = None
    analysis_timestamp: Optional[datetime] = None
    analysis_confidence: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class RecommendationState(BaseModel):
    """State for product recommendations."""
    recommended_products: List[ProductRecommendation] = Field(default_factory=list)
    portfolio_allocation: Optional[Dict[str, float]] = None
    justification_text: Optional[str] = None
    compliance_check: Optional[ComplianceCheck] = None

    class Config:
        arbitrary_types_allowed = True


class MeetingState(BaseModel):
    """State for meeting preparation."""
    meeting_guide: Optional[MeetingGuide] = None
    presentation_slides: Optional[List[str]] = None
    client_materials: Optional[List[str]] = None

    class Config:
        arbitrary_types_allowed = True


class ChatState(BaseModel):
    """State for interactive chat."""
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    current_query: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    response: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class WorkflowState(BaseModel):
    """Complete workflow state combining all sub-states."""
    # Core states
    prospect: ProspectState = Field(default_factory=ProspectState)
    analysis: AnalysisState = Field(default_factory=AnalysisState)
    recommendations: RecommendationState = Field(default_factory=RecommendationState)
    meeting: MeetingState = Field(default_factory=MeetingState)
    chat: ChatState = Field(default_factory=ChatState)

    # Workflow metadata
    workflow_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Execution tracking
    current_step: str = "start"
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    agent_executions: List[AgentExecution] = Field(default_factory=list)

    # Configuration
    workflow_config: Dict[str, Any] = Field(default_factory=dict)

    # Results summary
    overall_confidence: Optional[float] = None
    key_insights: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def add_agent_execution(self, agent_name: str) -> AgentExecution:
        """Add a new agent execution record."""
        execution = AgentExecution(
            agent_name=agent_name,
            start_time=datetime.now()
        )
        self.agent_executions.append(execution)
        return execution

    def complete_agent_execution(self, agent_name: str, success: bool = True, error: Optional[str] = None):
        """Mark an agent execution as completed."""
        for execution in reversed(self.agent_executions):
            if execution.agent_name == agent_name and execution.status == "running":
                execution.end_time = datetime.now()
                execution.status = "completed" if success else "failed"
                execution.error_message = error
                if execution.start_time and execution.end_time:
                    execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
                break

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total_executions = len(self.agent_executions)
        completed = len([e for e in self.agent_executions if e.status == "completed"])
        failed = len([e for e in self.agent_executions if e.status == "failed"])

        total_time = sum([
            e.execution_time for e in self.agent_executions
            if e.execution_time is not None
        ])

        return {
            "total_executions": total_executions,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total_executions if total_executions > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / completed if completed > 0 else 0
        }
