"""Main prospect analysis workflow using LangGraph."""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..state_models import WorkflowState, ProspectData
from ..agents.data_analyst_agent import DataAnalystAgent
from ..agents.risk_assessment_agent import RiskAssessmentAgent
from ..agents.persona_agent import PersonaAgent
from ..agents.product_specialist_agent import ProductSpecialistAgent
from config.logging_config import get_logger


class ProspectAnalysisWorkflow:
    """Main workflow for comprehensive prospect analysis."""
    
    def __init__(self):
        self.logger = get_logger("ProspectAnalysisWorkflow")
        self.graph = None
        self.checkpointer = MemorySaver()
        self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow."""
        self.logger.info("Building prospect analysis workflow")
        
        # Initialize agents
        self.data_analyst = DataAnalystAgent()
        self.risk_assessor = RiskAssessmentAgent()
        self.persona_classifier = PersonaAgent()
        self.product_specialist = ProductSpecialistAgent()
        
        # Create workflow graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (agents)
        workflow.add_node("data_analysis", self._data_analysis_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("persona_classification", self._persona_classification_node)
        workflow.add_node("product_recommendation", self._product_recommendation_node)
        workflow.add_node("finalize_analysis", self._finalize_analysis_node)
        
        # Define workflow edges
        workflow.set_entry_point("data_analysis")
        
        # Sequential flow with conditional routing
        workflow.add_edge("data_analysis", "risk_assessment")
        workflow.add_edge("risk_assessment", "persona_classification")
        workflow.add_edge("persona_classification", "product_recommendation")
        workflow.add_edge("product_recommendation", "finalize_analysis")
        workflow.add_edge("finalize_analysis", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.checkpointer)
        self.logger.info("Workflow compiled successfully")
    
    async def _data_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Data analysis node."""
        self.logger.info("Executing data analysis node")
        state.current_step = "data_analysis"
        
        try:
            result_state = await self.data_analyst.run(state)
            result_state.completed_steps.append("data_analysis")
            return result_state
        except Exception as e:
            self.logger.error(f"Data analysis failed: {str(e)}")
            state.failed_steps.append("data_analysis")
            raise
    
    async def _risk_assessment_node(self, state: WorkflowState) -> WorkflowState:
        """Risk assessment node."""
        self.logger.info("Executing risk assessment node")
        state.current_step = "risk_assessment"
        
        try:
            result_state = await self.risk_assessor.run(state)
            result_state.completed_steps.append("risk_assessment")
            return result_state
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            state.failed_steps.append("risk_assessment")
            raise
    
    async def _persona_classification_node(self, state: WorkflowState) -> WorkflowState:
        """Persona classification node."""
        self.logger.info("Executing persona classification node")
        state.current_step = "persona_classification"
        
        try:
            result_state = await self.persona_classifier.run(state)
            result_state.completed_steps.append("persona_classification")
            return result_state
        except Exception as e:
            self.logger.error(f"Persona classification failed: {str(e)}")
            state.failed_steps.append("persona_classification")
            # Non-critical - continue without persona
            return state
    
    async def _product_recommendation_node(self, state: WorkflowState) -> WorkflowState:
        """Product recommendation node."""
        self.logger.info("Executing product recommendation node")
        state.current_step = "product_recommendation"
        
        try:
            result_state = await self.product_specialist.run(state)
            result_state.completed_steps.append("product_recommendation")
            return result_state
        except Exception as e:
            self.logger.error(f"Product recommendation failed: {str(e)}")
            state.failed_steps.append("product_recommendation")
            raise
    
    async def _finalize_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize analysis and generate summary."""
        self.logger.info("Finalizing analysis")
        state.current_step = "finalize_analysis"
        
        try:
            # Calculate overall confidence
            confidence_scores = []
            
            if state.analysis.risk_assessment:
                confidence_scores.append(state.analysis.risk_assessment.confidence_score)
            
            if state.analysis.persona_classification:
                confidence_scores.append(state.analysis.persona_classification.confidence_score)
            
            if state.prospect.data_quality_score:
                confidence_scores.append(state.prospect.data_quality_score)
            
            if confidence_scores:
                state.overall_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Generate key insights
            state.key_insights = self._generate_key_insights(state)
            
            # Generate action items
            state.action_items = self._generate_action_items(state)
            
            # Update timestamps
            state.updated_at = datetime.now()
            state.completed_steps.append("finalize_analysis")
            
            self.logger.info("Analysis finalized successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Analysis finalization failed: {str(e)}")
            state.failed_steps.append("finalize_analysis")
            return state
    
    def _generate_key_insights(self, state: WorkflowState) -> list:
        """Generate key insights from the analysis."""
        insights = []
        
        if state.analysis.risk_assessment:
            insights.append(f"Risk Profile: {state.analysis.risk_assessment.risk_level}")
        
        if state.analysis.persona_classification:
            insights.append(f"Investor Persona: {state.analysis.persona_classification.persona_type}")
        
        if state.recommendations.recommended_products:
            top_product = state.recommendations.recommended_products[0]
            insights.append(f"Top Recommendation: {top_product.product_name}")
        
        if state.prospect.data_quality_score:
            if state.prospect.data_quality_score > 0.8:
                insights.append("High data quality - reliable analysis")
            elif state.prospect.data_quality_score < 0.6:
                insights.append("Data quality concerns - additional verification needed")
        
        return insights
    
    def _generate_action_items(self, state: WorkflowState) -> list:
        """Generate action items for the RM."""
        actions = []
        
        if state.prospect.validation_errors:
            actions.append("Verify and correct data validation errors")
        
        if state.analysis.risk_assessment and state.analysis.risk_assessment.risk_level == "High":
            actions.append("Discuss risk tolerance and investment experience in detail")
        
        if state.recommendations.recommended_products:
            actions.append("Present top product recommendations with justifications")
        
        if state.analysis.persona_classification:
            persona_type = state.analysis.persona_classification.persona_type
            if persona_type == "Cautious Planner":
                actions.append("Focus on capital preservation and security features")
            elif persona_type == "Aggressive Growth":
                actions.append("Emphasize growth potential and long-term returns")
        
        actions.append("Schedule follow-up meeting to discuss recommendations")
        
        return actions
    
    async def analyze_prospect(
        self, 
        prospect_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> WorkflowState:
        """Analyze a prospect using the complete workflow."""
        
        # Create initial state
        workflow_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        
        initial_state = WorkflowState(
            workflow_id=workflow_id,
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Set prospect data
        initial_state.prospect.prospect_data = ProspectData(**prospect_data)
        
        self.logger.info(f"Starting prospect analysis for {prospect_data.get('name', 'Unknown')}")
        
        try:
            # Execute workflow
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            self.logger.info(f"Prospect analysis completed successfully. Workflow ID: {workflow_id}")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Prospect analysis failed: {str(e)}")
            raise
    
    async def get_workflow_state(self, session_id: str) -> Optional[WorkflowState]:
        """Get the current state of a workflow session."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = await self.graph.aget_state(config)
            return state.values if state else None
        except Exception as e:
            self.logger.error(f"Failed to get workflow state: {str(e)}")
            return None
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get workflow configuration summary."""
        return {
            "workflow_name": "Prospect Analysis Workflow",
            "agents": [
                self.data_analyst.name,
                self.risk_assessor.name,
                self.persona_classifier.name,
                self.product_specialist.name
            ],
            "steps": [
                "data_analysis",
                "risk_assessment", 
                "persona_classification",
                "product_recommendation",
                "finalize_analysis"
            ],
            "critical_agents": [
                self.data_analyst.name,
                self.risk_assessor.name,
                self.product_specialist.name
            ],
            "optional_agents": [
                self.persona_classifier.name
            ]
        }