"""RM Assistant Agent for interactive chat and query handling."""

from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate

from ..base_agent import BaseAgent
from state import WorkflowState
from config.settings import get_settings


class RMAssistantAgent(BaseAgent):
    """Agent responsible for handling RM queries and providing interactive assistance."""
    
    def __init__(self):
        super().__init__(
            name="RM Assistant Agent",
            description="Provides interactive assistance and answers RM queries about client analysis"
        )
        self.settings = get_settings()
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute RM assistance - handle current query."""
        self.logger.info("Processing RM query")
        
        if not state.chat.current_query:
            self.logger.warning("No query to process")
            return state
        
        # Generate response based on current query and context
        response = await self._generate_response(state)
        
        # Update chat state
        state.chat.response = response
        state.chat.conversation_history.append({
            "role": "user",
            "content": state.chat.current_query,
            "timestamp": str(datetime.now())
        })
        state.chat.conversation_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": str(datetime.now())
        })
        
        # Clear current query
        state.chat.current_query = None
        
        self.logger.info("RM query processed successfully")
        return state
    
    async def _generate_response(self, state: WorkflowState) -> str:
        """Generate response to RM query."""
        prompt_template = self.get_prompt_template()
        
        # Prepare context
        context = self._prepare_context(state)
        
        input_variables = {
            "query": state.chat.current_query,
            "prospect_data": self._format_prospect_data(state),
            "analysis_results": self._format_analysis_results(state),
            "recommendations": self._format_recommendations(state),
            "conversation_history": self._format_conversation_history(state),
            "context": context
        }
        
        return await self.generate_response(prompt_template, input_variables)
    
    def _prepare_context(self, state: WorkflowState) -> str:
        """Prepare context information for the query."""
        context_parts = []
        
        # Add prospect summary
        if state.prospect.prospect_data:
            prospect = state.prospect.prospect_data
            context_parts.append(f"Client: {prospect.name}, Age: {prospect.age}")
        
        # Add analysis summary
        if state.analysis.risk_assessment:
            context_parts.append(f"Risk Level: {state.analysis.risk_assessment.risk_level}")
        
        if state.analysis.persona_classification:
            context_parts.append(f"Persona: {state.analysis.persona_classification.persona_type}")
        
        # Add recommendation count
        if state.recommendations.recommended_products:
            context_parts.append(f"Recommendations: {len(state.recommendations.recommended_products)} products")
        
        return " | ".join(context_parts)
    
    def _format_prospect_data(self, state: WorkflowState) -> str:
        """Format prospect data for context."""
        if not state.prospect.prospect_data:
            return "No prospect data available"
        
        prospect = state.prospect.prospect_data
        return f"""
        Name: {prospect.name}
        Age: {prospect.age}
        Annual Income: ₹{prospect.annual_income:,}
        Current Savings: ₹{prospect.current_savings:,}
        Target Goal: ₹{prospect.target_goal_amount:,}
        Investment Horizon: {prospect.investment_horizon_years} years
        Experience Level: {prospect.investment_experience_level}
        Dependents: {prospect.number_of_dependents}
        Investment Goal: {prospect.investment_goal or 'Not specified'}
        """
    
    def _format_analysis_results(self, state: WorkflowState) -> str:
        """Format analysis results for context."""
        results = []
        
        if state.analysis.risk_assessment:
            risk = state.analysis.risk_assessment
            results.append(f"Risk Assessment: {risk.risk_level} (Confidence: {risk.confidence_score:.1%})")
            if risk.risk_factors:
                results.append(f"Key Risk Factors: {', '.join(risk.risk_factors[:3])}")
        
        if state.analysis.persona_classification:
            persona = state.analysis.persona_classification
            results.append(f"Persona: {persona.persona_type} (Confidence: {persona.confidence_score:.1%})")
        
        if state.prospect.data_quality_score:
            results.append(f"Data Quality: {state.prospect.data_quality_score:.1%}")
        
        return "\n".join(results) if results else "No analysis results available"
    
    def _format_recommendations(self, state: WorkflowState) -> str:
        """Format recommendations for context."""
        if not state.recommendations.recommended_products:
            return "No product recommendations available"
        
        rec_text = []
        for i, rec in enumerate(state.recommendations.recommended_products[:3], 1):
            rec_text.append(
                f"{i}. {rec.product_name} ({rec.product_type}) - "
                f"Suitability: {rec.suitability_score:.1%}, Risk: {rec.risk_alignment}"
            )
        
        if state.recommendations.justification_text:
            rec_text.append(f"\nJustification: {state.recommendations.justification_text}")
        
        return "\n".join(rec_text)
    
    def _format_conversation_history(self, state: WorkflowState) -> str:
        """Format conversation history for context."""
        if not state.chat.conversation_history:
            return "No previous conversation"
        
        history = []
        for msg in state.chat.conversation_history[-6:]:  # Last 6 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]  # Truncate long messages
            history.append(f"{role.title()}: {content}")
        
        return "\n".join(history)
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get prompt template for RM assistance."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            You are an AI assistant helping a Relationship Manager (RM) with client analysis and advisory.
            
            Current Context:
            {context}
            
            Prospect Information:
            {prospect_data}
            
            Analysis Results:
            {analysis_results}
            
            Product Recommendations:
            {recommendations}
            
            Recent Conversation:
            {conversation_history}
            
            RM Query: {query}
            
            Instructions:
            1. Provide helpful, accurate responses based on the available data
            2. If asked about specific products, reference the recommendations
            3. For risk-related questions, use the risk assessment results
            4. Be professional and client-focused in your responses
            5. If information is not available, clearly state so
            6. Provide actionable insights when possible
            
            Respond in a clear, professional manner that helps the RM serve their client better.
            """)
        ])
    
    async def handle_query(self, state: WorkflowState, query: str) -> str:
        """Handle a specific query and return response."""
        # Set the query in state
        state.chat.current_query = query
        
        # Execute the agent
        updated_state = await self.execute(state)
        
        # Return the response
        return updated_state.chat.response or "I apologize, but I couldn't generate a response to your query."
    
    def get_suggested_questions(self, state: WorkflowState) -> List[str]:
        """Get suggested questions based on current analysis."""
        suggestions = []
        
        # Basic questions
        suggestions.extend([
            "What is the client's risk profile?",
            "Why are these products recommended?",
            "What are the key talking points for the meeting?"
        ])
        
        # Context-specific suggestions
        if state.analysis.risk_assessment:
            if state.analysis.risk_assessment.risk_level == "High":
                suggestions.append("How should I discuss high-risk investments with this client?")
            elif state.analysis.risk_assessment.risk_level == "Low":
                suggestions.append("What conservative options should I emphasize?")
        
        if state.analysis.persona_classification:
            persona = state.analysis.persona_classification.persona_type
            if persona == "Aggressive Growth":
                suggestions.append("What growth opportunities should I highlight?")
            elif persona == "Cautious Planner":
                suggestions.append("How can I address their risk concerns?")
        
        if state.recommendations.recommended_products:
            suggestions.append("What are the pros and cons of the top recommendation?")
            suggestions.append("How do I handle objections about fees?")
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for RM assistance."""
        return state.chat.current_query is not None and len(state.chat.current_query.strip()) > 0
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate RM assistance output."""
        return state.chat.response is not None and len(state.chat.response.strip()) > 0