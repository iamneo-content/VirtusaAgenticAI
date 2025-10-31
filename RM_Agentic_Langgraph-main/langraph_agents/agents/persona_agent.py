"""Persona Agent for client behavioral classification."""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from ..base_agent import BaseAgent
from state import WorkflowState, PersonaResult
from config.settings import get_settings


class PersonaAgent(BaseAgent):
    """Agent responsible for classifying client personas and behavioral insights."""
    
    def __init__(self):
        super().__init__(
            name="Persona Agent",
            description="Classifies client personas and provides behavioral insights for personalized advisory"
        )
        self.settings = get_settings()
        
        # Define persona types and their characteristics
        self.persona_types = {
            "Aggressive Growth": {
                "description": "High risk tolerance, seeks maximum returns, comfortable with volatility",
                "characteristics": ["High risk tolerance", "Growth-focused", "Long-term oriented", "Market-savvy"],
                "typical_profile": "Young professionals, high income, long investment horizon"
            },
            "Steady Saver": {
                "description": "Balanced approach, consistent investments, moderate risk tolerance",
                "characteristics": ["Consistent investor", "Balanced risk approach", "Goal-oriented", "Disciplined"],
                "typical_profile": "Middle-aged professionals, stable income, medium-term goals"
            },
            "Cautious Planner": {
                "description": "Conservative approach, capital preservation focus, low risk tolerance",
                "characteristics": ["Risk-averse", "Capital preservation", "Security-focused", "Conservative"],
                "typical_profile": "Pre-retirees, risk-averse individuals, short-term goals"
            }
        }
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute persona classification."""
        self.logger.info("Starting persona classification")
        
        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        
        if not prospect_data:
            raise ValueError("No prospect data available for persona classification")
        
        # Perform AI-based persona classification
        persona_result = await self._classify_persona(prospect_data, risk_assessment)
        
        # Enhance with behavioral insights
        behavioral_insights = await self._generate_behavioral_insights(prospect_data, persona_result)
        
        # Create final persona result
        final_result = PersonaResult(
            persona_type=persona_result['persona_type'],
            confidence_score=persona_result['confidence_score'],
            characteristics=self.persona_types[persona_result['persona_type']]['characteristics'],
            behavioral_insights=behavioral_insights
        )
        
        # Update state
        state.analysis.persona_classification = final_result
        
        self.logger.info(f"Persona classification completed: {final_result.persona_type}")
        return state
    
    async def _classify_persona(self, prospect_data, risk_assessment) -> Dict[str, Any]:
        """Classify client persona using AI."""
        prompt_template = self.get_classification_prompt()
        
        # Prepare context
        risk_info = ""
        if risk_assessment:
            risk_info = f"Risk Level: {risk_assessment.risk_level}, Confidence: {risk_assessment.confidence_score}"
        
        input_variables = {
            "prospect_data": prospect_data.dict(),
            "risk_assessment": risk_info,
            "persona_types": self._format_persona_types()
        }
        
        response = await self.generate_response(prompt_template, input_variables)
        
        # Parse response to extract persona type and confidence
        persona_type = self._extract_persona_type(response)
        confidence_score = self._calculate_confidence_score(prospect_data, persona_type)
        
        return {
            "persona_type": persona_type,
            "confidence_score": confidence_score,
            "ai_reasoning": response
        }
    
    def _extract_persona_type(self, ai_response: str) -> str:
        """Extract persona type from AI response."""
        response_lower = ai_response.lower()
        
        for persona_type in self.persona_types.keys():
            if persona_type.lower() in response_lower:
                return persona_type
        
        # Fallback logic based on keywords
        if any(word in response_lower for word in ['aggressive', 'growth', 'high risk']):
            return "Aggressive Growth"
        elif any(word in response_lower for word in ['cautious', 'conservative', 'low risk']):
            return "Cautious Planner"
        else:
            return "Steady Saver"  # Default
    
    def _calculate_confidence_score(self, prospect_data, persona_type: str) -> float:
        """Calculate confidence score based on data alignment."""
        score = 0.5  # Base score
        
        # Age alignment
        if persona_type == "Aggressive Growth" and prospect_data.age < 35:
            score += 0.2
        elif persona_type == "Cautious Planner" and prospect_data.age > 50:
            score += 0.2
        elif persona_type == "Steady Saver" and 30 <= prospect_data.age <= 55:
            score += 0.2
        
        # Investment horizon alignment
        if persona_type == "Aggressive Growth" and prospect_data.investment_horizon_years > 10:
            score += 0.15
        elif persona_type == "Cautious Planner" and prospect_data.investment_horizon_years < 5:
            score += 0.15
        
        # Experience level alignment
        experience_mapping = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        experience_score = experience_mapping.get(prospect_data.investment_experience_level, 0)
        
        if persona_type == "Aggressive Growth" and experience_score >= 1:
            score += 0.1
        elif persona_type == "Cautious Planner" and experience_score == 0:
            score += 0.1
        
        # Income alignment
        if prospect_data.annual_income > 1000000 and persona_type == "Aggressive Growth":
            score += 0.05
        
        return min(1.0, score)
    
    async def _generate_behavioral_insights(self, prospect_data, persona_result: Dict[str, Any]) -> List[str]:
        """Generate behavioral insights for the classified persona."""
        prompt_template = self.get_insights_prompt()
        
        input_variables = {
            "prospect_data": prospect_data.dict(),
            "persona_type": persona_result['persona_type'],
            "persona_description": self.persona_types[persona_result['persona_type']]['description']
        }
        
        response = await self.generate_response(prompt_template, input_variables)
        
        # Parse insights from response
        insights = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                insight = line[1:].strip()
                if insight:
                    insights.append(insight)
        
        return insights or ["Standard behavioral patterns apply for this persona type"]
    
    def _format_persona_types(self) -> str:
        """Format persona types for prompt."""
        formatted = ""
        for persona_type, info in self.persona_types.items():
            formatted += f"\n{persona_type}: {info['description']}\n"
            formatted += f"Typical Profile: {info['typical_profile']}\n"
        return formatted
    
    def get_classification_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for persona classification."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Classify the following prospect into one of the defined persona types based on their profile and risk assessment:
            
            Prospect Data:
            {prospect_data}
            
            Risk Assessment:
            {risk_assessment}
            
            Available Persona Types:
            {persona_types}
            
            Instructions:
            1. Analyze the prospect's age, income, investment horizon, experience level, and risk profile
            2. Consider their financial goals and current situation
            3. Match them to the most appropriate persona type
            4. Provide clear reasoning for your classification
            
            Respond with the persona type name and your reasoning.
            """)
        ])
    
    def get_insights_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for behavioral insights."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Generate specific behavioral insights for this prospect based on their classified persona:
            
            Prospect Data:
            {prospect_data}
            
            Classified Persona: {persona_type}
            Persona Description: {persona_description}
            
            Provide behavioral insights that will help the relationship manager:
            - Communication preferences
            - Decision-making patterns
            - Likely concerns or objections
            - Motivation factors
            - Preferred investment approaches
            
            Format as bullet points with actionable insights.
            """)
        ])
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Default prompt template."""
        return self.get_classification_prompt()
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for persona classification."""
        return state.prospect.prospect_data is not None
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate persona classification output."""
        return (
            state.analysis.persona_classification is not None and
            state.analysis.persona_classification.persona_type in self.persona_types.keys()
        )