"""Risk Assessment Agent for ML-based risk profiling."""

import joblib
import pandas as pd
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from ..base_agent import CriticalAgent
from state import WorkflowState, RiskAssessmentResult
from config.settings import get_settings


class RiskAssessmentAgent(CriticalAgent):
    """Agent responsible for risk profiling using ML models and AI analysis."""
    
    def __init__(self):
        super().__init__(
            name="Risk Assessment Agent",
            description="Performs comprehensive risk assessment using ML models and AI analysis"
        )
        self.settings = get_settings()
        self.risk_model = None
        self.label_encoders = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained ML models."""
        try:
            self.risk_model = joblib.load(self.settings.risk_model_path)
            self.label_encoders = joblib.load(self.settings.risk_encoders_path)
            self.logger.info("Risk assessment models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load risk models: {str(e)}")
            # Continue without models - will use AI-only assessment
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute risk assessment."""
        self.logger.info("Starting risk assessment")
        
        prospect_data = state.prospect.prospect_data
        if not prospect_data:
            raise ValueError("No prospect data available for risk assessment")
        
        # Perform ML-based risk assessment
        ml_risk_result = await self._ml_risk_assessment(prospect_data)
        
        # Perform AI-based risk analysis for additional insights
        ai_risk_analysis = await self._ai_risk_analysis(prospect_data, ml_risk_result)
        
        # Combine results
        risk_result = RiskAssessmentResult(
            risk_level=ml_risk_result['risk_level'],
            confidence_score=ml_risk_result['confidence_score'],
            risk_factors=ai_risk_analysis['risk_factors'],
            recommendations=ai_risk_analysis['recommendations']
        )
        
        # Update state
        state.analysis.risk_assessment = risk_result
        
        self.logger.info(f"Risk assessment completed. Risk level: {risk_result.risk_level}")
        return state
    
    async def _ml_risk_assessment(self, prospect_data) -> Dict[str, Any]:
        """Perform ML-based risk assessment."""
        if not self.risk_model or not self.label_encoders:
            # Fallback to rule-based assessment
            return self._rule_based_risk_assessment(prospect_data)
        
        try:
            # Prepare input data
            input_data = {
                "age": prospect_data.age,
                "annual_income": prospect_data.annual_income,
                "current_savings": prospect_data.current_savings,
                "target_goal_amount": prospect_data.target_goal_amount,
                "investment_horizon_years": prospect_data.investment_horizon_years,
                "number_of_dependents": prospect_data.number_of_dependents,
                "investment_experience_level": prospect_data.investment_experience_level,
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                    except ValueError:
                        # Handle unseen categories
                        input_df[col] = encoder.transform([encoder.classes_[0]])[0]
            
            # Make prediction
            prediction = self.risk_model.predict(input_df)[0]
            probabilities = self.risk_model.predict_proba(input_df)[0]
            
            # Map prediction to risk level
            risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
            # Handle both numeric and string predictions
            if isinstance(prediction, str):
                risk_level = prediction
            else:
                risk_level = risk_mapping.get(prediction, "Moderate")
            confidence_score = float(max(probabilities))
            
            return {
                "risk_level": risk_level,
                "confidence_score": confidence_score,
                "probabilities": {
                    "Low": float(probabilities[0]),
                    "Moderate": float(probabilities[1]),
                    "High": float(probabilities[2])
                }
            }
            
        except Exception as e:
            self.logger.error(f"ML risk assessment failed: {str(e)}")
            return self._rule_based_risk_assessment(prospect_data)
    
    def _rule_based_risk_assessment(self, prospect_data) -> Dict[str, Any]:
        """Fallback rule-based risk assessment."""
        risk_score = 0
        
        # Age factor
        if prospect_data.age < 30:
            risk_score += 2  # Young, can take more risk
        elif prospect_data.age < 50:
            risk_score += 1  # Middle-aged, moderate risk
        else:
            risk_score += 0  # Older, conservative
        
        # Income factor
        if prospect_data.annual_income > 1000000:
            risk_score += 2  # High income, can afford risk
        elif prospect_data.annual_income > 500000:
            risk_score += 1
        
        # Investment horizon
        if prospect_data.investment_horizon_years > 10:
            risk_score += 2  # Long horizon, can take risk
        elif prospect_data.investment_horizon_years > 5:
            risk_score += 1
        
        # Experience level
        experience_mapping = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        risk_score += experience_mapping.get(prospect_data.investment_experience_level, 0)
        
        # Dependents (reduces risk tolerance)
        if prospect_data.number_of_dependents > 2:
            risk_score -= 1
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = "High"
            confidence = 0.7
        elif risk_score >= 3:
            risk_level = "Moderate"
            confidence = 0.8
        else:
            risk_level = "Low"
            confidence = 0.7
        
        return {
            "risk_level": risk_level,
            "confidence_score": confidence,
            "rule_score": risk_score
        }
    
    async def _ai_risk_analysis(self, prospect_data, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-based risk factor analysis."""
        prompt_template = self.get_prompt_template()
        
        input_variables = {
            "prospect_data": prospect_data.dict(),
            "ml_risk_level": ml_result['risk_level'],
            "confidence_score": ml_result['confidence_score']
        }
        
        response = await self.generate_response(prompt_template, input_variables)
        
        # Parse AI response (in production, use structured output)
        lines = response.split('\n')
        risk_factors = []
        recommendations = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            if 'risk factors' in line.lower():
                current_section = 'factors'
            elif 'recommendations' in line.lower():
                current_section = 'recommendations'
            elif line.startswith('-') or line.startswith('•'):
                item = line[1:].strip()
                if current_section == 'factors':
                    risk_factors.append(item)
                elif current_section == 'recommendations':
                    recommendations.append(item)
        
        return {
            "risk_factors": risk_factors or ["Standard risk factors apply"],
            "recommendations": recommendations or ["Follow standard risk management practices"]
        }
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get prompt template for AI risk analysis."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Analyze the following prospect's risk profile and provide detailed insights:
            
            Prospect Data:
            {prospect_data}
            
            ML Model Assessment:
            - Risk Level: {ml_risk_level}
            - Confidence: {confidence_score}
            
            Please provide:
            
            Risk Factors:
            - List specific risk factors based on the prospect's profile
            - Consider age, income, investment horizon, experience, and dependents
            - Identify both positive and negative risk indicators
            
            Recommendations:
            - Provide specific risk management recommendations
            - Suggest appropriate investment strategies
            - Consider regulatory and compliance aspects
            
            Format your response with clear sections for Risk Factors and Recommendations.
            """)
        ])
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for risk assessment."""
        return (
            state.prospect.prospect_data is not None and
            state.prospect.data_quality_score is not None and
            state.prospect.data_quality_score > 0.5
        )
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate risk assessment output."""
        return (
            state.analysis.risk_assessment is not None and
            state.analysis.risk_assessment.risk_level in ["Low", "Moderate", "High"]
        )