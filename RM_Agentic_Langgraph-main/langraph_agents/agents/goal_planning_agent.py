"""Goal Planning Agent for investment goal success prediction and analysis."""

import joblib
import pandas as pd
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from ..base_agent import CriticalAgent
from state import WorkflowState, GoalPredictionResult
from config.settings import get_settings


class GoalPlanningAgent(CriticalAgent):
    """Agent responsible for goal success prediction and investment planning analysis."""
    
    def __init__(self):
        super().__init__(
            name="Goal Planning Agent",
            description="Analyzes investment goals and predicts success probability with strategic recommendations"
        )
        self.settings = get_settings()
        self.goal_model = None
        self.goal_encoders = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained goal prediction models."""
        try:
            self.goal_model = joblib.load(self.settings.goal_model_path)
            self.goal_encoders = joblib.load(self.settings.goal_encoders_path)
            self.logger.info("Goal prediction models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load goal models: {str(e)}")
            # Continue without models - will use rule-based prediction
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute goal planning analysis."""
        self.logger.info("Starting goal planning analysis")
        
        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        
        if not prospect_data:
            raise ValueError("No prospect data available for goal planning")
        
        # Perform ML-based goal prediction
        ml_prediction = await self._ml_goal_prediction(prospect_data)
        
        # Perform AI-based goal analysis
        ai_analysis = await self._ai_goal_analysis(prospect_data, risk_assessment, ml_prediction)
        
        # Create comprehensive goal prediction result
        goal_result = GoalPredictionResult(
            goal_success=ml_prediction['goal_success'],
            probability=ml_prediction['probability'],
            success_factors=ai_analysis['success_factors'],
            challenges=ai_analysis['challenges'],
            timeline_analysis=ai_analysis['timeline_analysis']
        )
        
        # Update state
        state.analysis.goal_prediction = goal_result
        
        self.logger.info(f"Goal planning completed. Success probability: {goal_result.probability:.1%}")
        return state
    
    async def _ml_goal_prediction(self, prospect_data) -> Dict[str, Any]:
        """Perform ML-based goal success prediction."""
        if not self.goal_model or not self.goal_encoders:
            return self._rule_based_goal_prediction(prospect_data)
        
        try:
            # Prepare input data
            input_data = {
                "age": prospect_data.age,
                "annual_income": prospect_data.annual_income,
                "current_savings": prospect_data.current_savings,
                "target_goal_amount": prospect_data.target_goal_amount,
                "investment_experience_level": prospect_data.investment_experience_level,
                "investment_horizon_years": prospect_data.investment_horizon_years,
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col, encoder in self.goal_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                    except ValueError:
                        # Handle unseen categories
                        input_df[col] = encoder.transform([encoder.classes_[0]])[0]
            
            # Make prediction
            if hasattr(self.goal_model, 'predict_proba'):
                # Classification model
                probabilities = self.goal_model.predict_proba(input_df)[0]
                prediction = self.goal_model.predict(input_df)[0]
                
                # Assuming binary classification: 0=Unlikely, 1=Likely
                goal_success = "Likely" if prediction == 1 else "Unlikely"
                probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            else:
                # Regression model - predict probability directly
                probability = float(self.goal_model.predict(input_df)[0])
                goal_success = "Likely" if probability > 0.6 else "Unlikely"
            
            return {
                "goal_success": goal_success,
                "probability": probability,
                "model_type": "ML"
            }
            
        except Exception as e:
            self.logger.error(f"ML goal prediction failed: {str(e)}")
            return self._rule_based_goal_prediction(prospect_data)
    
    def _rule_based_goal_prediction(self, prospect_data) -> Dict[str, Any]:
        """Fallback rule-based goal prediction."""
        # Calculate required monthly investment
        target_amount = prospect_data.target_goal_amount
        current_savings = prospect_data.current_savings
        years = prospect_data.investment_horizon_years
        annual_income = prospect_data.annual_income
        
        # Assume 8% annual return for calculation
        required_amount = target_amount - current_savings
        
        # Simple future value calculation for required monthly investment
        if years > 0:
            monthly_rate = 0.08 / 12  # 8% annual return
            months = years * 12
            
            if monthly_rate > 0:
                # PMT calculation for annuity
                required_monthly = required_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
            else:
                required_monthly = required_amount / months
        else:
            required_monthly = float('inf')
        
        # Calculate affordability
        monthly_income = annual_income / 12
        affordable_investment = monthly_income * 0.2  # Assume 20% of income can be invested
        
        # Determine success probability
        if required_monthly <= affordable_investment * 0.5:
            probability = 0.9  # Very achievable
            goal_success = "Likely"
        elif required_monthly <= affordable_investment:
            probability = 0.7  # Achievable with discipline
            goal_success = "Likely"
        elif required_monthly <= affordable_investment * 1.5:
            probability = 0.4  # Challenging but possible
            goal_success = "Unlikely"
        else:
            probability = 0.2  # Very challenging
            goal_success = "Unlikely"
        
        return {
            "goal_success": goal_success,
            "probability": probability,
            "required_monthly_investment": required_monthly,
            "affordable_monthly_investment": affordable_investment,
            "model_type": "Rule-based"
        }
    
    async def _ai_goal_analysis(self, prospect_data, risk_assessment, ml_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-based goal analysis for insights."""
        prompt_template = self.get_prompt_template()
        
        input_variables = {
            "prospect_data": prospect_data.dict(),
            "risk_level": risk_assessment.risk_level if risk_assessment else "Unknown",
            "goal_success": ml_prediction['goal_success'],
            "probability": ml_prediction['probability'],
            "required_monthly": ml_prediction.get('required_monthly_investment', 0)
        }
        
        response = await self.generate_response(prompt_template, input_variables)
        
        # Parse AI response
        return self._parse_goal_analysis(response)
    
    def _parse_goal_analysis(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI goal analysis response."""
        lines = ai_response.split('\n')
        
        success_factors = []
        challenges = []
        timeline_insights = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'success factors' in line.lower():
                current_section = 'success'
            elif 'challenges' in line.lower() or 'risks' in line.lower():
                current_section = 'challenges'
            elif 'timeline' in line.lower():
                current_section = 'timeline'
            elif line.startswith(('-', '•', '*')):
                item = line[1:].strip()
                if current_section == 'success':
                    success_factors.append(item)
                elif current_section == 'challenges':
                    challenges.append(item)
                elif current_section == 'timeline':
                    timeline_insights.append(item)
        
        # Default values if parsing fails
        if not success_factors:
            success_factors = [
                "Consistent investment discipline",
                "Long-term market growth",
                "Regular portfolio review"
            ]
        
        if not challenges:
            challenges = [
                "Market volatility risks",
                "Inflation impact",
                "Changing life circumstances"
            ]
        
        timeline_analysis = {
            "short_term": "Focus on building investment habit",
            "medium_term": "Monitor progress and adjust strategy",
            "long_term": "Stay committed to long-term goals",
            "insights": timeline_insights
        }
        
        return {
            "success_factors": success_factors,
            "challenges": challenges,
            "timeline_analysis": timeline_analysis
        }
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get prompt template for goal analysis."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Analyze the investment goal feasibility for this prospect:
            
            Prospect Profile:
            {prospect_data}
            
            Risk Assessment: {risk_level}
            
            Goal Prediction:
            - Success Likelihood: {goal_success}
            - Probability: {probability:.1%}
            - Required Monthly Investment: ₹{required_monthly:,.0f}
            
            Provide detailed analysis covering:
            
            Success Factors:
            - List factors that support goal achievement
            - Consider client's strengths and advantages
            - Include market and strategy factors
            
            Challenges:
            - Identify potential obstacles and risks
            - Consider market, personal, and economic factors
            - Highlight areas requiring attention
            
            Timeline Analysis:
            - Short-term milestones (1-2 years)
            - Medium-term checkpoints (3-5 years)
            - Long-term considerations
            - Key review points
            
            Format your response with clear sections and bullet points.
            """)
        ])
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for goal planning."""
        return (
            state.prospect.prospect_data is not None and
            state.prospect.prospect_data.target_goal_amount > 0 and
            state.prospect.prospect_data.investment_horizon_years > 0
        )
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate goal planning output."""
        return (
            state.analysis.goal_prediction is not None and
            state.analysis.goal_prediction.goal_success in ["Likely", "Unlikely"] and
            0 <= state.analysis.goal_prediction.probability <= 1
        )