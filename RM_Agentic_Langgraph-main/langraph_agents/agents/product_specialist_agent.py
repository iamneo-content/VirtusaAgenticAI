"""Product Specialist Agent for intelligent product recommendations."""

import pandas as pd
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from ..base_agent import CriticalAgent
from state import WorkflowState, ProductRecommendation
from config.settings import get_settings


class ProductSpecialistAgent(CriticalAgent):
    """Agent responsible for intelligent product recommendations and justifications."""
    
    def __init__(self):
        super().__init__(
            name="Product Specialist Agent",
            description="Provides intelligent product recommendations based on client profile and analysis"
        )
        self.settings = get_settings()
        self.products_df = None
        self._load_products()
    
    def _load_products(self):
        """Load product catalog."""
        try:
            self.products_df = pd.read_csv(self.settings.products_csv)
            self.logger.info(f"Loaded {len(self.products_df)} products from catalog")
        except Exception as e:
            self.logger.error(f"Failed to load products: {str(e)}")
            # Create dummy products for testing
            self.products_df = self._create_dummy_products()
    
    def _create_dummy_products(self) -> pd.DataFrame:
        """Create dummy products for testing."""
        return pd.DataFrame([
            {
                "product_id": "MF001",
                "product_name": "Growth Equity Fund",
                "product_type": "Mutual Fund",
                "risk_level": "High",
                "min_investment": 5000,
                "expected_return": "12-15%",
                "expense_ratio": "1.2%",
                "category": "Equity"
            },
            {
                "product_id": "MF002", 
                "product_name": "Balanced Advantage Fund",
                "product_type": "Mutual Fund",
                "risk_level": "Moderate",
                "min_investment": 1000,
                "expected_return": "8-12%",
                "expense_ratio": "1.5%",
                "category": "Hybrid"
            },
            {
                "product_id": "FD001",
                "product_name": "Fixed Deposit",
                "product_type": "Fixed Deposit",
                "risk_level": "Low",
                "min_investment": 1000,
                "expected_return": "6-7%",
                "expense_ratio": "0%",
                "category": "Debt"
            }
        ])
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute product recommendation."""
        self.logger.info("Starting product recommendation")
        
        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        persona_classification = state.analysis.persona_classification
        
        if not prospect_data or not risk_assessment:
            raise ValueError("Missing required data for product recommendation")
        
        # Filter products based on profile
        suitable_products = self._filter_products(prospect_data, risk_assessment, persona_classification)
        
        # Generate AI-powered recommendations
        recommendations = await self._generate_recommendations(
            prospect_data, risk_assessment, persona_classification, suitable_products
        )
        
        # Generate justification text
        justification = await self._generate_justification(
            prospect_data, risk_assessment, persona_classification, recommendations
        )
        
        # Update state
        state.recommendations.recommended_products = recommendations
        state.recommendations.justification_text = justification
        
        self.logger.info(f"Generated {len(recommendations)} product recommendations")
        return state
    
    def _filter_products(self, prospect_data, risk_assessment, persona_classification) -> pd.DataFrame:
        """Filter products based on client profile."""
        if self.products_df is None or self.products_df.empty:
            return pd.DataFrame()
        
        filtered_df = self.products_df.copy()
        
        # Filter by risk level
        risk_mapping = {
            "Low": ["Low"],
            "Moderate": ["Low", "Moderate"],
            "High": ["Low", "Moderate", "High"]
        }
        
        suitable_risk_levels = risk_mapping.get(risk_assessment.risk_level, ["Low"])
        filtered_df = filtered_df[filtered_df['risk_level'].isin(suitable_risk_levels)]
        
        # Filter by minimum investment
        if prospect_data.current_savings > 0:
            max_investment = min(prospect_data.current_savings * 0.8, 500000)  # Max 80% of savings or 5L
            filtered_df = filtered_df[filtered_df['min_investment'] <= max_investment]
        
        # Persona-based filtering
        if persona_classification:
            if persona_classification.persona_type == "Aggressive Growth":
                # Prefer equity and high-growth products
                filtered_df = filtered_df.sort_values('risk_level', ascending=False)
            elif persona_classification.persona_type == "Cautious Planner":
                # Prefer debt and low-risk products
                filtered_df = filtered_df[filtered_df['risk_level'] == 'Low']
        
        return filtered_df.head(10)  # Limit to top 10 products
    
    async def _generate_recommendations(
        self, 
        prospect_data, 
        risk_assessment, 
        persona_classification, 
        suitable_products: pd.DataFrame
    ) -> List[ProductRecommendation]:
        """Generate AI-powered product recommendations."""
        
        if suitable_products.empty:
            return []
        
        recommendations = []
        
        for _, product in suitable_products.iterrows():
            # Calculate suitability score
            suitability_score = self._calculate_suitability_score(
                product, prospect_data, risk_assessment, persona_classification
            )
            
            # Generate AI justification for this product
            justification = await self._generate_product_justification(
                product, prospect_data, risk_assessment, persona_classification
            )
            
            recommendation = ProductRecommendation(
                product_id=product['product_id'],
                product_name=product['product_name'],
                product_type=product['product_type'],
                suitability_score=suitability_score,
                justification=justification,
                risk_alignment=product['risk_level'],
                expected_returns=product.get('expected_return'),
                fees=product.get('expense_ratio')
            )
            
            recommendations.append(recommendation)
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _calculate_suitability_score(self, product, prospect_data, risk_assessment, persona_classification) -> float:
        """Calculate suitability score for a product."""
        score = 0.5  # Base score
        
        # Risk alignment
        if product['risk_level'] == risk_assessment.risk_level:
            score += 0.3
        elif (product['risk_level'] == 'Moderate' and risk_assessment.risk_level in ['Low', 'High']):
            score += 0.1
        
        # Investment amount alignment
        if product['min_investment'] <= prospect_data.current_savings * 0.1:
            score += 0.1
        
        # Persona alignment
        if persona_classification:
            if (persona_classification.persona_type == "Aggressive Growth" and 
                product['risk_level'] == 'High'):
                score += 0.1
            elif (persona_classification.persona_type == "Cautious Planner" and 
                  product['risk_level'] == 'Low'):
                score += 0.1
        
        return min(1.0, score)
    
    async def _generate_product_justification(
        self, 
        product, 
        prospect_data, 
        risk_assessment, 
        persona_classification
    ) -> str:
        """Generate AI justification for a specific product."""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Generate a concise justification for recommending this product to the prospect:
            
            Product Details:
            - Name: {product_name}
            - Type: {product_type}
            - Risk Level: {risk_level}
            - Expected Return: {expected_return}
            - Minimum Investment: ₹{min_investment:,}
            
            Prospect Profile:
            - Age: {age}
            - Annual Income: ₹{annual_income:,}
            - Current Savings: ₹{current_savings:,}
            - Investment Horizon: {investment_horizon_years} years
            - Risk Profile: {risk_profile}
            - Persona: {persona_type}
            
            Provide a 2-3 sentence justification explaining why this product is suitable.
            """)
        ])
        
        input_variables = {
            "product_name": product['product_name'],
            "product_type": product['product_type'],
            "risk_level": product['risk_level'],
            "expected_return": product.get('expected_return', 'N/A'),
            "min_investment": product['min_investment'],
            "age": prospect_data.age,
            "annual_income": prospect_data.annual_income,
            "current_savings": prospect_data.current_savings,
            "investment_horizon_years": prospect_data.investment_horizon_years,
            "risk_profile": risk_assessment.risk_level,
            "persona_type": persona_classification.persona_type if persona_classification else "N/A"
        }
        
        return await self.generate_response(prompt_template, input_variables)
    
    async def _generate_justification(
        self, 
        prospect_data, 
        risk_assessment, 
        persona_classification, 
        recommendations: List[ProductRecommendation]
    ) -> str:
        """Generate overall justification for the recommendation set."""
        
        prompt_template = self.get_prompt_template()
        
        products_summary = "\n".join([
            f"- {rec.product_name} ({rec.product_type}): {rec.justification}"
            for rec in recommendations[:3]  # Top 3 products
        ])
        
        input_variables = {
            "prospect_data": prospect_data.dict(),
            "risk_assessment": risk_assessment.dict(),
            "persona_type": persona_classification.persona_type if persona_classification else "N/A",
            "products_summary": products_summary,
            "num_recommendations": len(recommendations)
        }
        
        return await self.generate_response(prompt_template, input_variables)
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get prompt template for overall justification."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Generate a comprehensive justification for the product recommendations:
            
            Prospect Profile:
            {prospect_data}
            
            Risk Assessment:
            {risk_assessment}
            
            Persona Type: {persona_type}
            
            Recommended Products ({num_recommendations} total):
            {products_summary}
            
            Provide a comprehensive justification that:
            1. Explains the overall investment strategy
            2. Connects the recommendations to the client's profile
            3. Addresses risk management
            4. Highlights key benefits
            5. Mentions diversification if applicable
            
            Keep it professional and client-focused.
            """)
        ])
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for product recommendation."""
        return (
            state.prospect.prospect_data is not None and
            state.analysis.risk_assessment is not None
        )
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate product recommendation output."""
        return (
            len(state.recommendations.recommended_products) > 0 and
            state.recommendations.justification_text is not None
        )