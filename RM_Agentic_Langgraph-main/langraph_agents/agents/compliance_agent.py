"""Compliance Agent for regulatory compliance and risk checks."""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from ..base_agent import CriticalAgent
from state import WorkflowState, ComplianceCheck
from config.settings import get_settings


class ComplianceAgent(CriticalAgent):
    """Agent responsible for ensuring regulatory compliance and conducting compliance checks."""
    
    def __init__(self):
        super().__init__(
            name="Compliance Agent",
            description="Ensures regulatory compliance and conducts risk-based compliance checks"
        )
        self.settings = get_settings()
        
        # Define compliance rules and thresholds
        self.compliance_rules = {
            "max_single_product_allocation": 0.6,  # Max 60% in single product
            "min_diversification_products": 2,     # Minimum 2 products for diversification
            "high_risk_age_limit": 65,            # Age limit for high-risk products
            "max_investment_to_income_ratio": 0.3, # Max 30% of annual income
            "min_emergency_fund_months": 6         # Minimum 6 months emergency fund
        }
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute compliance checks."""
        self.logger.info("Starting compliance checks")
        
        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        recommendations = state.recommendations.recommended_products
        
        if not prospect_data or not recommendations:
            self.logger.warning("Insufficient data for compliance checks")
            return state
        
        # Perform comprehensive compliance checks
        compliance_result = await self._perform_compliance_checks(
            prospect_data, risk_assessment, recommendations
        )
        
        # Update state
        state.recommendations.compliance_check = compliance_result
        
        compliance_status = "COMPLIANT" if compliance_result.is_compliant else "NON-COMPLIANT"
        self.logger.info(f"Compliance checks completed: {compliance_status}")
        
        return state
    
    async def _perform_compliance_checks(
        self, 
        prospect_data, 
        risk_assessment, 
        recommendations
    ) -> ComplianceCheck:
        """Perform comprehensive compliance checks."""
        
        violations = []
        warnings = []
        required_disclosures = []
        
        # Check 1: Age-based risk suitability
        if prospect_data.age > self.compliance_rules["high_risk_age_limit"]:
            high_risk_products = [r for r in recommendations if r.risk_alignment == "High"]
            if high_risk_products:
                violations.append(
                    f"High-risk products recommended for client aged {prospect_data.age} "
                    f"(limit: {self.compliance_rules['high_risk_age_limit']})"
                )
        
        # Check 2: Investment to income ratio
        total_investment = sum([
            prospect_data.current_savings * 0.1  # Assume 10% of savings per product
            for _ in recommendations
        ])
        investment_ratio = total_investment / prospect_data.annual_income
        
        if investment_ratio > self.compliance_rules["max_investment_to_income_ratio"]:
            warnings.append(
                f"Recommended investment ({investment_ratio:.1%}) exceeds "
                f"{self.compliance_rules['max_investment_to_income_ratio']:.1%} of annual income"
            )
        
        # Check 3: Diversification requirements
        if len(recommendations) < self.compliance_rules["min_diversification_products"]:
            warnings.append(
                f"Insufficient diversification: {len(recommendations)} products "
                f"(minimum: {self.compliance_rules['min_diversification_products']})"
            )
        
        # Check 4: Emergency fund adequacy
        monthly_expenses = prospect_data.annual_income / 12 * 0.7  # Assume 70% of income as expenses
        emergency_fund_needed = monthly_expenses * self.compliance_rules["min_emergency_fund_months"]
        
        if prospect_data.current_savings < emergency_fund_needed:
            warnings.append(
                f"Insufficient emergency fund: ₹{prospect_data.current_savings:,} "
                f"(recommended: ₹{emergency_fund_needed:,})"
            )
        
        # Check 5: Risk alignment
        if risk_assessment:
            misaligned_products = []
            for rec in recommendations:
                if (risk_assessment.risk_level == "Low" and rec.risk_alignment == "High") or \
                   (risk_assessment.risk_level == "High" and rec.risk_alignment == "Low"):
                    misaligned_products.append(rec.product_name)
            
            if misaligned_products:
                warnings.append(
                    f"Risk misalignment detected for products: {', '.join(misaligned_products)}"
                )
        
        # Generate required disclosures
        required_disclosures = await self._generate_required_disclosures(
            prospect_data, risk_assessment, recommendations, violations, warnings
        )
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(violations, warnings)
        
        # Determine overall compliance status
        is_compliant = len(violations) == 0 and compliance_score >= 0.7
        
        return ComplianceCheck(
            is_compliant=is_compliant,
            compliance_score=compliance_score,
            violations=violations,
            warnings=warnings,
            required_disclosures=required_disclosures
        )
    
    def _calculate_compliance_score(self, violations: List[str], warnings: List[str]) -> float:
        """Calculate overall compliance score."""
        base_score = 1.0
        
        # Deduct for violations (major issues)
        violation_penalty = len(violations) * 0.3
        
        # Deduct for warnings (minor issues)
        warning_penalty = len(warnings) * 0.1
        
        final_score = max(0.0, base_score - violation_penalty - warning_penalty)
        return final_score
    
    async def _generate_required_disclosures(
        self, 
        prospect_data, 
        risk_assessment, 
        recommendations, 
        violations: List[str], 
        warnings: List[str]
    ) -> List[str]:
        """Generate required regulatory disclosures."""
        
        disclosures = [
            "Investment products are subject to market risks",
            "Past performance does not guarantee future results",
            "Please read all scheme-related documents carefully before investing"
        ]
        
        # Add risk-specific disclosures
        if risk_assessment and risk_assessment.risk_level == "High":
            disclosures.extend([
                "High-risk investments may result in significant losses",
                "Suitable only for investors with high risk tolerance",
                "Regular monitoring and review recommended"
            ])
        
        # Add product-specific disclosures
        product_types = set([rec.product_type for rec in recommendations])
        
        if "Mutual Fund" in product_types:
            disclosures.append("Mutual fund investments are subject to market risks")
        
        if "ELSS" in product_types:
            disclosures.append("ELSS investments have a mandatory lock-in period of 3 years")
        
        if "Fixed Deposit" in product_types:
            disclosures.append("Fixed deposits are subject to credit risk of the issuing bank")
        
        # Add compliance-specific disclosures
        if violations:
            disclosures.append("Please review compliance violations before proceeding")
        
        if warnings:
            disclosures.append("Please consider compliance warnings in your investment decision")
        
        return disclosures
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get prompt template for compliance analysis."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Perform regulatory compliance analysis for this investment recommendation:
            
            Client Profile: {prospect_data}
            Risk Assessment: {risk_assessment}
            Recommendations: {recommendations}
            
            Check for:
            1. Age-based suitability
            2. Risk alignment
            3. Investment limits
            4. Diversification requirements
            5. Regulatory disclosures needed
            
            Identify any compliance violations or warnings.
            """)
        ])
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for compliance checks."""
        return (
            state.prospect.prospect_data is not None and
            len(state.recommendations.recommended_products) > 0
        )
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate compliance check output."""
        return (
            state.recommendations.compliance_check is not None and
            state.recommendations.compliance_check.compliance_score is not None
        )