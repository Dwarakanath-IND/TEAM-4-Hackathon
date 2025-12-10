# TODO: Create ComplianceAgent class inheriting from BaseAgent
# TODO: Implement check_suitability_compliance verifying product matches investor profile
# TODO: Implement check_regulatory_compliance validating against regulations
# TODO: Implement generate_required_disclosures creating disclosure list
# TODO: Implement identify_potential_violations finding compliance issues
# TODO: Implement async run method:
#   - Review recommendations and prospect profile
#   - Check suitability of recommended products
#   - Verify regulatory compliance
#   - Generate required disclosures
#   - Identify any violations or warnings
#   - Calculate compliance score
#   - Return ComplianceCheck in state

from typing import Dict,Any,List
from langchain_core.prompts import ChatPromptTemplate


from langraph_agents.base_agent import CriticalAgent
from state import WorkflowState, ComplianceCheck
from config.settings import get_settings #check this out too 


class ComplianceAgent(CriticalAgent):
    def __init__(self):
        super().__init__(
            name = "Compliance Agent",
            description= "Ensures regulatory compliance and conducts risked-based compliance checks"
        )
        self.settings = get_settings()

        self.compliance_rules = {
            "max_single_product_allocation":0.6,
            "min_diversification_products":2,
            "high_risk_age_limit":65,
            "max_investment_to_income_ratio":0.3,
            "min_emergency_fund_months":6
        }
    async def execute(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("Starting compliance checks")

        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        recommendations = state.recommendations.recommended_product

        if not prospect_data or not recommendations:
            self.logger.warning("THere is insuficient data for compliance checks")
            return state

        compliance_result = await self._perform_compliance_checks(
            prospect_data, risk_assessment, recommendations
        )

        state.recommendations.compliance_check =  compliance_result

        compliance_status = "COMPLIANT" if compliance_result.is_compliant else "NON-COMPLIANT"
        self.logger.info(f"The compliance checks are completed: {compliance_status}")

        return state

    async def _perform_compliance_checks(self,prospect_data,risk_assessment,recommendations) -> ComplianceCheck:
        
        violations = []
        warnings = []
        required_disclosures = []

        if prospect_data.age > self.compliance_rules["high_risk_age_limit"]:
            high_risk_products = [r for r in recommendations if r.risk_alignment == "High"]
            if high_risk_products:
                violations.append(
                    f"High-risk products recommended for client aged {prospect_data.age}" 
                    f"(The limit is: {self.compliance_rules['high_risk_age_limit']})"
                )
        
        total_investment = sum([
            prospect_data.current_savings * 0.1 for _ in recommendations     #Assuming 10pc of savings per product
        ])

        investment_ratio = total_investment/prospect_data.annual_income

        if investment_ratio > self.compliance_rules["max_investment_to_income_ratio"]:
            warnings.append(
                f"The Recommended investment ({investment_ratio:.1%}) exceeds "
                f"{self.compliance_rules['max_investment_to_income_ratio']:.1%} of annual income"
            )

        if len(recommendations) < self.compliance_rules["min_diversification_products"]:
            warnings.append(
                f"THere is insufficient diversification: {len(recommendations)} products "
                f"(Minimum: {self.compliance_rules['min_diversification_products']})"
            )

        monthly_expenses  = prospect_data.annual_income / 12 * 0.7
        emergency_fund_needed = monthly_expenses * self.compliance_rules['min_emergency_fund_months']

        if prospect_data.current_savings < emergency_fund_needed:
            warnings.append(
                f"There is insufficient emergency fund: Rupees {prospect_data.current_savings} "
                f"(Recommended: Rupees{emergency_fund_needed})"
            )

        if risk_assessment:
            misaligned_products = []
            for rec in recommendations:
                if (risk_assessment.risk_level =="Low" and rec.risk_alignment == "Hihg") or (risk_assessment.risk_level =="High" and rec.risk_alignment=="Low"):
                    misaligned_products.append(rec.product_name)
            
            if misaligned_products:
                warnings.append(
                    f"Risk misalignment detected for products: {','.join(misaligned_products)}"
                )
        
        required_disclosures = await self._generate_required_disclosures(prospect_data, risk_assessment, recommendations, violations, warnings)


        compliance_score = self._calculate_compliance_score(violations,warnings)

        is_compliant = len(violations)==0 and compliance_score >=0.7

        return ComplianceCheck(
            is_compliant = is_compliant,
            compliance_score = compliance_score,
            violations = violations,
            warnings = warnings,
            required_disclosures = required_disclosures
        )
    
    def _calculate_compliance_score(self, violations: List[str], warnings: List[str]) -> float:

        base_score = 1.0

        violation_penalty = len(violations)* 0.3

        warning_penalty = len(warnings) * 0.1

        final_score = max(0.0,base_score - violation_penalty - warning_penalty)

        return final_score

    async def _generate_required_disclosures(self,prospect_data,risk_assessment,recommendations,violations: List[str],warnings: List[str])-> List[str]:

        disclosures = [
            "Investment products are subject to market risks",
            "Please read all scheme-related documents carefully before investing",
            "Past performance is not an indicator of future returns"
        ]

        if risk_assessment and risk_assessment.risk_level == "High":
            disclosures.extend([
                "High-risk investments may result in significant losses",
                "Suitable only for investors with high risk tolerance",
                "Regular monitoring and review recommended"
            ])

        product_types = set([rec.product_type for rec in recommendations])

        if "Mutual Fund" in product_types:
            disclosures.append("Mutual fund investments are subject to market risks")
        
        if "ELSS" in product_types:
            disclosures.append("ELSS investments have a mandatory lock-in period of 3 years")

        if "Fixed Deposit" in product_types:
            disclosures.append("Fixed Deposits are subject to credit risk")

        if violations:
            disclosures.append("Please review compliance violations before proceeding")

        if warnings:
            disclosures.append("Please conisder compliance warnings before proceeding")
    
        return disclosures

    def get_prompt_template(self)-> ChatPromptTemplate:

        return ChatPromptTemplate.from_messages([
            ("system",self.get_system_prompt()),
            ("human","""
            Please perform regulatory compliance analysis for this investment recommendation:
            
            Client Profile: {prospect_data}
            Risk Assessment: {risk_assessment}
            Recommendations: {recommendations}

            Check for:
            1. Risk allignment
            2. Age-based suitability
            3. Investment limits
            4. Diversification requirements
            5. Regulatory disclosures needed

            Identify any compliance violations or warnings
            """)
        ])

    def validate_input(self, state: WorkflowState) -> bool:
        return(
            state.prospect.prospect_data is not None and
            len(state.recommendations.recommended_product) > 0
        )

    def validate_output(self, state: WorkflowState) -> bool:
        return (
            state.recommendations.compliance_check is not None and
            state.recommendations.compliance_check.complaince_score is not None
        )
