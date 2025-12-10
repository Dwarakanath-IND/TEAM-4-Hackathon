# TODO: Create PortfolioOptimizerAgent class inheriting from BaseAgent
# TODO: Implement optimize_allocation creating balanced portfolio
# TODO: Implement apply_risk_constraints limiting allocation based on risk
# TODO: Implement calculate_diversification metrics
# TODO: Implement validate_allocation checking portfolio constraints
# TODO: Implement async run method:
#   - Get risk assessment and recommended products
#   - Create initial allocation based on modern portfolio theory
#   - Apply risk-based constraints
#   - Optimize for diversification
#   - Validate allocation meets requirements
#   - Return portfolio_allocation dict in state


"""Portfolio Optimizer Agent for investment allocation optimization."""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from langraph_agents.base_agent import OptionalAgent
from state import WorkflowState
from config.settings import get_settings


class PortfolioOptimizerAgent(OptionalAgent):
    """Agent responsible for optimizing portfolio allocation based on client profile and recommendations."""
    
    def __init__(self):
        super().__init__(
            name="Portfolio Optimizer Agent",
            description="Optimizes portfolio allocation and provides asset allocation recommendations"
        )
        self.settings = get_settings()
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute portfolio optimization."""
        self.logger.info("Starting portfolio optimization")
        
        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        recommendations = state.recommendations.recommended_products
        
        if not prospect_data or not recommendations:
            self.logger.warning("Insufficient data for portfolio optimization")
            return state
        
        # Generate portfolio allocation
        portfolio_allocation = await self._generate_portfolio_allocation(
            prospect_data, risk_assessment, recommendations
        )
        
        # Update state
        state.recommendations.portfolio_allocation = portfolio_allocation
        
        self.logger.info("Portfolio optimization completed")
        return state
    
    async def _generate_portfolio_allocation(
        self, 
        prospect_data, 
        risk_assessment, 
        recommendations
    ) -> Dict[str, float]:
        """Generate optimal portfolio allocation."""
        
        # Simple rule-based allocation for now
        # In production, this would use sophisticated optimization algorithms
        
        allocation = {}
        total_allocation = 100.0
        
        if risk_assessment:
            if risk_assessment.risk_level == "High":
                # Aggressive allocation
                allocation = {
                    "Equity": 70.0,
                    "Debt": 20.0,
                    "Alternative": 10.0
                }
            elif risk_assessment.risk_level == "Moderate":
                # Balanced allocation
                allocation = {
                    "Equity": 50.0,
                    "Debt": 40.0,
                    "Alternative": 10.0
                }
            else:  # Low risk
                # Conservative allocation
                allocation = {
                    "Equity": 30.0,
                    "Debt": 60.0,
                    "Cash": 10.0
                }
        else:
            # Default balanced allocation
            allocation = {
                "Equity": 50.0,
                "Debt": 40.0,
                "Cash": 10.0
            }
        
        # Adjust based on age (simple age-based rule)
        if prospect_data.age > 50:
            # Reduce equity allocation for older clients
            if "Equity" in allocation:
                equity_reduction = min(10.0, allocation["Equity"])
                allocation["Equity"] -= equity_reduction
                allocation["Debt"] = allocation.get("Debt", 0) + equity_reduction
        
        return allocation
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get prompt template for portfolio optimization."""
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Optimize portfolio allocation for this client:
            
            Client Profile: {prospect_data}
            Risk Assessment: {risk_assessment}
            Available Products: {recommendations}
            
            Provide optimal asset allocation percentages.
            """)
        ])
    
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for portfolio optimization."""
        return (
            state.prospect.prospect_data is not None and
            len(state.recommendations.recommended_products) > 0
        )
    
    def validate_output(self, state: WorkflowState) -> bool:
        """Validate portfolio optimization output."""
        return state.recommendations.portfolio_allocation is not None
