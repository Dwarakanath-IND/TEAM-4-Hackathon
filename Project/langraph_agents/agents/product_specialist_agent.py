# TODO: Create ProductSpecialistAgent class inheriting from BaseAgent
# TODO: Load products.csv data
# TODO: Implement match_products_to_profile matching products to risk level
# TODO: Implement calculate_suitability_score based on prospect-product fit
# TODO: Implement generate_justification creating recommendation reasoning
# TODO: Implement rank_products sorting by suitability
# TODO: Implement async run method:
#   - Get risk assessment and persona from state
#   - Load available products from products.csv
#   - Filter products matching risk level
#   - Calculate suitability scores for each product
#   - Rank products by score
#   - Generate justifications for top products
#   - Create ProductRecommendation list
#   - Return recommendations in state



import pandas as pd
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from langraph_agents.base_agent import CriticalAgent
from state import WorkflowState, ProductRecommendation, RecommendationState
from config.settings import get_settings
from functools import lru_cache
from config.logging_config import get_logger

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


class ProductSpecialistAgent(CriticalAgent):
    """Agent responsible for intelligent product recommendations and justifications."""
    
    def __init__(self):
        super().__init__(
            name="Product Specialist Agent",
            description="Provides intelligent product recommendations based on client profile and analysis"
        )
        self.settings = get_settings()
        self.logger = get_logger("ProductSpecialistAgent")
        self.products_df = None
        self._load_products()
        self.name = "Product Specialist Agent"
    
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

        if not self.validate_input(state):
            self.logger.error(f"Input validation failed for agent: {self.name}")
            state.complete_agent_execution(self.name, status="failed")
            return state
        

        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment
        #persona_classification = state.analysis.persona_classification
        persona_classification = getattr(state.analysis, "persona_classification", None)
        
        if not prospect_data or not risk_assessment:
            raise ValueError("Missing required data for product recommendation")
        
        # Filter products based on profile
        suitable_products = self._filter_products(prospect_data, risk_assessment, persona_classification)
        
        # Generate AI-powered recommendations
        recommendations = await self._generate_recommendations_batch(
            prospect_data, risk_assessment, persona_classification, suitable_products
        )
        
        # Generate justification text
        justification = await self._generate_overall_justification(
            prospect_data, risk_assessment, persona_classification, recommendations
        )
        

        if state.recommendations is None:
            state.recommendations = RecommendationState()
        # Update state
        state.recommendations.recommended_products = recommendations
        state.recommendations.justification_text = justification

        state.complete_agent_execution(self.name, status="completed")
        
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
    
    async def _generate_recommendations_batch(
        self, 
        prospect_data, 
        risk_assessment, 
        persona_classification, 
        suitable_products: pd.DataFrame
    ) -> List[ProductRecommendation]:
        """Generate AI-powered product recommendations."""
        
        if suitable_products.empty:
            return []

        products_summary='\n'.join([f'{row["product_id"]} | {row["product_name"]} ({row["product_type"]},{row["risk_level"]},Expected Return: {row.get("expected_return","N/A")})'
            for _, row in suitable_products.iterrows()
        ])

        json_example="""
            Return ONLY a valid JSON array in this exact format - no extra text, commentary, markdowns etc, it should be a PURE JSON file only,
            emphasis on PURE JSON because i will be using a json loads function on this output
            Format:
                {{
                    "product_id" :"...",
                    "suitability_score":0.85,
                    "justification":"..."
                }}
            """

        prompt_template=ChatPromptTemplate.from_messages([ 
            ("system",self.get_system_prompt()),
            ("human",f"""
            Given the following prospect and products, assign a consice suitability justification
            and a suitability score (0 to 1) for each product.

            Prospect Profile:
            - Age: {prospect_data.age}
            - Annual Income: {prospect_data.current_savings:,}
            - Current Savings: {prospect_data.current_savings} 
            - Investment Horizon Years: {prospect_data.investment_horizon_years}
            - Risk Profile: {risk_assessment.risk_level}
            - Persona: {persona_classification.persona_type if persona_classification else "N/A"}

            Products:
            {products_summary}
            
            Output Json Format:
            {json_example}""")
        ])

        response=await self.generate_response(prompt_template,{})

        import re, json,ast 
        
        self.logger.info(f"response=============>{response!r}")

        parsed = [] 

        try:
            clean=response.strip()

            clean=re.sub(r"^```(?:json)?\s*","",clean,flags=re.IGNORECASE)
            clean=re.sub(r"\s*```$","",clean,flags=re.IGNORECASE)

            match = re.search(r"\[.*\]",clean,flags=re.DOTALL)
            if match:
                candidate=match.group(0).strip()
            else:
                match_obj=re.search(r"\{.*\}",clean,flags=re.DOTALL)
                candidate=match_obj.group(0).strip() if match_obj else clean

            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                candidate_fixed=candidate.replace("'",'"')
                try: 
                    parsed=json.loads(candidate_fixed)
                except json.JSONDecodeError:
                    try:
                        parsed=ast.literal_eval(candidate)
                    except Exception as e:
                        raise ValueError(f" Failed to parse JSON/py-literal:{e}")
            if isinstance(parsed,dict):
                parsed=[parsed]
            if not isinstance(parsed,list):
                raise ValueError("Parsed content is not in a list")
            self.logger.info(f"Parsed {len(parsed)} items from LLM response")
        except Exception as e:
            self.logger.warning("LLM response parsing failed; using default scores {e}")
            parsed=[]


        recommendations = []
        
        for _, product in suitable_products.iterrows():

            entry=next(
                (x for x in parsed if str(x.get("product_id")).strip().upper()==str(product["product_id"]).strip().upper()),{})
            # Calculate suitability score
       
            suitability_score = self._calculate_suitability_score(
                product, prospect_data, risk_assessment, persona_classification
            )
            
            # Generate AI justification for this product
           # justification = await self._generate_product_justification(
            #    product, prospect_data, risk_assessment, persona_classification
            #)

            justification= entry.get("justification", "This product alignts with the client's profile.")
            
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
    
    # async def _generate_product_justification(
    #     self, 
    #     product, 
    #     prospect_data, 
    #     risk_assessment, 
    #     persona_classification
    # ) -> str:
    #     """Generate AI justification for a specific product."""
        
    #     prompt_template = ChatPromptTemplate.from_messages([
    #         ("system", self.get_system_prompt()),
    #         ("human", """
    #         Generate a concise justification for recommending this product to the prospect:
            
    #         Product Details:
    #         - Name: {product_name}
    #         - Type: {product_type}
    #         - Risk Level: {risk_level}
    #         - Expected Return: {expected_return}
    #         - Minimum Investment: ₹{min_investment:,}
            
    #         Prospect Profile:
    #         - Age: {age}
    #         - Annual Income: ₹{annual_income:,}
    #         - Current Savings: ₹{current_savings:,}
    #         - Investment Horizon: {investment_horizon_years} years
    #         - Risk Profile: {risk_profile}
    #         - Persona: {persona_type}
            
    #         Provide a 2-3 sentence justification explaining why this product is suitable.
    #         """)
    #     ])
        
    #     input_variables = {
    #         "product_name": product['product_name'],
    #         "product_type": product['product_type'],
    #         "risk_level": product['risk_level'],
    #         "expected_return": product.get('expected_return', 'N/A'),
    #         "min_investment": product['min_investment'],
    #         "age": prospect_data.age,
    #         "annual_income": prospect_data.annual_income,
    #         "current_savings": prospect_data.current_savings,
    #         "investment_horizon_years": prospect_data.investment_horizon_years,
    #         "risk_profile": risk_assessment.risk_level,
    #         "persona_type": persona_classification.persona_type if persona_classification else "N/A"
    #     }
        
    #     return await self.generate_response(prompt_template, input_variables)
    
    async def _generate_overall_justification(
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
    
    #modified for main
    def validate_input(self, state: WorkflowState) -> bool:
        """Validate input for product recommendation."""
        return (
            state.prospect and
            getattr(state.prospect, "prospect_data", None) is not None and
            state.analysis and
            getattr(state.analysis, "risk_assessment", None) is not None
        )

    def validate_output(self, state: WorkflowState) -> bool:
        """Validate product recommendation output."""
        return (
            len(getattr(state.recommendations, "recommended_products", [])) > 0 and
            getattr(state.recommendations, "justification_text", None) is not None
        )
