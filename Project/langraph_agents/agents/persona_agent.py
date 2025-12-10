# # TODO: Create PersonaAgent class inheriting from BaseAgent
# # TODO: Implement classify_persona method determining investor type
# # TODO: Implement generate_insights method creating behavioral insights
# # TODO: Implement extract_behavioral_signals from prospect data
# # TODO: Implement persona classification logic (Aggressive Growth, Steady Saver, Cautious Planner)
# # TODO: Implement async run method:
# #   - Extract behavioral signals from prospect data
# #   - Classify persona type using AI analysis
# #   - Generate behavioral characteristics
# #   - Create behavioral insights
# #   - Calculate confidence score
# #   - Return PersonaResult in state


# from typing import Dict, Any, List
# from langchain_core.prompts import ChatPromptTemplate
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# from langraph_agents.base_agent import BaseAgent
# from state import WorkflowState, PersonaResult
# from config.settings import get_settings

# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))


# class PersonaAgent(BaseAgent):

    
#     def __init__(self):
#         super().__init__(
#             name="Persona Agent",
#             description="Classifies client personas and provides behavioral insights for personalized advisory"
#         )
#         self.settings = get_settings()
        

#         self.persona_types = {
#             "Aggressive Growth": {
#                 "description": "High risk tolerance, seeks maximum returns, comfortable with volatility",
#                 "characteristics": ["High risk tolerance", "Growth-focused", "Long-term oriented", "Market-savvy"],
#                 "typical_profile": "Young professionals, high income, long investment horizon"
#             },
#             "Steady Saver": {
#                 "description": "Balanced approach, consistent investments, moderate risk tolerance",
#                 "characteristics": ["Consistent investor", "Balanced risk approach", "Goal-oriented", "Disciplined"],
#                 "typical_profile": "Middle-aged professionals, stable income, medium-term goals"
#             },
#             "Cautious Planner": {
#                 "description": "Conservative approach, capital preservation focus, low risk tolerance",
#                 "characteristics": ["Risk-averse", "Capital preservation", "Security-focused", "Conservative"],
#                 "typical_profile": "Pre-retirees, risk-averse individuals, short-term goals"
#             }
#         }
    
#     async def execute(self, state: WorkflowState) -> WorkflowState:

#         self.logger.info("Starting persona classification")
        
#         prospect_data = state.prospect.prospect_data
#         risk_assessment = state.analysis.risk_assessment
        
#         if not prospect_data:
#             raise ValueError("No prospect data available for persona classification")
        

#         persona_result = await self._classify_persona(prospect_data, risk_assessment)
        
#         self.logger.info(" persona_result done")

#         behavioral_insights = await self._generate_behavioral_insights(prospect_data, persona_result)
        
#         self.logger.info(" behavioral insights done")

#         final_result = PersonaResult(
#             persona_type=persona_result['persona_type'],
#             confidence_score=persona_result['confidence_score'],
#             characteristics=self.persona_types[persona_result['persona_type']]['characteristics'],
#             behavioral_insights=behavioral_insights
#         )
        
#         state.analysis.persona_classification = final_result
        
#         self.logger.info(f"Persona classification completed: {final_result.persona_type}")
#         return state
    
#     async def _classify_persona(self, prospect_data, risk_assessment) -> Dict[str, Any]:

#         prompt_template = self.get_classification_prompt()
        

#         risk_info = ""
#         if risk_assessment:
#             risk_info = f"Risk Level: {risk_assessment.risk_level}, Confidence: {risk_assessment.confidence_score}"
        
#         input_variables = {
#             "prospect_data": prospect_data.model_dump(),
#             "risk_assessment": risk_info,
#             "persona_types": self._format_persona_types()
#         }
        
#         response = await self.generate_response(prompt_template, input_variables)
        

#         persona_type = self._extract_persona_type(response)
#         confidence_score = self._calculate_confidence_score(prospect_data, persona_type)
        
#         return {
#             "persona_type": persona_type,
#             "confidence_score": confidence_score,
#             "ai_reasoning": response
#         }
    
#     def _extract_persona_type(self, ai_response: str) -> str:

#         response_lower = ai_response.lower()
        
#         for persona_type in self.persona_types.keys():
#             if persona_type.lower() in response_lower:
#                 return persona_type
        
#         if any(word in response_lower for word in ['aggressive', 'growth', 'high risk']):
#             return "Aggressive Growth"
#         elif any(word in response_lower for word in ['cautious', 'conservative', 'low risk']):
#             return "Cautious Planner"
#         else:
#             return "Steady Saver"  # Default
    
#     def _calculate_confidence_score(self, prospect_data, persona_type: str) -> float:
#         score = 0.5  # Base score
        

#         if persona_type == "Aggressive Growth" and prospect_data.age < 35:
#             score += 0.2
#         elif persona_type == "Cautious Planner" and prospect_data.age > 50:
#             score += 0.2
#         elif persona_type == "Steady Saver" and 30 <= prospect_data.age <= 55:
#             score += 0.2
        

#         if persona_type == "Aggressive Growth" and prospect_data.investment_horizon_years > 10:
#             score += 0.15
#         elif persona_type == "Cautious Planner" and prospect_data.investment_horizon_years < 5:
#             score += 0.15
        

#         experience_mapping = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
#         experience_score = experience_mapping.get(prospect_data.investment_experience_level, 0)
        
#         if persona_type == "Aggressive Growth" and experience_score >= 1:
#             score += 0.1
#         elif persona_type == "Cautious Planner" and experience_score == 0:
#             score += 0.1
        

#         if prospect_data.annual_income > 1000000 and persona_type == "Aggressive Growth":
#             score += 0.05
        
#         return min(1.0, score)

#     async def _generate_behavioral_insights(self, prospect_data, persona_result: Dict[str, Any]) -> Dict[str, List[str]]:
#         prompt_template = self.get_insights_prompt()

#         input_variables = {
#             "prospect_data": prospect_data.model_dump(),
#             "persona_type": persona_result['persona_type'],
#             "persona_description": self.persona_types[persona_result['persona_type']]['description']
#         }

#         response = await self.generate_response(prompt_template, input_variables)

#         structured_insights: Dict[str, List[str]] = {}
#         current_section = None

#         for line in response.split('\n'):
#             line = line.strip()
#             if not line:
#                 continue

#             # Section headings
#             if line.endswith(":") and len(line.split()) <= 6:
#                 current_section = line.rstrip(":").strip()
#                 structured_insights[current_section] = []
#             elif current_section:
#                 # Clean bullets
#                 line = line.lstrip("-*• ").strip()
#                 if line:
#                     structured_insights[current_section].append(line)
#             else:
#                 if 'General Insights' not in structured_insights:
#                     structured_insights['General Insights'] = []
#                 structured_insights['General Insights'].append(line)

#         if not structured_insights:
#             structured_insights = {"General Insights": ["Standard behavioral patterns apply for this persona type"]}

#         return structured_insights
    
#     def _format_persona_types(self) -> str:

#         formatted = ""
#         for persona_type, info in self.persona_types.items():
#             formatted += f"\n{persona_type}: {info['description']}\n"
#             formatted += f"Typical Profile: {info['typical_profile']}\n"
#         return formatted
    
#     def get_classification_prompt(self) -> ChatPromptTemplate:

#         return ChatPromptTemplate.from_messages([
#             ("system", self.get_system_prompt()),
#             ("human", """
#             Classify the following prospect into one of the defined persona types based on their profile and risk assessment:
            
#             Prospect Data:
#             {prospect_data}
            
#             Risk Assessment:
#             {risk_assessment}
            
#             Available Persona Types:
#             {persona_types}
            
#             Instructions:
#             1. Analyze the prospect's age, income, investment horizon, experience level, and risk profile
#             2. Consider their financial goals and current situation
#             3. Match them to the most appropriate persona type
#             4. Provide clear reasoning for your classification
            
#             Respond with the persona type name and your reasoning.
#             """)
#         ])
    
#     def get_insights_prompt(self) -> ChatPromptTemplate:

#         return ChatPromptTemplate.from_messages([
#             ("system", self.get_system_prompt()),
#             ("human", """
#             Generate specific behavioral insights for this prospect based on their classified persona:
            
#             Prospect Data:
#             {prospect_data}
            
#             Classified Persona: {persona_type}
#             Persona Description: {persona_description}
            
#             Provide behavioral insights that will help the relationship manager:
#             - Communication preferences
#             - Decision-making patterns
#             - Likely concerns or objections
#             - Motivation factors
#             - Preferred investment approaches
            
#             Format as bullet points with actionable insights.
#             """)
#         ])
    
#     def get_prompt_template(self) -> ChatPromptTemplate:

#         return self.get_classification_prompt()
    
#     def validate_input(self, state: WorkflowState) -> bool:

#         return state.prospect.prospect_data is not None
    
#     def validate_output(self, state: WorkflowState) -> bool:

#         return (
#             state.analysis.persona_classification is not None and
#             state.analysis.persona_classification.persona_type in self.persona_types.keys()
#         )


from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

from langraph_agents.base_agent import BaseAgent
from state import WorkflowState, PersonaResult
from config.settings import get_settings

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))


class PersonaAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            name="Persona Agent",
            description="Classifies client personas and provides behavioral insights for personalized advisory"
        )
        self.settings = get_settings()

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
        self.logger.info("Starting persona classification")

        prospect_data = state.prospect.prospect_data
        risk_assessment = state.analysis.risk_assessment

        if not prospect_data:
            raise ValueError("No prospect data available for persona classification")

        persona_result = await self._classify_persona(prospect_data, risk_assessment)
        self.logger.info(" persona_result done")

        behavioral_insights = await self._generate_behavioral_insights(prospect_data, persona_result)
        self.logger.info(" behavioral insights done")

        final_result = PersonaResult(
            persona_type=persona_result['persona_type'],
            confidence_score=persona_result['confidence_score'],
            characteristics=self.persona_types[persona_result['persona_type']]['characteristics'],
            behavioral_insights=behavioral_insights
        )

        state.analysis.persona_classification = final_result
        self.logger.info(f"Persona classification completed: {final_result.persona_type}")
        return state

    async def _classify_persona(self, prospect_data, risk_assessment) -> Dict[str, Any]:
        prompt_template = self.get_classification_prompt()

        risk_info = ""
        if risk_assessment:
            risk_info = f"Risk Level: {risk_assessment.risk_level}, Confidence: {risk_assessment.confidence_score}"

        input_variables = {
            "prospect_data": prospect_data.model_dump(),
            "risk_assessment": risk_info,
            "persona_types": self._format_persona_types()
        }

        response = await self.generate_response(prompt_template, input_variables)

        persona_type = self._extract_persona_type(response)
        confidence_score = self._calculate_confidence_score(prospect_data, persona_type)

        return {
            "persona_type": persona_type,
            "confidence_score": confidence_score,
            "ai_reasoning": response
        }

    def _extract_persona_type(self, ai_response: str) -> str:
        response_lower = ai_response.lower()
        for persona_type in self.persona_types.keys():
            if persona_type.lower() in response_lower:
                return persona_type
        if any(word in response_lower for word in ['aggressive', 'growth', 'high risk']):
            return "Aggressive Growth"
        elif any(word in response_lower for word in ['cautious', 'conservative', 'low risk']):
            return "Cautious Planner"
        else:
            return "Steady Saver"  # Default

    def _calculate_confidence_score(self, prospect_data, persona_type: str) -> float:
        score = 0.5  # Base score

        if persona_type == "Aggressive Growth" and prospect_data.age < 35:
            score += 0.2
        elif persona_type == "Cautious Planner" and prospect_data.age > 50:
            score += 0.2
        elif persona_type == "Steady Saver" and 30 <= prospect_data.age <= 55:
            score += 0.2

        if persona_type == "Aggressive Growth" and prospect_data.investment_horizon_years > 10:
            score += 0.15
        elif persona_type == "Cautious Planner" and prospect_data.investment_horizon_years < 5:
            score += 0.15

        experience_mapping = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        experience_score = experience_mapping.get(prospect_data.investment_experience_level, 0)

        if persona_type == "Aggressive Growth" and experience_score >= 1:
            score += 0.1
        elif persona_type == "Cautious Planner" and experience_score == 0:
            score += 0.1

        if prospect_data.annual_income > 1000000 and persona_type == "Aggressive Growth":
            score += 0.05

        return min(1.0, score)

    async def _generate_behavioral_insights(self, prospect_data, persona_result: Dict[str, Any]) -> Dict[str, List[str]]:
        prompt_template = self.get_insights_prompt()

        input_variables = {
            "prospect_data": prospect_data.model_dump(),
            "persona_type": persona_result['persona_type'],
            "persona_description": self.persona_types[persona_result['persona_type']]['description']
        }

        response = await self.generate_response(prompt_template, input_variables)

        # --- Clean AI output ---
        # Remove ```json ...``` if present
        if "```json" in response:
            response = response.split("```json")[-1].split("```")[0].strip()

        try:
            structured_insights = json.loads(response)
        except json.JSONDecodeError:
            structured_insights = {
                "General Insights": [
                    "Standard behavioral patterns apply for this persona type",
                    "Raw AI output: " + response
                ]
            }

        # Ensure all values are lists
        for key, value in structured_insights.items():
            if not isinstance(value, list):
                structured_insights[key] = [str(value)]

        return structured_insights


    def _format_persona_types(self) -> str:
        formatted = ""
        for persona_type, info in self.persona_types.items():
            formatted += f"\n{persona_type}: {info['description']}\n"
            formatted += f"Typical Profile: {info['typical_profile']}\n"
        return formatted

    def get_classification_prompt(self) -> ChatPromptTemplate:
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
        return ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", """
            Generate specific behavioral insights for this prospect based on their classified persona.

            Prospect Data:
            {prospect_data}

            Classified Persona: {persona_type}
            Persona Description: {persona_description}

            Format your response as a JSON object with the following structure:

            {{
                "Communication Preferences": ["..."],
                "Decision-Making Patterns": ["..."],
                "Likely Concerns or Objections": ["..."],
                "Motivation Factors": ["..."],
                "Preferred Investment Approaches": ["..."]
            }}

            Include actionable bullet points in each list. Avoid text outside the JSON. The output should be pure JSON only.
            """)
        ])

    def get_prompt_template(self) -> ChatPromptTemplate:
        return self.get_classification_prompt()

    def validate_input(self, state: WorkflowState) -> bool:
        return state.prospect.prospect_data is not None

    def validate_output(self, state: WorkflowState) -> bool:
        return (
            state.analysis.persona_classification is not None and
            state.analysis.persona_classification.persona_type in self.persona_types.keys()
        )


import asyncio
from state import WorkflowState, ProspectState, ProspectData, AnalysisState, RiskAssessmentResult
#from persona_agent import PersonaAgent  # assuming your class is in persona_agent.py

if __name__ == "__main__":
    # --- Mock Prospect Data ---
    mock_prospect = ProspectData(
        prospect_id="P001",
        name="John Doe",
        age=32,
        annual_income=1200000,
        current_savings=300000,
        target_goal_amount=5000000,
        investment_horizon_years=15,
        number_of_dependents=1,
        investment_experience_level="Intermediate",
        investment_goal="Retirement planning"
    )

    # --- Mock Risk Assessment ---
    mock_risk = RiskAssessmentResult(
        risk_level="Medium",
        confidence_score=0.8,
        risk_factors=["Volatile market", "High debt ratio"],
        recommendations=["Diversify portfolio", "Consider moderate-risk instruments"]
    )

    # --- Workflow State ---
    state = WorkflowState(
        workflow_id="wf_test_001",
        session_id="sess_test_001",
        prospect=ProspectState(prospect_data=mock_prospect),
        analysis=AnalysisState(risk_assessment=mock_risk)
    )

    # --- Initialize Persona Agent ---
    agent = PersonaAgent()

    # --- Run Persona Agent ---
    async def run_agent():
        updated_state = await agent.execute(state)
        persona_result = updated_state.analysis.persona_classification
        print("Persona Type:", persona_result.persona_type)
        print("Confidence Score:", persona_result.confidence_score)
        print("Characteristics:", persona_result.characteristics)
        print("Behavioral Insights:")
        for section, insights in persona_result.behavioral_insights.items():
            print(f"  {section}:")
            for item in insights:
                print(f"    - {item}")

    asyncio.run(run_agent())
