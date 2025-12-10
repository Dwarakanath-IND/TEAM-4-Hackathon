# TODO: Create GoalPlanningAgent class inheriting from BaseAgent
# TODO: Import predict_goal_success from ml.training
# TODO: Implement analyze_goal_feasibility assessing if goals are achievable
# TODO: Implement calculate_success_probability based on financial metrics
# TODO: Implement identify_success_factors listing favorable conditions
# TODO: Implement identify_challenges listing obstacles
# TODO: Implement analyze_timeline assessing goal timeline
# TODO: Implement async run method:
#   - Extract investment goal and timeline from prospect
#   - Call ML model via predict_goal_success
#   - Perform AI-based feasibility analysis
#   - Identify success factors and challenges
#   - Analyze timeline
#   - Return GoalPredictionResult in state

import asyncio 
import logging
from typing import Dict, Any, List
import os 
import re
from dotenv import load_dotenv
import google.generativeai as genai 

from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import HumanMessage, SystemMessage 

from langraph_agents.base_agent import BaseAgent
from ml.training.predict_goal_success import predict_goal_success, load_goal_model, rule_based_scoring_goal
from state import WorkflowState, ProspectState, AnalysisState, GoalPredictionResult
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))


class GoalPlanningAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Goal Planning Agent",
            description="Evaluates investment goals, feasibility, and challenges using ML and AI insights"
        )
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def load_models(self):
        return load_goal_model()
    
    def ml_goal_prediction(self, prospect_data):
        return predict_goal_success(prospect_data)
    
    def rule_based_goal_prediction(self, prospect_data):
        return rule_based_scoring_goal(prospect_data)
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        try:
            logger.info("Starting goal planning workflow...")
            prospect_data = getattr(state.prospect, "prospect_data", None)
            if not prospect_data:
                raise ValueError("Missing prospect data in state.")

            if not isinstance(prospect_data, dict):
                prospect_data = vars(prospect_data)

            goal = prospect_data.get("investment_goal", "Not specified")
            horizon = prospect_data.get("investment_horizon_years", 0)
            logger.info(f"Analyzing goal: {goal} | Horizon: {horizon} years")

            try:
                prob_result = await self.ml_goal_prediction(prospect_data)
                if isinstance(prob_result, tuple):
                    probability = float(prob_result[1])
                else:
                    probability = float(prob_result)
                logger.info(f"======> Predicted success probability: {probability:.2%}")
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                probability = 0.0
            
            ai_result = await self.ai_goal_analysis(prospect_data, probability)

            goal_result = GoalPredictionResult(
                goal_success=ai_result.get("goal_success", "Partially achievable"),
                probability=probability,
                success_factors=ai_result.get("success_factors", []),
                challenges=ai_result.get("challenges", []),
                timeline_analysis=ai_result.get("timeline_analysis", {"summary": "Timeline analysis unavailable"}),
            )

            state.analysis.goal_prediction = goal_result

            #state.goal_prediction_result = goal_result
            state.analysis.goal_prediction = goal_result
            logger.info("Goal Planning completed successfully")
            return state 
        
        except Exception as e:
            logger.error(f"Error in Goal Planning Agent: {e}")
            return state

    async def ai_goal_analysis(self, prospect_data, probability):
        """Enhanced AI goal analysis with better structured prompting"""
        
        prompt_text = f"""Analyze this investment goal and provide a structured feasibility assessment.

**Prospect Profile:**
- Age: {prospect_data.get('age', 'N/A')}
- Annual Income: ${prospect_data.get('annual_income', 0):,.2f}
- Current Savings: ${prospect_data.get('current_savings', 0):,.2f}
- Target Goal Amount: ${prospect_data.get('target_goal_amount', 0):,.2f}
- Investment Horizon: {prospect_data.get('investment_horizon_years', 'N/A')} years
- Number of Dependents: {prospect_data.get('number_of_dependents', 'N/A')}
- Investment Experience: {prospect_data.get('investment_experience_level', 'N/A')}
- Investment Goal: {prospect_data.get('investment_goal', 'Not specified')}

**ML Model Prediction:**
- Success Probability: {probability:.2%}

Provide your analysis in this exact format. Each bullet point must be a COMPLETE sentence on ONE line:

FEASIBILITY_STATUS:
[Write ONE clear sentence stating if the goal is: Highly Achievable, Moderately Achievable, Challenging, or Unrealistic]

SUCCESS_FACTORS:
- Complete success factor description here
- Another complete success factor description here
- Third complete success factor description here

CHALLENGES:
- Complete challenge description here
- Another complete challenge description here
- Third complete challenge description here

TIMELINE_ASSESSMENT:
- Complete timeline assessment in one sentence

Important: 
- Be specific to THIS prospect's situation
- Each bullet point must be complete on a single line
- No placeholders or generic templates
- Focus on the most important 3-4 items per category"""

        try:
            # response = await self.model.generate_content_async(prompt_text)
            response = await asyncio.to_thread(self.model.generate_content, prompt_text)
            text = self._extract_text_from_response(response)
            
            if not text:
                logger.warning("Empty response from Gemini API")
                return self._get_fallback_analysis(probability)
            
            logger.debug(f"AI Response:\n{text}")
            
            # Parse the structured response
            parsed = self._parse_structured_response(text, probability)
            
            return parsed
            
        except Exception as e:
            logger.error(f"AI goal analysis failed: {e}")
            return self._get_fallback_analysis(probability)
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text content from Gemini API response"""
        try:
            if hasattr(response, 'text'):
                return response.text
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts_text = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            parts_text.append(part.text)
                    if parts_text:
                        return '\n'.join(parts_text)
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error extracting text from response: {e}")
            return ""
    
    def _parse_structured_response(self, text: str, probability: float) -> Dict[str, Any]:
        """Parse the AI response into structured goal analysis"""
        goal_success = "Partially achievable"
        success_factors = []
        challenges = []
        timeline_summary = "Timeline assessment unavailable"
        
        current_section = None
        
        for line in text.split('\n'):
            line = line.strip()
            line_upper = line.upper()
            
            # Detect section headers
            if 'FEASIBILITY_STATUS:' in line_upper or 'FEASIBILITY STATUS:' in line_upper:
                current_section = 'feasibility'
                continue
            elif 'SUCCESS_FACTORS:' in line_upper or 'SUCCESS FACTORS:' in line_upper:
                current_section = 'success'
                continue
            elif 'CHALLENGES:' in line_upper:
                current_section = 'challenges'
                continue
            elif 'TIMELINE_ASSESSMENT:' in line_upper or 'TIMELINE ASSESSMENT:' in line_upper:
                current_section = 'timeline'
                continue
            
            # Skip empty lines
            if not line:
                continue
            
            # Process content based on section
            if current_section == 'feasibility':
                # Remove any leading markers
                clean_line = re.sub(r'^[-•*]\s*', '', line).strip()
                if clean_line and len(clean_line) > 10:
                    goal_success = clean_line
                    current_section = None  # Only take first line
                    
            elif current_section == 'success':
                if line.startswith(('-', '•', '*')):
                    clean_line = re.sub(r'^[-•*]\s*', '', line).strip()
                    if clean_line and len(clean_line) > 10:
                        success_factors.append(clean_line)
                        
            elif current_section == 'challenges':
                if line.startswith(('-', '•', '*')):
                    clean_line = re.sub(r'^[-•*]\s*', '', line).strip()
                    if clean_line and len(clean_line) > 10:
                        challenges.append(clean_line)
                        
            elif current_section == 'timeline':
                # Accept both bulleted and non-bulleted lines for timeline
                clean_line = re.sub(r'^[-•*]\s*', '', line).strip()
                if clean_line and len(clean_line) > 10 and not clean_line.isupper():
                    timeline_summary = clean_line
                    current_section = None  # Only take first line
        
        # Fallbacks if parsing didn't capture content
        if not success_factors:
            success_factors = [
                "High predicted success probability indicates favorable conditions",
                "Consistent income provides stable foundation for goal achievement"
            ]
        
        if not challenges:
            challenges = [
                "Requires disciplined savings and investment strategy",
                "Market volatility may impact short-term performance"
            ]
        
        # Limit to top items
        return {
            "goal_success": goal_success,
            "success_factors": success_factors[:5],
            "challenges": challenges[:5],
            "timeline_analysis": {"summary": timeline_summary}
        }
    
    def _get_fallback_analysis(self, probability: float) -> Dict[str, Any]:
        """Provide fallback analysis when AI fails"""
        if probability >= 0.7:
            status = "Moderately achievable with disciplined planning"
        elif probability >= 0.5:
            status = "Challenging but possible with strategic adjustments"
        else:
            status = "Requires significant changes to financial plan"
        
        return {
            "goal_success": status,
            "success_factors": [
                "ML model indicates baseline feasibility",
                "Financial profile provides foundation for planning"
            ],
            "challenges": [
                "Detailed analysis unavailable - manual review recommended",
                "Market conditions and timing factors require consideration"
            ],
            "timeline_analysis": {"summary": "Timeline assessment requires detailed review"}
        }
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        system_message = SystemMessage(
            content=(
                "You are a financial planning assistant. "
                "Given a user's financial data and goal success probability, "
                "evaluate if the goal is feasible, identify key success factors, challenges, "
                "and comment on timeline feasibility. Be specific and actionable."
            )
        )

        human_message = HumanMessage(
            content=(
                "Prospect profile:\n{prospect}\n\n"
                "Predicted Success Probability: {probability}\n\n"
                "Provide:\n"
                "- Goal feasibility summary\n"
                "- Success factors (bullet points)\n"
                "- Challenges (bullet points)\n"
                "- Timeline feasibility comment\n"
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])


if __name__ == "__main__":
    prospect = {
        "age": 35,
        "annual_income": 90000,
        "current_savings": 17000,
        "target_goal_amount": 100000,
        "investment_horizon_years": 3,
        "number_of_dependents": 1,
        "investment_experience_level": "Beginner",
        "investment_goal": "Save for child's education"
    }

    state = WorkflowState(
        workflow_id="wf_test_001",
        session_id="sess_01",
        prospect=ProspectState(prospect_data=prospect),
        analysis=AnalysisState()
    )

    agent = GoalPlanningAgent()
    updated_state = asyncio.run(agent.execute(state))

    print("\n" + "=" * 60)
    print("GOAL PLANNING ANALYSIS RESULT")
    print("=" * 60)

    if updated_state.analysis and updated_state.analysis.goal_prediction:
        res = updated_state.analysis.goal_prediction

        print("\nGoal Feasibility Report")
        print("-" * 60)
        print(f"Feasibility Status: {res.goal_success}")
        print(f"Predicted Success Probability: {res.probability * 100:.2f}%")

        print("\nKey Success Factors:")
        for i, f in enumerate(res.success_factors, 1):
            print(f"  {i}. {f}")

        print("\nChallenges:")
        for i, c in enumerate(res.challenges, 1):
            print(f"  {i}. {c}")

        print("\nTimeline Analysis:")
        if isinstance(res.timeline_analysis, dict):
            print(f"  {res.timeline_analysis.get('summary', 'Not available')}")
        else:
            print(f"  {res.timeline_analysis}")

        print("-" * 60)
    else:
        print("No goal prediction result generated.")
