# TODO: Create RiskAssessmentAgent class inheriting from BaseAgent
# TODO: Import predict_risk_profile function from ml.training
# TODO: Implement get_risk_factors method analyzing prospect characteristics
# TODO: Implement generate_recommendations method based on risk level
# TODO: Implement analyze_risk_indicators extracting age, income, savings factors
# TODO: Implement async run method:
#   - Extract prospect data
#   - Call ML model via predict_risk_profile for initial assessment
#   - Generate detailed risk factors using AI analysis
#   - Create risk recommendations
#   - Calculate confidence score
#   - Return RiskAssessmentResult in state

import asyncio 
from typing import Dict, Any, List
from ml.training.predict_risk_profile import predict_risk_profile, load_risk_model, rule_based_scoring
from state import RiskAssessmentResult, WorkflowState, ProspectState, AnalysisState
from langraph_agents.base_agent import BaseAgent
from dataclasses import dataclass 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

import google.generativeai as genai
import logging
import argparse 
import re

import os
from dotenv import load_dotenv 
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))


class RiskAssessmentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Risk Assessment Agent",
            description="Assess financial risk profile using machine learning models and AI analysis. Combines objective ML prediction with contextual AI risk factor analysis")
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    async def execute(self, state: WorkflowState) -> WorkflowState:
        try:
            logger.info("Starting risk assessment workflow...")
            prospect_data = getattr(state.prospect, "prospect_data", None)
            if not prospect_data:
                raise ValueError("Missing prospect data in state.")

            if not isinstance(prospect_data, dict):
                prospect_data = vars(prospect_data)

            logger.info("Performing ML-based risk prediction...")
            try:
                risk_level, confidence = await predict_risk_profile(prospect_data)
                ml_result = {"risk_level": risk_level, "confidence": confidence}
                logger.info(f"ML Result: {ml_result}")
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}. Using fallback.")
                ml_result = {"risk_level": "Medium", "confidence": 0.5}

            logger.info("Generating AI-based risk factor analysis...")
            ai_result = await self.ai_risk_analysis(prospect_data, ml_result)

            risk_result = RiskAssessmentResult(
                risk_level=ml_result["risk_level"],
                confidence_score=ml_result.get("confidence", 0.0),
                risk_factors=ai_result.get("risk_factors", []),
                recommendations=ai_result.get("recommendations", [])
            )

            state.risk_assessment_result = risk_result
            state.analysis.risk_assessment = risk_result  
            logger.info(f"Risk assessment complete: Risk Level={risk_result.risk_level}, Confidence={risk_result.confidence_score}")
            return state 
        except Exception as e:
            logger.error(f"Error in RiskAssessmentAgent: {e}")
            return state

    async def ai_risk_analysis(self, prospect_data, ml_result):
        """Enhanced AI risk analysis with better response parsing"""
        
        prompt_text = f"""Analyze this financial prospect profile and provide a structured risk assessment.

    **Prospect Profile:**
    - Age: {prospect_data.get('age', 'N/A')}
    - Annual Income: ${prospect_data.get('annual_income', 0):,.2f}
    - Current Savings: ${prospect_data.get('current_savings', 0):,.2f}
    - Investment Horizon: {prospect_data.get('investment_horizon_years', 'N/A')} years
    - Number of Dependents: {prospect_data.get('number_of_dependents', 'N/A')}
    - Investment Experience: {prospect_data.get('investment_experience_level', 'N/A')}

    **ML Model Assessment:**
    - Risk Level: {ml_result.get('risk_level', 'N/A')}
    - Confidence: {ml_result.get('confidence', 0):.2%}

    Provide your analysis with exactly this format. Each bullet point should be a COMPLETE sentence on ONE line:

    RISK_FACTORS:
    - Complete risk factor description here
    - Another complete risk factor description here
    - Third complete risk factor description here

    RECOMMENDATIONS:
    - Complete recommendation description here
    - Another complete recommendation description here
    - Third complete recommendation description here

    Important: Each bullet point must be complete on a single line. Do not split descriptions across multiple lines."""

        try:
            #response = await self.model.generate_content_async(prompt_text)
            response= await asyncio.to_thread(self.model.generate_content,prompt_text)
            text = self._extract_text_from_response(response)
            
            if not text:
                logger.warning("Empty response from Gemini API")
                return self._get_fallback_analysis()
            
            logger.debug(f"AI Response:\n{text}")
            
            # Simplified parsing
            risk_factors = []
            recommendations = []
            current_section = None
            
            for line in text.split('\n'):
                line = line.strip()
                
                if 'RISK_FACTORS:' in line or 'RISK FACTORS:' in line:
                    current_section = 'risk'
                    continue
                elif 'RECOMMENDATIONS:' in line:
                    current_section = 'recommendations'
                    continue
                
                # Extract bullet points
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    clean_line = re.sub(r'^[-•*]\s*', '', line).strip()
                    if clean_line and len(clean_line) > 10:
                        if current_section == 'risk':
                            risk_factors.append(clean_line)
                        elif current_section == 'recommendations':
                            recommendations.append(clean_line)
            
            # Ensure we have content
            if not risk_factors:
                risk_factors = ["Unable to parse specific risk factors from AI response"]
            if not recommendations:
                recommendations = ["Consult with a financial advisor for personalized recommendations"]
            
            return {
                "risk_factors": risk_factors[:5],
                "recommendations": recommendations[:5]
            }
            
        except Exception as e:
            logger.error(f"AI risk analysis failed: {e}")
            return self._get_fallback_analysis()
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text content from Gemini API response"""
        try:
            # Method 1: Check for text attribute directly
            if hasattr(response, 'text'):
                return response.text
            
            # Method 2: Check candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Check if candidate has content with parts
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts_text = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            parts_text.append(part.text)
                    if parts_text:
                        return '\n'.join(parts_text)
            
            # Fallback: convert to string
            return str(response)
            
        except Exception as e:
            logger.error(f"Error extracting text from response: {e}")
            return ""
    
    def _parse_structured_response(self, text: str) -> tuple[List[str], List[str]]:
        """Parse the AI response into risk factors and recommendations"""
        risk_factors = []
        recommendations = []
        
        # Split text into lines
        lines = text.split('\n')
        current_section = None
        current_item = []
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Detect section headers
            if 'risk factor' in line_lower and ('**' in line or ':' in line):
                # Save any accumulated item
                if current_item and current_section == 'recommendations':
                    recommendations.append(' '.join(current_item))
                current_section = 'risk'
                current_item = []
                continue
            elif 'recommendation' in line_lower and ('**' in line or ':' in line):
                # Save any accumulated item
                if current_item and current_section == 'risk':
                    risk_factors.append(' '.join(current_item))
                current_section = 'recommendations'
                current_item = []
                continue
            
            # Skip empty lines
            if not line_stripped:
                # Empty line might indicate end of an item
                if current_item:
                    if current_section == 'risk':
                        risk_factors.append(' '.join(current_item))
                    elif current_section == 'recommendations':
                        recommendations.append(' '.join(current_item))
                    current_item = []
                continue
            
            # Process content lines
            if current_section:
                # Remove bullet point markers
                cleaned = re.sub(r'^[-•*]\s*', '', line_stripped)
                
                # Check if this is a new bullet point (starts with -, *, • or is after empty line)
                is_new_bullet = line_stripped.startswith(('-', '*', '•'))
                
                if is_new_bullet and current_item:
                    # Save previous item
                    if current_section == 'risk':
                        risk_factors.append(' '.join(current_item))
                    elif current_section == 'recommendations':
                        recommendations.append(' '.join(current_item))
                    current_item = []
                
                if cleaned:
                    current_item.append(cleaned)
        
        # Don't forget the last accumulated item
        if current_item:
            if current_section == 'risk':
                risk_factors.append(' '.join(current_item))
            elif current_section == 'recommendations':
                recommendations.append(' '.join(current_item))
        
        # Clean up items - remove trailing colons and merge title+description
        def clean_items(items):
            cleaned = []
            for item in items:
                # If item ends with colon and is short, it might be a title that got separated
                # Skip it as it will be merged with next item
                if item.endswith(':') and len(item) < 50:
                    continue
                cleaned.append(item)
            return cleaned
        
        risk_factors = clean_items(risk_factors)
        recommendations = clean_items(recommendations)
        
        # Fallback if nothing was parsed
        if not risk_factors:
            risk_factors = ["Unable to parse specific risk factors from AI response"]
        if not recommendations:
            recommendations = ["Consult with a financial advisor for personalized recommendations"]
        
        return risk_factors[:5], recommendations[:5]  # Limit to top 5 each
    
    def _get_fallback_analysis(self) -> Dict[str, List[str]]:
        """Provide fallback analysis when AI fails"""
        return {
            "risk_factors": [
                "AI analysis unavailable - manual review required",
                "Unable to generate detailed risk assessment"
            ],
            "recommendations": [
                "Consult with a financial advisor for personalized advice",
                "Review prospect profile manually for comprehensive assessment"
            ]
        }
    
    def get_prompt_template(self) -> ChatPromptTemplate:
        system_message = SystemMessage(
            content=(
                "You are a financial risk analysis assistant. "
                "Given a user's financial data and machine learning risk classification, "
                "identify key risk factors and propose mitigation recommendations. "
                "Be specific and actionable, considering age, income, savings, experience, "
                "number of dependents and investment horizon years."
            )
        )

        human_message = HumanMessage(
            content=(
                "Prospect profile:\n{prospect}\n\n"
                "ML model output:\n{ml_result}\n\n"
                "Analyze this data and provide: \n"
                "- Specific risk factors (bullet points)\n"
                "- Mitigation recommendations (bullet points)\n"
            )
        )
        return ChatPromptTemplate.from_messages([system_message, human_message])


if __name__ == "__main__":
    prospect = {
        "age": 35,
        "annual_income": 222343,
        "current_savings": 1700,
        "investment_horizon_years": 5,
        "number_of_dependents": 3,
        "investment_experience_level": "Beginner"
    }
    
    prospect_state = ProspectState(prospect_data=prospect)

    state = WorkflowState(
        workflow_id="wf_test_001",
        session_id="sess_01",
        prospect=prospect_state,
        analysis=AnalysisState()
    )
    
    agent = RiskAssessmentAgent()
    updated_state = asyncio.run(agent.execute(state))

    logger.info("=========== RESULTS =================")
    if hasattr(updated_state, 'risk_assessment_result') and updated_state.risk_assessment_result:
        r = updated_state.risk_assessment_result
        
        logger.info(f"\nRisk Level: {r.risk_level}")
        logger.info(f"Confidence: {r.confidence_score:.2%}\n")
        
        logger.info("Risk Factors:")
        for i, factor in enumerate(r.risk_factors, 1):
            logger.info(f"  {i}. {factor}")
        
        logger.info("\nRecommendations:")
        for i, rec in enumerate(r.recommendations, 1):
            logger.info(f"  {i}. {rec}")
    else:
        logger.info("No risk assessment result generated")
