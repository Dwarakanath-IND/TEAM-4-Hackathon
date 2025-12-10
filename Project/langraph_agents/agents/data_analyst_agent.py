# TODO: Create DataAnalystAgent class inheriting from BaseAgent
# TODO: Implement validate_prospect_data method checking required fields
# TODO: Implement check_for_errors method identifying invalid data patterns
# TODO: Implement calculate_quality_score method based on completeness and validity
# TODO: Implement identify_missing_fields method listing absent required data
# TODO: Implement async run method:
#   - Validate prospect data presence
#   - Check for data errors
#   - Calculate quality score
#   - Identify missing fields
#   - Update state with validation_errors and data_quality_score
#   - Return updated state

from typing import Dict,Any,List
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate


from langraph_agents.base_agent import CriticalAgent
from state import WorkflowState, ProspectData, ProspectState
from config.settings import get_settings #Work on this my G.abs
import logging

class DataAnalystAgent(CriticalAgent):
    
    def __init__(self):
        super().__init__(
            name = "Data Analyst Agent",
            description="Validates, cleans and assesses the quality of prospect data"
        )
        self.settings = get_settings() #This needs to be sorted out too.
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    async def execute(self,state: WorkflowState) -> WorkflowState:

        self.logger.info("Starting data analysis and validation")

        if not state.prospect.prospect_data:

            if hasattr(state,'raw_prospect_data'):
                # prospect_data = self._create_prospect_data(state.raw_prospect_data)
                # state.prospect.prospect_data = prospect_data
                state.prospect.prospect_data = self._create_prospect_data(state.raw_prospect_data)
            else:
                raise ValueError("No prospect data available for analysis")
        
        validation_results = await self._validate_data_quality(state.prospect.prospect_data)

        state.prospect.validation_errors = validation_results['errors']
        state.prospect.missing_fields = validation_results['missing_fields']
        state.prospect.data_quality_score = validation_results['quality_score']

        if validation_results['quality_score']<0.8:
            state.prospect.prospect_data = await self._clean_and_enhance_data(
                state.prospect.prospect_data
            )
        
        self.logger.info(f"Data analysis completed. The quality score is: {validation_results['quality_score']}")
        return state

    def _create_prospect_data(self,raw_data: Dict[str,Any]) -> ProspectData:

        return ProspectData(
            prospect_id = raw_data.get('prospect_id',''),
            name = raw_data.get('name',''),
            age = int(raw_data.get('age',0)),
            annual_income = float(raw_data.get('annual_income',0)),
            current_savings = float(raw_data.get('current_savings',0)),
            target_goal_amount = float(raw_data.get('investment_horizon_years',0)),
            investment_horizon_years = int(raw_data.get('investment_horizon_years',0)),
            number_of_dependents = int(raw_data.get('number_of_dependents',0)),
            investment_experience_level = raw_data.get('investment_experience_level',''),
            investment_goal = raw_data.get('investment_goal')
        )

    async def _validate_data_quality(self,prospect_data: ProspectData) -> Dict[str,Any]:
        errors = []
        missing_fields = []
        quality_score = 1.0

        required_fields = [
            'prospect_id','name','age','annual_income',
            'current_savings','target_goal_amount',
            'investment_horizon_years','investment_experience_level'
        ]

        for field in required_fields:
            value = getattr(prospect_data, field)
            if not value or (isinstance(value,(int,float)) and value <=0):
                missing_fields.append(field)
                quality_score -=0.1
        
        if prospect_data.age < 18 or prospect_data.age >100:
            errors.append("Age must be between 18 and 100")
            quality_score -= 0.1

        if prospect_data.annual_income < 10000:
            errors.append("Annual income seems unusually low")
            quality_score -= 0.05
        
        if prospect_data.current_savings < 0:
            errors.append("Current savings cannot be negative")
            quality_score -= 0.1

        if prospect_data.target_goal_amount <= prospect_data.current_savings:
            errors.append("Target goal amount should be greater than current savings")
            quality_score -= 0.1

        if prospect_data.investment_horizon_years <= 0:
            errors.append("Investment horizon must be positive")
            quality_score -= 0.1

        if prospect_data.number_of_dependents < 0:
            errors.append("Number of dependents cannot be negative")
            quality_score -= 0.05

        valid_experience_levels = ['Beginner','Intermediate','Advanced']
        if prospect_data.investment_experience_level not in valid_experience_levels:
            errors.append(f"Invalid experience level. Must be one of: {valid_experience_levels}")
            quality_score -= 0.1

        return {
            'errors':errors,
            'missing_fields':missing_fields,
            'quality_score': max(0.0,quality_score)
        }
    
    async def _clean_and_enhance_data(self,prospect_data: ProspectData) -> ProspectData:
        prompt_template = self.get_prompt_template()

        input_variables = {
            "prospect_data": prospect_data.dict(),
            "validation_errors":self.logger.info("cleaning data with AI assistance")
        }

        response = await self.generate_response(prompt_template, input_variables)

        return prospect_data

    def get_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",self.get_system_prompt()),
            ("human","""
            Please analyze the following prospect data and suggest corrections for any issues:

            Prospect Data: {prospect_data}

            Identified Issues:
            - Missing or invalid fields
            - Business logic violations
            - Data consistency problems

            Provide suggestions for:
            1. Filling missing required fields with reasonable defaults
            2. Correcting invalid values
            3. Ensuring data consistency

            Format your response as actionable recommendations
            """)
        ])

    def validate_input(self, state: WorkflowState) -> bool:
        return (
            state.prospect.prospect_data is not None or 
            hasattr(state,'raw_prospect_data')
        )
    
    def validate_output(self,state: WorkflowState) -> bool:
        return(
            state.prospect.prospect_data is not None and 
            state.prospect.data_quality_score is not None
        )
