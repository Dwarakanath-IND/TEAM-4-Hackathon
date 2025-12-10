# TODO: Import Dict, List, Optional, Any, Union from typing
# TODO: Import BaseModel, Field from pydantic
# TODO: Import datetime
# TODO: Import pandas as pd
# TODO: Create ProspectData model with fields: prospect_id, name, age, annual_income, current_savings, target_goal_amount, investment_horizon_years, number_of_dependents, investment_experience_level, investment_goal
# TODO: Create RiskAssessmentResult model with: risk_level, confidence_score, risk_factors list, recommendations list
# TODO: Create GoalPredictionResult model with: goal_success, probability, success_factors, challenges, timeline_analysis dict
# TODO: Create PersonaResult model with: persona_type, confidence_score, characteristics list, behavioral_insights list
# TODO: Create ProductRecommendation model with: product_id, product_name, product_type, suitability_score, justification, risk_alignment, expected_returns, fees
# TODO: Create MeetingGuide model with: agenda_items, key_talking_points, questions_to_ask, objection_handling dict, next_steps, estimated_duration
# TODO: Create ComplianceCheck model with: is_compliant bool, compliance_score, violations list, warnings list, required_disclosures list
# TODO: Create AgentExecution model with: agent_name, start_time, end_time, status, error_message, execution_time
# TODO: Create ProspectState model aggregating prospect_data, validation_errors, data_quality_score, missing_fields
# TODO: Create AnalysisState model aggregating risk_assessment, goal_prediction, persona_classification results
# TODO: Create RecommendationState model aggregating recommended_products, portfolio_allocation, compliance_check
# TODO: Create MeetingState model with meeting_guide, presentation_slides, client_materials
# TODO: Create ChatState model with conversation_history, current_query, context, response
# TODO: Create WorkflowState model combining all sub-states plus metadata (workflow_id, session_id, timestamps, execution tracking, configuration)
# TODO: Add add_agent_execution method to track agent execution records
# TODO: Add complete_agent_execution method to mark agents as completed/failed with timing
# TODO: Add get_execution_summary method returning execution statistics

from typing import Dict,List,Optional,Any,Union
from pydantic import BaseModel, Field
import datetime
import pandas as pd 
import time

class ProspectData(BaseModel):
    prospect_id: Optional[str] = Field(None)
    name: Optional[str] = Field(None)
    age: int = Field(...)
    annual_income: float = Field(...)
    current_savings: float= Field(...)
    target_goal_amount: Optional[float] = Field(None)
    investment_horizon_years:int = Field(...)
    number_of_dependents: int = Field (...)
    investment_experience_level: str= Field(...)
    investment_goal : Optional[str]= Field(None)

class RiskAssessmentResult(BaseModel):
    risk_level:str
    confidence_score:float 
    risk_factors: List[str]
    recommendations:List[str]

class FinancialProduct(BaseModel):
    product_id: str = Field(..., description="Unique product identifier like MF001")
    product_name: str
    product_type: str  
    risk_level: str
    min_investment: str
    expected_return: str
    expense_ratio: str
    category: str
    description: str
    
class GoalPredictionResult(BaseModel):
    goal_success:str 
    probability: float 
    success_factors:List[str]
    challenges:List[str]
    timeline_analysis:Dict[str,Any]

class PersonaResult(BaseModel):
    persona_type:str 
    confidence_score:float 
    characteristics:List[str]
    #behavioral_insights:List[str]
    behavioral_insights: Dict[str, List[str]]

class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    product_type: str
    suitability_score: float 
    justification: str
    risk_alignment: str 
    expected_returns: str 
    fees: str

class MeetingGuide(BaseModel): #?
    agenda_items: List[str]
    key_talking_points: List[str]
    questions_to_ask: List[str]
    objection_handling: Dict[str,str]
    next_steps: List[str]
    estimated_duration: float 

class ComplianceCheck(BaseModel):
    is_compliant: bool 
    compliance_score:float 
    violations: List[str]
    warnings: List[str]
    required_disclosures:List[str]

class AgentExecution(BaseModel): #?
    agent_name:str 
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None 
    status: str 
    error_message: Optional[str] = None 
    execution_time: Optional[float] = None  


## Aggregated State Models

class ProspectState(BaseModel): #?
    prospect_data : Optional[ProspectData] = None
    validation_errors: Optional[List[str]]= None
    data_quality_score: Optional[float] = None
    missing_fields: Optional[List[str]] = None 

class AnalysisState(BaseModel): #?
    risk_assessment: Optional[RiskAssessmentResult] = None 
    goal_prediction: Optional[GoalPredictionResult]= None 
    persona_classification: Optional[PersonaResult] = None

class RecommendationState(BaseModel): #?
    recommended_products: Optional[List[ProductRecommendation]]= None
    portfolio_allocation:Optional[Dict[str,float]] = None
    compliance_check: Optional[ComplianceCheck] = None 
    justification_text:Optional[str]=None 

## Not Aggregating 
class MeetingState(BaseModel): #?
    meeting_guide: Optional[MeetingGuide]= None
    presentation_slides:Optional[List[str]] = None 
    client_materials:Optional[List[str]] = None 

class ChatState(BaseModel): #?
    conversation_history: Optional[List[str]] = None 
    current_query: Optional[str]= None 
    context: Optional[str] = None 
    response: Optional[str] = None 

## Master State

class WorkflowState(BaseModel):
    workflow_id : str 
    session_id: Optional[str]

    prospect:Optional[ProspectState] =None 
    analysis:Optional[AnalysisState] = None 
    #recommendations:Optional[RecommendationState]=None
    recommendations:Optional[RecommendationState]=Field(default_factory=lambda:RecommendationState())
    risk_assessment_result:Optional[RiskAssessmentResult]=None
    goal_planning_result:Optional[GoalPredictionResult]=None 
 
    created_at: datetime.datetime = datetime.datetime.now(datetime.UTC)
    updated_at: datetime.datetime = datetime.datetime.now(datetime.UTC)

    agent_executions: List[AgentExecution]=[] 
    configuration: Optional[Dict[str,str]] = None
    overall_confidence: Optional[float] =None 

    key_insights:Optional[List[str]]=[]
    action_items:Optional[List[str]]=[]

    current_step: Optional[str] = None # added extra cuz was needed for graph.py
    completed_steps: Optional[List[str]] = []   # add this
    failed_steps: Optional[List[str]] = []      # add this

    chat:Optional[str]= None 

    def add_agent_execution(self,agent_name:str,status:str ="running", error_message:Optional[str]=None):
        execution = AgentExecution(
            agent_name=agent_name,
            start_time=datetime.datetime.utcnow(),
            status=status,
            error_message=error_message,
        )
        self.agent_executions.append(execution)
        self.updated_at=datetime.datetime.utcnow()
        return execution # extra for main.py error 

    def complete_agent_execution(self,agent_name:str,status:str ="completed", error_message:Optional[str]=None):
        for execution in self.agent_executions:
            if execution.agent_name == agent_name and execution.end_time is None:
                execution.end_time= datetime.datetime.utcnow()
                execution.status = status
                execution.error_message = error_message
                execution.execution_time = (execution.end_time- execution.start_time).total_seconds()
                break 
        self.updated_at = datetime.datetime.utcnow()

    def get_execution_summary(self) -> Dict[str,any]:
        completed = sum(1 for e in self.agent_executions if e.status == "completed")
        failed = sum(1 for e in self.agent_executions if e.status == "failed")
        running = sum( 1 for e in self.agent_executions if e.status == "running")
        total_time = sum(e.execution_time or 0 for e in self.agent_executions)
        return{
            "total_agents":len(self.agent_executions),
            "completed": completed,
            "failed": failed,
            "running": running,
            "total_execution_time_sec": total_time,
            "last_updated": self.updated_at.isoformat(),
        }


# testing 

# if __name__ == "__main__":
#     state = WorkflowState(
#         workflow_id="wf_123",
#         session_id="sess_456",
#             )
    
#     state.add_agent_execution("RiskAnalyzer")
#     time.sleep(2)
#     state.complete_agent_execution("RiskAnalyzer",status = "completed")

#     state.add_agent_execution("GoalPredictor")
#     time.sleep(0.5)
#     state.complete_agent_execution("GoalPredictor",status="failed",error_message="Timeout")

#     state.add_agent_execution("PersonaBuilder")
    
#     summary=state.get_execution_summary()
#     print(summary)
