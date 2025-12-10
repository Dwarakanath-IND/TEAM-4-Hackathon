# TODO: Import all necessary pydantic and typing modules
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
import datetime
import pandas as pd
import time

# TODO: Define all state model classes mirroring state.py but for agent internal use
# TODO: Include validation logic for state transitions
# TODO: Add serialization methods for logging and debugging

class ProspectData(BaseModel):
    prospect_id: Optional[str] = Field(None)
    name: Optional[str] = Field(None)
    age: int = Field(...)
    annual_income: float = Field(...)
    current_savings: float = Field(...)
    target_goal_amount: Optional[float] = Field(None)
    investment_horizon_years: int = Field(...)
    number_of_dependents: int = Field(...)
    investment_experience_level: str = Field(...)
    investment_goal: Optional[str] = Field(None)

    @field_validator("age")
    def validate_age(cls, v):
        if v <= 0:
            raise ValueError("Age must be positive")
        return v

    @field_validator("annual_income", "current_savings")
    def validate_nonnegative(cls, v):
        if v < 0:
            raise ValueError("Values must be non-negative")
        return v

class RiskAssessmentResult(BaseModel):
    risk_level: str
    confidence_score: float
    risk_factors: List[str]
    recommendations: List[str]

    @field_validator("confidence_score")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("confidence_score must be between 0 and 1")
        return v

class GoalPredictionResult(BaseModel):
    goal_success: str
    probability: float
    success_factors: List[str]
    challenges: List[str]
    timeline_analysis: Dict[str, Any]

    @field_validator("probability")
    def validate_probability(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("probability must be between 0 and 1")
        return v

class PersonaResult(BaseModel):
    persona_type: str
    confidence_score: float
    characteristics: List[str]
    behavioral_insights: List[str]

    @field_validator("confidence_score")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("confidence_score must be between 0 and 1")
        return v

class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    product_type: str
    suitability_score: float
    justification: str
    risk_alignment: str
    expected_returns: str
    fees: str

class MeetingGuide(BaseModel):
    agenda_items: List[str]
    key_talking_points: List[str]
    questions_to_ask: List[str]
    objection_handling: Dict[str, str]
    next_steps: List[str]
    estimated_duration: float

class ComplianceCheck(BaseModel):
    is_compliant: bool
    compliance_score: float
    violations: List[str]
    warnings: List[str]
    required_disclosures: List[str]

    @field_validator("compliance_score")
    def validate_compliance(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("compliance_score must be between 0 and 1")
        return v

class AgentExecution(BaseModel):
    agent_name: str
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    status: str
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

    @field_validator("status")
    def validate_status(cls, v):
        allowed = {"running", "completed", "failed"}
        if v not in allowed:
            raise ValueError(f"Invalid status: {v}, must be one of {allowed}")
        return v

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump()

class ProspectState(BaseModel):
    prospect_data: Optional[ProspectData] = None
    validation_errors: Optional[List[str]] = None
    data_quality_score: Optional[float] = None
    missing_fields: Optional[List[str]] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump()

class AnalysisState(BaseModel):
    risk_assessment: Optional[RiskAssessmentResult] = None
    goal_prediction: Optional[GoalPredictionResult] = None
    persona_classification: Optional[PersonaResult] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump()

class RecommendationState(BaseModel):
    recommended_products: Optional[List[ProductRecommendation]] = None
    portfolio_allocation: Optional[Dict[str, float]] = None
    compliance_check: Optional[ComplianceCheck] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump()

class MeetingState(BaseModel):
    meeting_guide: Optional[MeetingGuide] = None
    presentation_slides: Optional[List[str]] = None
    client_materials: Optional[List[str]] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump()

class ChatState(BaseModel):
    conversation_history: Optional[List[str]] = None
    current_query: Optional[str] = None
    context: Optional[str] = None
    response: Optional[str] = None

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump()

class WorkflowState(BaseModel):
    workflow_id: str
    session_id: Optional[str]

    prospect: Optional[ProspectState] = None
    analysis: Optional[AnalysisState] = None
    recommendations: Optional[RecommendationState] = None
    risk_assessment_result: Optional[RiskAssessmentResult] = None

    created_at: datetime.datetime = datetime.datetime.utcnow()
    updated_at: datetime.datetime = datetime.datetime.utcnow()

    agent_executions: List[AgentExecution] = []
    configuration: Optional[Dict[str, str]] = None
    overall_confidence: Optional[float] = None

    key_insights: Optional[List[str]] = []
    action_items: Optional[List[str]] = []

    def add_agent_execution(self, agent_name: str, status: str = "running", error_message: Optional[str] = None):
        execution = AgentExecution(
            agent_name=agent_name,
            start_time=datetime.datetime.utcnow(),
            status=status,
            error_message=error_message,
        )
        self.agent_executions.append(execution)
        self.updated_at = datetime.datetime.utcnow()

    def complete_agent_execution(self, agent_name: str, status: str = "completed", error_message: Optional[str] = None):
        for execution in self.agent_executions:
            if execution.agent_name == agent_name and execution.end_time is None:
                execution.end_time = datetime.datetime.utcnow()
                execution.status = status
                execution.error_message = error_message
                execution.execution_time = (execution.end_time - execution.start_time).total_seconds()
                break
        self.updated_at = datetime.datetime.utcnow()

    def get_execution_summary(self) -> Dict[str, Any]:
        completed = sum(1 for e in self.agent_executions if e.status == "completed")
        failed = sum(1 for e in self.agent_executions if e.status == "failed")
        running = sum(1 for e in self.agent_executions if e.status == "running")
        total_time = sum(e.execution_time or 0 for e in self.agent_executions)
        return {
            "total_agents": len(self.agent_executions),
            "completed": completed,
            "failed": failed,
            "running": running,
            "total_execution_time_sec": total_time,
            "last_updated": self.updated_at.isoformat(),
        }

    def serialize(self) -> Dict[str, Any]:
        return self.model_dump()



if __name__ == "__main__":
    pass
    # from state import IncidentState
    # from langraph_agents.utils.agent_execution import update_agent_execution_state

    # Create a dummy IncidentState
    # state = IncidentState()
    # print("Before update:", state.agent_execution)

    # # Call the function
    # update_agent_execution_state(state, "GoalPlanningAgent", "completed")

    # # Show what changed
    # print("After update:", state.agent_execution)
