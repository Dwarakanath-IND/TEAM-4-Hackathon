# # TODO: Import uuid, Dict, Any, Optional from typing
# # TODO: Import datetime
# # TODO: Import StateGraph and END from langgraph.graph
# # TODO: Import MemorySaver from langgraph.checkpoint.memory
# # TODO: Import WorkflowState and ProspectData from state
# # TODO: Import all specialized agents (DataAnalystAgent, RiskAssessmentAgent, PersonaAgent, ProductSpecialistAgent)
# # TODO: Import get_logger from config
# # TODO: Create ProspectAnalysisWorkflow class with __init__ that initializes agents and builds workflow
# # TODO: Implement _build_workflow to create StateGraph with 5 nodes:
# #   - data_analysis node
# #   - risk_assessment node
# #   - persona_classification node
# #   - product_recommendation node
# #   - finalize_analysis node
# # TODO: Set workflow entry point to data_analysis
# # TODO: Define sequential edges connecting all nodes ending at END
# # TODO: Compile graph with MemorySaver checkpointer
# # TODO: Implement async node methods that execute agents and track completed/failed steps
# # TODO: Implement _finalize_analysis_node to calculate overall confidence, generate insights and action items
# # TODO: Implement _generate_key_insights extracting insights from risk, persona, products, and data quality
# # TODO: Implement _generate_action_items based on validation errors, risk level, recommendations, and persona
# # TODO: Implement async analyze_prospect method creating initial state and invoking graph
# # TODO: Implement async get_workflow_state to retrieve state from checkpoint
# # TODO: Implement get_workflow_summary returning workflow configuration details

import uuid
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import WorkflowState, ProspectData
from langraph_agents.agents.data_analyst_agent import DataAnalystAgent
from langraph_agents.agents.risk_assessment_agent import RiskAssessmentAgent
from langraph_agents.agents.persona_agent import PersonaAgent
from langraph_agents.agents.product_specialist_agent import ProductSpecialistAgent
from langraph_agents.agents.goal_planning_agent import GoalPlanningAgent
from langraph_agents.agents.data_fetch_agent import FinancialDataAgent
from config.logging_config import get_logger



class ProspectAnalysisWorkflow:
    def __init__(self):
        self.logger = get_logger("ProspectAnalysisWorkflow")
        self.graph = None
        self.checkpointer = MemorySaver()
        self.progress_callback = None  # Add progress callback support
        self._build_workflow()

    def _build_workflow(self):
        self.logger.info("Building prospect analysis workflow")
        # Initializing Agents
        self.data_analyst = DataAnalystAgent()
        self.risk_assessor = RiskAssessmentAgent()
        self.persona_classifier = PersonaAgent()
        self.product_specialist = ProductSpecialistAgent()
        self.goal_planning = GoalPlanningAgent()
        self.data_fetcher = FinancialDataAgent() #Placeholder

        #testing if agents initialized
        for agent in [self.data_analyst, self.risk_assessor, self.persona_classifier, self.product_specialist,self.goal_planning]:
            if agent is None:
                self.logger.error(f"Agent {agent} failed to initialize.")
                raise RuntimeError(f"Agent {agent} initialization failed")
    
        # Creating graph workflow
        workflow = StateGraph(WorkflowState)
        # Adding Agent Nodes
        workflow.add_node("data_analysis", self._data_analysis_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("goal_planning",self._goal_planning_node)
        workflow.add_node("financial_data_fetch", self._financial_data_fetch_node)
        workflow.add_node("persona_classification", self._persona_classification_node)
        workflow.add_node("product_recommendation", self._product_recommendation_node)
        workflow.add_node("finalize_analysis", self._finalize_analysis_node)
        # Defining workflow entry point
        workflow.set_entry_point("data_analysis")
        # Sequential flow with conditional routing
        workflow.add_edge("data_analysis","risk_assessment")
        workflow.add_edge("risk_assessment","goal_planning")
        workflow.add_edge("goal_planning","persona_classification")
        workflow.add_edge("persona_classification","financial_data_fetch")
        workflow.add_edge("financial_data_fetch","product_recommendation")
        # workflow.add_edge("persona_classification","product_recommendation")
        workflow.add_edge("product_recommendation","finalize_analysis")
        workflow.add_edge("finalize_analysis", END)
        # Compiling the graph
        self.graph = workflow.compile(checkpointer = self.checkpointer)
        self.logger.info("Workflow compiled successfully")
        print(self.graph.get_graph().draw_ascii())

    def _notify_progress(self, percent: int, message: str):
        """Helper to notify progress callback if available"""
        if self.progress_callback:
            try:
                self.progress_callback(percent, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    # 1) Data Analysis Node
    async def _data_analysis_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("Executing data analysis node")
        state.current_step = "data_analysis"
        self._notify_progress(5, ("Validating prospect data..."))
        try:
            result_state = await self.data_analyst.run(state)
            result_state.completed_steps.append("data_analysis")
            return result_state
        except Exception as e:
            self.logger.error(f"Data analysis failed: {str(e)}")
            state.failed_steps.append("data_analysis")
            raise

    # 2) Risk Assessment Node
    async def _risk_assessment_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("Executing risk assessment node")
        state.current_step = "risk_assessment"
        self._notify_progress(20, ("Executing risk assessment agent..."))
        try:
            result_state = await self.risk_assessor.run(state)
            result_state.completed_steps.append("risk_assessment")
            return result_state
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            state.failed_steps.append("risk_assessment")
            raise


    # 3) Goal Planning Node
    async def _goal_planning_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("Executing goal planning node")
        state.current_step = "goal_planning"
        self._notify_progress(35, ("Executing goal planning agent..."))
        try:
            result_state = await self.goal_planning.run(state)
            result_state.completed_steps.append("goal_planning")
            return result_state
        except Exception as e:
            self.logger.error(f"Goal Planning failed: {str(e)}")
            state.failed_steps.append("goal_planning")
            raise

    # 4) Persona Classification Node
    async def _persona_classification_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("Executing persona classification node")
        state.current_step = "persona_classification"
        self._notify_progress(50, ("Classifying investor persona..."))
        try:
            result_state = await self.persona_classifier.run(state)
            result_state.completed_steps.append("persona_classification")
            return result_state
        except Exception as e:
            self.logger.error(f"Persona classification failed: {str(e)}")
            state.failed_steps.append("persona_classification")
            # Non-critical continue without persona
            return state

    # 4) Financial Data Fetch Node
    async def _financial_data_fetch_node(self,state: WorkflowState) -> WorkflowState:
        self.logger.info("Executing Financial Data fetch node")
        state.current_step = "financial_data_fetch"
        self._notify_progress(65, ("Fetching live financial data..."))
        try:
            result_state = await self.data_fetcher.run(state)
            result_state.completed_steps.append("financial_data_fetch")
            return result_state
        except Exception as e:
            self.logger.error(f"Financial Data fetch failed: {str(e)}")
            state.failed_steps.append("financial_data_fetch")
            raise

    # 5) Product Recommendation Node
    async def _product_recommendation_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("Executing product recommendation node")
        state.current_step = "product_recommendation"
        self._notify_progress(80, ("Generating product recommendations..."))
        try:
            result_state = await self.product_specialist.run(state)
            result_state.completed_steps.append("product_recommendation")
            return result_state
        except Exception as e:
            self.logger.error(f"Product recommendation failed: {str(e)}")
            state.failed_steps.append("product_recommendation")
            raise
    
    # 6) Finalize Analysis Node
    async def _finalize_analysis_node(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("Finalizing analysis")
        state.current_step = "finalize_analysis"
        self._notify_progress(90, ("Finalizing analysis and generating insights..."))
        try:
            # Overall confidence calculation
            confidence_scores = []
            if state.analysis.risk_assessment:
                confidence_scores.append(state.analysis.risk_assessment.confidence_score)
            if state.analysis.goal_prediction:
                confidence_scores.append(state.analysis.goal_prediction.probability)
            if state.analysis.persona_classification:
                confidence_scores.append(state.analysis.persona_classification.confidence_score)
            if state.prospect.data_quality_score:
                confidence_scores.append(state.prospect.data_quality_score)
            if confidence_scores:
                state.overall_confidence = sum(confidence_scores) / len(confidence_scores)
            # generate key insights
            state.key_insights = self._generate_key_insights(state)
            # generate action items
            state.action_items = self._generate_action_items(state)
            # timestamps update
            state.updated_at = datetime.now()
            state.completed_steps.append("finalize_analysis")
            self.logger.info("Analysis finalized successfully")
            return state
        except Exception as e:
            self.logger.error(f"Analysis finalized failed: {str(e)}")
            state.failed_steps.append("finalize_analysis")
            return state

    # Generates key insights from the analysis
    def _generate_key_insights(self, state: WorkflowState) -> list:
        insights = []
        if state.analysis.risk_assessment:
            insights.append(f"Risk Profile: {state.analysis.risk_assessment.risk_level}")
        if state.analysis.goal_prediction:
            insights.append(f"Goal Prediction: {state.analysis.goal_prediction.probability}")
        if state.analysis.persona_classification:
            insights.append(f"Investor Persona: {state.analysis.persona_classification.persona_type}")
        if state.recommendations.recommended_products:
            top_product = state.recommendations.recommended_products[0]
            insights.append(f"Top Recommendation: {top_product.product_name}")
        if state.prospect.data_quality_score:
            if state.prospect.data_quality_score > 0.8:
                insights.append("High data quality - reliable analysis")
            elif state.prospect.data_quality_score < 0.6:
                insights.append("Data quality concerns - additional verification needed")
        return insights
    
    # Generates action items for RM
    def _generate_action_items(self, state: WorkflowState) -> list:
        actions = []
        if state.prospect.validation_errors:
            actions.append("Verify and correct data validation errors")
        if (state.analysis.risk_assessment and state.analysis.risk_assessment.risk_level == "High") :
            actions.append("Discuss risk tolerance and investment experience in detail")
        if state.analysis.goal_prediction:
            goal_result=state.analysis.goal_prediction
            if goal_result.goal_success.lower()=="likely":
                actions.append("Review and confirm client's short-term and long-term goals for alignment")
        elif goal_result.goal_success.lower()=="unlikely":
            actions.append("Revisit client's income, expense, and investment allocations to improve goal feasability"
            )
        if goal_result.success_factors:
            top_factors=", ".join(goal_result.success_factors[:2])
            actions.append(f"Reinforce strong factors: {top_factors}")
        if state.recommendations.recommended_products:
            actions.append("Present top product recommendations with justifications")
        if state.analysis.persona_classification:
            persona_type = state.analysis.persona_classification.persona_type
            if persona_type == "Cautious Planner":
                actions.append("Focus on capital preservation and security features")
            elif persona_type == "Aggressive Growth":
                actions.append("Emphasize growth potential and ling-term returns")
        actions.append("Schedule follow-up meeting to discuss recommendations")
        return actions

    # Analyze a prospect using the workflow
    async def analyze_prospect(self, prospect_data: Dict[str, Any], session_id: Optional[str] = None) -> WorkflowState:
        # create initial state
        workflow_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        initial_state = WorkflowState(
            workflow_id = workflow_id,
            session_id = session_id,
            created_id = datetime.now(),
            updated_at = datetime.now()
        )
        # set prospect data
        #initial_state.prospect.prospect_data = ProspectData(**prospect_data)
        from state import ProspectState
        initial_prospect_state = ProspectState(prospect_data=ProspectData(**prospect_data))
        self.logger.info(f"Starting prospect analysis for {prospect_data.get('name', 'Unknown')}")
        try:
            # Execute workflow
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.graph.ainvoke(initial_state, config  = config)
            self.logger.info(f"Prospect analysis completed successfully. Workflow ID: {workflow_id}")
            return final_state
        except Exception as e:
            self.logger.error(f"Prospect analysis failed: {str(e)}")
            raise

        
    # To get current state of a workflow session
    async def get_workflow_state(self, session_id: str) -> Optional[WorkflowState]:
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = await self.graph.aget_state(config)
            return state.values if state else None
        except Exception as e:
            self.logger.error(f"Failed to get workflow state: {str(e)}")
            return None

    # Workflow configuration summary
    def get_workflow_summary(self) -> Dict[str, Any]:
        return {
            "workflow_name": "Prospect Analysis Workflow",
            "agents": [
                self.data_analyst.name,
                self.risk_assessor.name,
                self.goal_planning.name,
                self.persona_classifier.name,
                self.product_specialist.name
            ],
            "steps": [
                "data_analysis",
                "risk_assessment",
                "goal_planning",
                "persona_classification",
                "product_recommendation",
                "finalize_analysis"
            ],
            "critical_agents": [
                self.data_analyst.name,
                self.risk_assessor.name,
                self.product_specialist.name
            ],
            "optional_agents": [
                self.persona_classifier.name,
                self.goal_planning.name
            ]
        }
    
    async def run(self, state: WorkflowState) -> WorkflowState:
        """Wrapper to allow workflow.run(state) usage with progress tracking"""
        config = {"configurable": {"thread_id": state.session_id}}
        return await self.graph.ainvoke(state, config=config)
    
    async def run_with_progress(self, state: WorkflowState, progress_callback: Callable[[int, str], None]) -> WorkflowState:
        """Execute workflow with progress tracking"""
        self.progress_callback = progress_callback
        try:
            # Initial progress
            progress_callback(0, "Initializing workflow...")
            
            # Execute the workflow
            config = {"configurable": {"thread_id": state.session_id}}
            result = await self.graph.ainvoke(state, config=config)
            
            # Final progress
            progress_callback(100, "Analysis complete!")
            return result
        finally:
            # Clean up callback reference
            self.progress_callback = None







        
