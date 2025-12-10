
import logging
from datetime import datetime
from state import WorkflowState
from langraph_agents.agents.goal_planning_agent import GoalPlanningAgent

async def risk_assessment_node(state: WorkflowState) -> WorkflowState:
    try:
        if "data_analysis" not in state.completed_steps:
            raise Exception("Data analysis must be completed before goal planning")
            
        goal_planning_agent = GoalPlanningAgent() 
        start_time = datetime.now()
        logging.info(f"Goal planning node started at {start_time}")
        await goal_planning_agent.execute(state)
        state.current_step = "goal_planning"
        state.completed_steps.append("goal_planning")
        end_time = datetime.now()
        logging.info(f"Goal planning node completed successfully at {end_time}")
    except Exception as e:
        state.failed_steps.append("goal_planning")
        raise Exception(f"Critical failure in goal planning node: {e}")

    return state
