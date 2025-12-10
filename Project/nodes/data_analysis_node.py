# TODO: Import necessary modules and agents
import logging
from datetime import datetime
from state import WorkflowState
from langraph_agents.agents.data_analyst_agent import DataAnalystAgent

# TODO: Create async data_analysis_node function accepting WorkflowState
async def data_analysis_node(state: WorkflowState) -> WorkflowState:

    # TODO: Instantiate DataAnalystAgent
    data_analyst_agent = DataAnalystAgent()
    try:
        # TODO: Log node execution with timestamps
        start_time = datetime.now()
        logging.info(f"Data analysis node started at {start_time}")

        # TODO: Call agent.execute(state) with error handling
        await data_analyst_agent.execute(state)

        # TODO: Update state.current_step to "data_analysis"
        state.current_step = "data_analysis"

        # TODO: Add step to completed_steps on success or failed_steps on failure
        state.completed_steps.append("data_analysis")
        end_time = datetime.now()
        logging.info(f"Data analysis node completed successfully at {end_time}")
    except Exception as e:
        state.failed_steps.append("data_analysis")
        logging.error(f"Data analysis node failed: {e}")

    # TODO: Return updated state
    return state
