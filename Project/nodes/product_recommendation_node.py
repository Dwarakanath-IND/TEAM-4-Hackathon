# TODO: Import necessary modules and agents
import logging
from datetime import datetime
from state import WorkflowState
from langraph_agents.agents.product_specialist_agent import ProductSpecialistAgent

# TODO: Create async product_recommendation_node function accepting WorkflowState
async def product_recommendation_node(state: WorkflowState) -> WorkflowState:
    try:
        # TODO: Check that risk_assessment completed successfully
        if "risk_assessment" not in WorkflowState.completed_steps:
            raise Exception("Risk assessment must be completed before product recommendation")
            
        # TODO: Instantiate ProductSpecialistAgent
        product_specialist_agent = ProductSpecialistAgent()

        # TODO: Log node execution with timestamps
        start_time = datetime.now()
        logging.info(f"Product recommendation node started at {start_time}")

        # TODO: Call agent.execute(state) with error handling
        await product_specialist_agent.execute(state)

        # TODO: Update state.current_step to "product_recommendation"
        state.current_step = "product_recommendation"

        # TODO: Add step to completed_steps on success or failed_steps on failure
        state.completed_steps.append("product_recommendation")
        end_time = datetime.now()
        logging.info(f"Product recommendation node completed successfully at {end_time}")
    except Exception as e:
        state.failed_steps.append("product_recommendation")

        # TODO: Raise exception on failure (critical node)
        raise Exception(f"Critical failure in product recommendation node: {e}")
        # logging.error(f"Product recommendation node failed: {e}")

    # TODO: Return updated state
    return state
    
