# TODO: Import necessary modules and agents
import logging
from datetime import datetime
from state import WorkflowState
from langraph_agents.agents.persona_agent import PersonaAgent

# TODO: Create async persona_node function accepting WorkflowState
async def persona_node(state: WorkflowState) -> WorkflowState:

    # TODO: Instantiate PersonaAgent
    persona_agent = PersonaAgent()

    # TODO: Try to execute agent with error handling
    try:
        # TODO: Log node execution with timestamps
        start_time = datetime.now()
        logging.info(f"Persona node staarted at {start_time}")
        await persona_agent.execute(state)

        # TODO: Update state.current_step to "persona_classification"
        state.current_step = "persona_classification"

        # TODO: Add step to completed_steps on success or failed_steps on failure
        state.completed_steps.append("persona_classification")
        end_time = datetime.now()
        logging.info(f"Persona node completed successfully at {end_time}")
    except Exception as e:
        state.failed_steps.append("persona_classification")
        logging.error(f"Persona node failed: {e}")

# TODO: Return state without raising (optional node - continue even if fails)
    return state
