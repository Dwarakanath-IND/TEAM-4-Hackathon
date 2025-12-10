# TODO: Import ABC, abstractmethod from abc
from abc import ABC, abstractmethod
# TODO: Import Dict, Any, Optional from typing
from typing import Dict, Any, Optional
# TODO: Import BaseModel from pydantic
from pydantic import BaseModel
# TODO: Import datetime
from datetime import datetime
#Added asyncio extra
import asyncio
#Added loguru extra, will have to implement
from loguru import logger

from state import WorkflowState
import logging

# Added these in extra
from langchain_core.language_models import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv 
import os 
load_dotenv()
#genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))
#api_key=os.getenv("GEMINI_API_KEY_1")

# TODO: Create abstract BaseAgent class with:
#   - name attribute
#   - description attribute
#   - __init__ method setting agent name and logger
# TODO: Create abstract async run method accepting WorkflowState
# TODO: Implement validate_input method to check required fields
# TODO: Implement validate_output method to check result structure
# TODO: Implement execute method wrapping run with validation and error handling
# TODO: Implement track_execution method recording start/end times and success/failure
# TODO: Add monitoring capabilities for performance metrics
# TODO: Add error handling with detailed logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        llm: Optional[BaseLanguageModel] = None,
        temperature: float = 0.1,
        max_tokens: int = 4000      
    ):
        self.name = name
        self.description = description

        self.logger = logging.getLogger(f"Agent-{self.name}")
        self.logger.setLevel(logging.INFO)
    
        if llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model = "gemini-2.5-flash",
                google_api_key = os.getenv("GEMINI_API_KEY_1"),   # Come back to this and look at it Niraj - ArcNir
                temperature = temperature,
                max_tokens = max_tokens
            )
        else:
            self.llm = llm

        self.created_at = datetime.now()
        self.execution_count = 0
        self.total_execution_time =  0.0
        self.success_count = 0
        self.error_count = 0

        logger.info(f"Initialized agent: {self.name}")


    @abstractmethod
    async def execute(self, state: WorkflowState) -> WorkflowState:
        pass

    @abstractmethod
    def get_prompt_template(self)-> ChatPromptTemplate:
        pass

    async def run(self, state: WorkflowState) -> WorkflowState:
        self.logger.info(f"Running agent {self.name} with state: {state.__dict__}") #temp
        try:
                
            execution = state.add_agent_execution(self.name)    #Check this out if things do not

            if not self.validate_input(state):
                raise ValueError(f"Input validation failed, Agent: {self.name}")
            result_state = await self.execute(state)

            if not self.validate_output(result_state):
                raise ValueError(f"Output validation failed, Agent: {self.name}")

            state.complete_agent_execution(self.name,status = "completed")
            self.success_count += 1

            self.logger.info(f"Successfully completed execution for agent: {self.name}")
            return result_state
        except Exception as e:
            error_msg = f"There is an error in agent {self.name}: {str(e)}"
            self.logger.error(error_msg)

            state.complete_agent_execution(self.name, status= "failed", error_message = error_msg)
            self.error_count += 1

            if hasattr(state,'errors'):
                state.errors.append(error_msg)
            
            if self.is_critical(): #This part is a bit sketchy to me- ArcNir
                raise
            else:
                raise state
        finally:
            self.execution_count +=1
            if execution.execution_time:
                self.total_execution_time += execution.execution_time

    

    def validate_input(self, state:WorkflowState) -> bool:
        return state is not None

    def validate_output(self,state:WorkflowState) -> bool:
        return state is not None       

    def is_critical(self)->bool:
        return False

    async def generate_response(self,prompt_template: ChatPromptTemplate,input_variables: Dict[str,Any]) -> str:
        try:

            chain = prompt_template | self.llm | StrOutputParser()
            response = await chain.ainvoke(input_variables)
            return response.strip()
        except Exception as e:
            self.logger.error(f"There is an error generating a respones: {str(e)}")
            raise
    def get_system_prompt(self)->str:
        return f"""You are {self.name}, a specialized AI agent in a financial advisory system.

        Your role: {self.description}

        You are to follow the following guidelines.

        Guidelines:
        - Be absolutely clear and concise in your responses
        - Use Data-driven insights or advice when available
        - Maintain confidentiality for the client and ensure professionalism
        - Conisder regulatroy compliance and risk factors


        Always respond in a structured manner such that your output can be easily understood by other agents.
        """

    def get_performance_metrics(self)-> Dict[str,any]:
        avg_execution_time = (
            self.total_execution_time/self.execution_count if self.execution_count > 0 else 0
        )

        success_rate = (
            self.success_count/self.execution_count if self.execution_count > 0 else 0
        )

        return {
            "agent_name":self.name,
            "execution_count":self.execution_count,
            "success_count":self.success_count,
            "error_count":self.error_count,
            "success_rate":success_rate,
            "total_execution_time":self.total_execution_time,
            "average_execution_time":avg_execution_time,
            "created_at":self.created_at.isoformat()
        }

    def reset_metrics(self):
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.success_count = 0,
        self.error_count = 0,
        self.logger.info(f"Reset all the metrics for the agent: {self.name}")

    def __str__(self) -> str:
        return f"Agent({self.name})"
    
    def __repr__(self) -> str:
        return f"Agent(name={self.name})"

class CriticalAgent(BaseAgent):
    def is_critical(self)->bool:
        return True

class OptionalAgent(BaseAgent):
    def is_critical(self)->bool:
        return False



        
