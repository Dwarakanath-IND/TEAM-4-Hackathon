from typing import TypedDict, List
import pandas as pd
from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from state import WorkflowState, FinancialProduct
from langraph_agents.base_agent import CriticalAgent
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

Gemini_api_key = os.getenv("GEMINI_API_KEY_1")
Tavily_api_key = os.getenv("TAVILY_API_KEY")





class FinancialDataAgent(CriticalAgent):
    def __init__(self):
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=Gemini_api_key, temperature=0)
    
        search_tool = TavilySearch(api_key = Tavily_api_key,max_results=5)
        # search_tool = TavilySearchResults(api_key = Tavily_api_key,max_results=5)
        llm_with_search = llm.bind_tools([search_tool])

        super().__init__(
            name = "FinancialDataAgent",
            description = "Fetches and stuctures financial product data (MFs, Bonds etc.) using Gemini and Tavily",
            llm = llm_with_search
        )
    
    def get_prompt_template(self)-> ChatPromptTemplate:
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
    You are a financial analyst. Provide structured data for 10 Indian investment products
    such as mutual funds or bonds. Each entry must include:

    product_id, product_name, product_type, risk_level, min_investment, expected_return,
    expense_ratio, category, and description.

    Guidelines:
    - For product_id, use format like MF001, FD001, ELSS001, etc. depending on wether the instrument is a Mutual Fund, Fixed Deposti, ELSS or anything else etc.
    - For product_type, specify the exact type like Mutual Fund, Fixed Deposit, ELSS, PPF, Bond Fund etc.
    - For risk_level, classify as Low, Moderate, or High.
    - For min_investment, provide a numeric value (e.g., 1000).
    - For expected_return, provide in percentage format (e.g., "8-12%" or "7.5%").
    - For expense_ratio, provide in percentage format (e.g., "1.2%" or "0%"). If a product involved has no concept of expense ratio, then give "0%
    - For category, classify it into either Equity, Debt, or Hybrid, based on how its managed.
    - Keep descriptions concise (under 50 words).

    """
        return ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            ("user",user_prompt)
        ])
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        structured_llm = self.llm.with_structured_output(FinancialProduct)

        prompt_template = self.get_prompt_template()
        products = ["Mutual Fund","Bond","Fixed Deposits","ELSS","PPF"]
        results = []
        for product in products:
            for i in range(2):
                res = await structured_llm.ainvoke(f"{prompt_template} Focus on {product}.")
                results.append(res.dict())
        
        df = pd.DataFrame(results)
        df.to_csv("financial_products.csv",index = False)
        self.logger.info("Save data to financial_products.csv")

        return state
