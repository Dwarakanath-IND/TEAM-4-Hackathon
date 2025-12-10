# new ai

# main.py
import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from functools import lru_cache

import streamlit as st
import pandas as pd
import datetime
import nest_asyncio

# state models are provided by user (do not change)
from state import WorkflowState, ProspectState, AnalysisState, ChatState
from graph import ProspectAnalysisWorkflow
from langraph_agents.agents.risk_assessment_agent import RiskAssessmentAgent
from langraph_agents.agents.goal_planning_agent import GoalPlanningAgent
import langraph_agents.agents.rm_assistant_agent as rm_assistant_module
RMAssistant = getattr(rm_assistant_module, "RMAssistant", None) or getattr(rm_assistant_module, "RMAssistantAgent", None)
from deep_translator import GoogleTranslator

# List of languages
languages = {
    "English": "en",
    "Amharic": "am",
    "Arabic": "ar",
    "Basque": "eu",
    "Bengali": "bn",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Cherokee": "chr",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "Estonian": "et",
    "Filipino": "fil",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Hebrew": "iw",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Malay": "ms",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese (Brazil)": "pt-BR",
    "Portuguese (Portugal)": "pt-PT",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "sr",
    "Chinese (PRC)": "zh-CN",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Chinese (Taiwan)": "zh-TW",
    "Turkish": "tr",
    "Urdu": "ur",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
    "Welsh": "cy"
}


# Let user choose a language
selected_lang = st.sidebar.selectbox("Choose a language", list(languages.keys()))

import requests

# Translate any text dynamically
def t(text):
    if selected_lang == 'English':
        return text
    # translator = GoogleTranslator(source='auto', target=languages[selected_lang]).translate(text)
    # result = translator.translate("Data Quality", timeout=10)
    # from deep_translator import MyMemoryTranslator
    # MyMemoryTranslator(source='en', target='bn').translate("Data Quality")
    try:
        return GoogleTranslator(source='auto', target=languages[selected_lang]).translate(text)
    except requests.exceptions.ConnectTimeout:
        print("⚠️ Translation timed out. Check your internet or Google Translate accessibility.")
        return text  # fallback to original text
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return text

# Gemini client
import google.generativeai as genai

# ML model loaders
MODEL_CHECK_IMPORTS = {}
try:
    from ml.training.predict_risk_profile import load_risk_model
    MODEL_CHECK_IMPORTS["risk"] = True
except Exception:
    MODEL_CHECK_IMPORTS["risk"] = False

try:
    from ml.training.predict_goal_success import load_goal_model
    MODEL_CHECK_IMPORTS["goal"] = True
except Exception:
    MODEL_CHECK_IMPORTS["goal"] = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY_2"))
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# --- PAGE CONFIG ---
st.set_page_config(t("Multi-Agent Financial AI System"), page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded")


# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Page background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #2a2d3a 0%, #313647 50%, #3a3f52 100%);
        color: #FFF8D4;
        font-family: 'Inter', sans-serif;
    }

    /* Remove toolbar background */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Headers with elegant styling */
    h1 {
        color: #A3B087;
        font-weight: 300;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    h2 {
        color: #b8c29a;
        font-weight: 400;
        font-size: 1.4rem;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    
    h3 {
        color: #b8c29a;
        font-weight: 500;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    
    h4 {
        color: #A3B087;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Subtitle styling */
    .subtitle {
        color: #8fa070;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
        font-style: italic;
        opacity: 0.9;
    }

    /* Main content area */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        padding-left:3rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3d4657 0%, #435663 100%);
        border-right: 1px solid rgba(163, 176, 135, 0.2);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFF8D4;
    }

    /* Sidebar heading - remove button-like appearance */
    .sidebar-heading {
        color: #FFF8D4;
        text-align: left;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0;
        margin-bottom: 1.5rem;
        margin-top: 1rem;
        background: transparent;
        box-shadow: none;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border-bottom: 2px solid rgba(163, 176, 135, 0.3);
        padding-bottom: 0.5rem;
    }

    /* Enhanced buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #A3B087 0%, #8fa070 100%);
        color: #313647;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 12px rgba(163, 176, 135, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(163, 176, 135, 0.4);
        background: linear-gradient(135deg, #b8c29a 0%, #A3B087 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(67, 86, 99, 0.6);
        color: #FFF8D4;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px;
        border: 1px solid rgba(163, 176, 135, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(67, 86, 99, 0.8);
        border-color: rgba(163, 176, 135, 0.4);
    }
    
    .streamlit-expanderContent {
        background-color: rgba(67, 86, 99, 0.3);
        border-radius: 0 0 10px 10px;
        padding: 16px;
        border: 1px solid rgba(163, 176, 135, 0.1);
        border-top: none;
    }

    /* Select box styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(67, 86, 99, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(163, 176, 135, 0.2);
        color: #FFF8D4;
    }

    /* Tables with better contrast */
    .stTable {
        background-color: rgba(67, 86, 99, 0.4);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stTable td, .stTable th {
        background-color: transparent;
        color: #FFF8D4;
        padding: 12px;
        border-bottom: 1px solid rgba(163, 176, 135, 0.1);
    }
    
    .stTable th {
        font-weight: 600;
        color: #A3B087;
    }

    /* Enhanced progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #A3B087 0%, #8fa070 100%);
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(163, 176, 135, 0.4);
    }
    
    .stProgress > div > div {
        background-color: rgba(67, 86, 99, 0.4);
        border-radius: 10px;
    }

    /* Alert boxes with better styling */
    .stAlert {
        border-radius: 10px;
        padding: 16px;
        backdrop-filter: blur(10px);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(67, 86, 99, 0.4);
        color: #b8c29a;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(67, 86, 99, 0.6);
        color: #A3B087;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(163, 176, 135, 0.3) !important;
        color: #A3B087 !important;
    }

    /* Card containers */

    .class-container { 
        display:flex;
        flex-direction:column;
        justify-content:flex-start;
        gap:15px;
        margin-top:1rem;
    }

    .info-card {
        display:flex;
        align-items:left;
        gap:20px;
        background: rgba(67, 86, 99, 0.4);
        border-radius: 14px;
        padding: 16px 20px;
        border: 1px solid rgba(163, 176, 135, 0.2);
        transition: all 0.3 ease;
        box-shadow: 0 3x 3px rgba(0,0,0,0.2);
        margin-bottom:8px;
    }

    .card-section {
    font-size: 1.4rem;  /* increase size */
    font-weight: 600;   /* semi-bold */
    color: #A3B087;     /* same accent color */
    margin-bottom: 5px;
    }


    .metric-card {
        background: rgba(163, 176, 135, 0.15);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #A3B087;
        margin-bottom: 10px;
    }

    .card-index{
        background:linear-gradient(135deg, #A3B087 0%, #8fa070 100%);
        color: #2a2d3a;
        font-weight:700
        font-size:1rem;
        border-radius: 6px;
        border: 10px solid rgba( 163, 176, 135, 0.2);
        width: 26px;
        height:26px;
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow:0 0 10px rgba(163,176,135,0.5);
        flex-shrink:0;
    }

    .card-index-persona{
        background:linear-gradient(135deg, #A3B087 0%, #8fa070 100%);
        color: #2a2d3a;
        font-weight:700
        font-size:1.5rem;
        border-radius: 6px;
        border: 10px solid rgba( 163, 176, 135, 0.2);
        width: 26px;
        height:26px;
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow:0 0 10px rgba(163,176,135,0.5);
        flex-shrink:0;
    }
    .card-text{
        color:#FFF8D4;
        font-size:1rem;
        font-weight:400;
        line-height:1.5;
        word-wrap:break-word;
        flex:1;
    }

    /* Chat message styling */
    .chat-message {
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }

    .chat-message.user {
        background: rgba(163, 176, 135, 0.3);
        margin-left: auto;
        text-align: right;
    }

    .chat-message.assistant {
        background: rgba(67, 86, 99, 0.6);
        margin-right: auto;
    }

    .chat-container {
        background: rgba(67, 86, 99, 0.3);
        border-radius: 12px;
        padding: 20px;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 15px;
    }

    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(163, 176, 135, 0.3), transparent);
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #A3B087;
        font-weight: 600;
        font-size: 1.8rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8c29a;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: rgba(67, 86, 99, 0.6);
        color: #FFF8D4;
        border: 1px solid rgba(163, 176, 135, 0.2);
        border-radius: 10px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #A3B087;
        box-shadow: 0 0 0 1px #A3B087;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Utility & Workflow Helpers
# ---------------------------

def ensure_models_trained() -> bool:
    """Check for model artifacts and attempt to load them."""
    models_ok = True
    base = Path("ml/models")
    if not base.exists():
        logger.warning("ml/models directory not found.")
        return False

    try:
        if MODEL_CHECK_IMPORTS.get("risk"):
            _ = load_risk_model()
        else:
            raise ImportError("risk loader not available")
    except Exception as e:
        logger.warning(f"Risk model not available: {e}")
        models_ok = False

    try:
        if MODEL_CHECK_IMPORTS.get("goal"):
            _ = load_goal_model()
        else:
            raise ImportError("goal loader not available")
    except Exception as e:
        logger.warning(f"Goal model not available: {e}")
        models_ok = False

    return models_ok


@st.cache_resource
def get_workflow():
    """Return a workflow instance, caching it for the Streamlit session."""
    if ProspectAnalysisWorkflow:
        try:
            wf = ProspectAnalysisWorkflow()
            return wf
        except Exception as e:
            logger.warning(f"Failed instantiate ProspectAnalysisWorkflow: {e}")

    class SimpleWorkflow:
        def __init__(self):
            self.risk_agent = RiskAssessmentAgent() if RiskAssessmentAgent else None
            self.goal_agent = GoalPlanningAgent() if GoalPlanningAgent else None

        async def run(self, state: WorkflowState) -> WorkflowState:
            if self.risk_agent:
                try:
                    state = await self.risk_agent.execute(state)
                except Exception as e:
                    logger.exception("Risk agent failed")
            if self.goal_agent:
                try:
                    state = await self.goal_agent.execute(state)
                except Exception as e:
                    logger.exception("Goal agent failed")
            return state

    return SimpleWorkflow()


def check_model_status() -> Dict[str, bool]:
    """Try to load models and return status dict."""
    status = {"risk": False, "goal": False}
    try:
        if MODEL_CHECK_IMPORTS.get("risk"):
            load_risk_model()
            status["risk"] = True
    except Exception as e:
        logger.debug(f"Risk model load error: {e}")
    try:
        if MODEL_CHECK_IMPORTS.get("goal"):
            load_goal_model()
            status["goal"] = True
    except Exception as e:
        logger.debug(f"Goal model load error: {e}")
    return status


def load_prospects() -> List[Dict[str, Any]]:
    """Read prospects.csv or return empty list."""
    csv_path = Path("data/input_data/prospects.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            records = df.to_dict(orient="records")
            return records
        except Exception as e:
            logger.error(f"Failed to read prospects CSV: {e}")
    return []




async def analyze_prospect_async(workflow, prospect_data: Dict[str, Any], progress_callback=None) -> WorkflowState:
    """Asynchronously execute workflow with progress tracking."""
    prospect_state = ProspectState(prospect_data=prospect_data)
    state = WorkflowState(
        workflow_id=f"wf_{int(datetime.datetime.now().timestamp())}",
        session_id=f"sess_{int(datetime.datetime.now().timestamp())}",
        prospect=prospect_state,
        analysis=AnalysisState(),
    )
    
    state.add_agent_execution("workflow_runner", status="running")
    
    try:
        # Track progress through workflow
        # if progress_callback:
        #     progress_callback(0, "Initializing workflow...")
        
        # # Execute workflow
        # if progress_callback:
        #     progress_callback(25, "Loading prospect data...")
        
        # result = await workflow.run(state)
        
        # if progress_callback:
        #     progress_callback(100, "Analysis complete!")

        # Check if workflow supports progress tracking
        if hasattr(workflow, 'run_with_progress') and progress_callback:
            result = await workflow.run_with_progress(state, progress_callback)
        else:
            # Fallback to regular run
            result = await workflow.run(state)
        
        state.complete_agent_execution("workflow_runner", status="completed")
        return result
    except Exception as e:
        logger.exception("Workflow execution failed")
        state.complete_agent_execution("workflow_runner", status="failed", error_message=str(e))
        if progress_callback:
            progress_callback(100, f"Error: {str(e)}")
        return state


def run_analysis(workflow, prospect_data: Dict[str, Any], progress_callback=None) -> WorkflowState:
    """Runs analysis with progress tracking."""
    try:
        return asyncio.run(analyze_prospect_async(workflow, prospect_data, progress_callback))
    except RuntimeError as e:
        # import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(analyze_prospect_async(workflow, prospect_data, progress_callback))


def safe_get(obj: Any, path: str, default=None):
    """Safely retrieve nested attribute using dot notation."""
    if obj is None:
        return default
    parts = path.split(".")
    cur = obj
    for p in parts:
        try:
            if isinstance(cur, dict):
                cur = cur.get(p, default)
            else:
                cur = getattr(cur, p, default)
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def display_analysis_results(state_to_show):
    """Display formatted analysis results in tabs."""
    tabs = st.tabs([
        t("Risk Assessment"), 
        t("Goals"),
        t("Persona"),
        t("Product Recommendations"), 
        t("Data Quality"), 
        t("Agent Performance"), 
        t("Chat"),
    ])
    
    # Risk Assessment Tab
    with tabs[0]:
        st.markdown(t("### Risk Assessment Overview"))
        risk = safe_get(state_to_show, "analysis.risk_assessment", None)
        
        if risk:
            col1, col2 = st.columns(2)
            with col1:
                risk_level = risk.get("risk_level", "Unknown") if isinstance(risk, dict) else getattr(risk, "risk_level", "Unknown")
                st.markdown(t(f'<div class="metric-card"><h4>Risk Level</h4><p style="font-size: 1.2rem; color: #A3B087; font-weight: 600;">{risk_level}</p></div>'), unsafe_allow_html=True)
            
            with col2:
                confidence = risk.get("confidence", 0) if isinstance(risk, dict) else getattr(risk, "confidence_score", 0)
                st.markdown(t(f'<div class="metric-card"><h4>Confidence Score</h4><p style="font-size: 1.2rem; color: #A3B087; font-weight: 600;">{confidence:.2%}</p></div>'), unsafe_allow_html=True)
            
            st.markdown(t("#### Key Risk Factors"))
            rf = risk.get("risk_factors", []) if isinstance(risk, dict) else getattr(risk, "risk_factors", [])
            if rf:
                st.markdown('<div class="class-container>',unsafe_allow_html=True)
                for idx, factor in enumerate(rf, 1):
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <div class="card-index">{idx}</div>
                            <div class="card-test">{t(factor)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                #st.markdown("/div",unsafe_allow_html=True)
            else:
                st.info("No risk factors identified.")
            
            st.markdown(t("#### Recommendations"))
            recs = risk.get("recommendations", []) if isinstance(risk, dict) else getattr(risk, "recommendations", [])
            st.markdown('<div class="class-container">', unsafe_allow_html=True)
            if recs:
                for idx, rec in enumerate(recs, 1):
                    
                    st.markdown(f"""
                        <div class="info-card">
                            <div class="card-index">{idx}</div>
                            <div class="card-test">{t(rec)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No recommendations available.")
        else:
            st.info("No risk assessment data available.")
    # Goals Tab
    with tabs[1]:
        st.markdown(t("### Goal Prediction"))
        
        gp = safe_get(state_to_show, "analysis.goal_prediction", None)
        
        if gp:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(t(f'<div class="metric-card"><h4>Goal Feasibility</h4><p style="font-size: 1.1rem; color: #A3B087; font-weight: 600;">{gp.goal_success}</p></div>'), unsafe_allow_html=True)
            with col2:
                st.markdown(t(f'<div class="metric-card"><h4>Success Probability</h4><p style="font-size: 1.3rem; color: #A3B087; font-weight: 600;">{gp.probability:.2%}</p></div>'), unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(t("#### Success Factors"))
                for factor in gp.success_factors:
                    st.markdown(t(f'<div class="info-card">✓  {factor}</div>'), unsafe_allow_html=True)
            
            with col4:
                st.markdown(t("#### Challenges"))
                for challenge in gp.challenges:
                    st.markdown(t(f'<div class="info-card">⚠  {challenge}</div>'), unsafe_allow_html=True)
            
            # st.markdown("#### Timeline Analysis")
            # if isinstance(gp.timeline_analysis, dict):
            #     for k, v in gp.timeline_analysis.items():
            #         st.markdown(f'<div class="info-card"><strong>{k}:</strong{v}</div>', unsafe_allow_html=True)
            # else:
            #     st.markdown(f'<div class="info-card">{gp.timeline_analysis}</div>', unsafe_allow_html=True)
        else:
            st.info("No goal prediction data available.")

# Persona Tab
    with tabs[2]:
        st.markdown(t("### Persona Analysis"))
        persona = safe_get(state_to_show, "analysis.persona_classification", None)
        
        if persona:
            st.markdown("---")
            st.markdown(t("### Persona Classification"))
            st.markdown(t(
                f'<div class="metric-card"><h4>Persona Type</h4>'
                f'<p style="font-size: 1.3rem; color: #A3B087; font-weight: 600;">{persona.persona_type}</p></div>'),
                unsafe_allow_html=True
            )

            st.markdown(t("#### Behavioral Insights"))
            if persona.behavioral_insights:
                st.markdown(t('<div class="class-container">'), unsafe_allow_html=True)

                card_idx = 1
                # Ensure the content is a dict, not a string
                if isinstance(persona.behavioral_insights, str):
                    try:
                        insights_dict = json.loads(persona.behavioral_insights)
                    except json.JSONDecodeError:
                        insights_dict = {"General Insights": [persona.behavioral_insights]}
                else:
                    insights_dict = persona.behavioral_insights

                for section, insights in insights_dict.items():
                    # Safely handle non-list outputs
                    if not isinstance(insights, list):
                        insights = [str(insights)]

                    section_html = "<br>".join([f"• {i}" for i in insights])
                    st.markdown(t(
                        f"""
                        <div class="info-card">
                            <div class="card-index">{card_idx}</div>
                            <div class="card-test">
                                <div class="card-section">{section}</div>
                                <div class="card-text">{section_html}</div>
                            </div>
                        </div>
                        """),
                        unsafe_allow_html=True,
                    )

                    card_idx += 1

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No behavioral insights available.")


                

    # Products Tab
    with tabs[3]:
        st.markdown(t("### Product Recommendations"))
        # Dropdown (Selectbox)
        options = [t("Base Data"), t("Live Data")]
        selected_option = st.selectbox(t("Choose an option:"), options)
        st.write(t(f"You selected: **{selected_option}**"))

        if selected_option:

            if selected_option == t("Base Data"):

            # Display selected value
            

                products = safe_get(state_to_show, "recommendations.recommended_products", None) or []
                
                if products:
                    df_prod = pd.DataFrame([p.model_dump() if hasattr(p, "model_dump") else p for p in products])
                    for i in range(len(df_prod)):
                        # for j in range(len(df_prod.columns)):
                        df_prod.iloc[i,1] = t(df_prod.iloc[i,1])
                        df_prod.iloc[i,2] = t(df_prod.iloc[i,2])
                        df_prod.iloc[i,4] = t(df_prod.iloc[i,4])
                        df_prod.iloc[i,5] = t(df_prod.iloc[i,5])

                    # Configure column settings for better display
                    column_config = {
                        "product_name": st.column_config.TextColumn("Product Name", width="medium"),
                        "product_type": st.column_config.TextColumn("Type", width="small"),
                        "suitability_score": st.column_config.NumberColumn("Suitability", format="%.2f", width="small"),
                        "justification": st.column_config.TextColumn("Justification", width="large"),
                    }
                    
                    st.dataframe(
                        df_prod,
                        column_config=column_config,
                        width='stretch',
                        hide_index=True,
                    )

            if selected_option == t("Live Data"):
                file_path = "financial_products.csv" 
                df = pd.read_csv(file_path)
                for i in range(len(df)):
                    # for j in range(len(df_prod.columns)):
                    df.iloc[i,1] = t(df.iloc[i,1])
                    df.iloc[i,2] = t(df.iloc[i,2])
                    df.iloc[i,3] = t(df.iloc[i,3])
                    df.iloc[i,7] = t(df.iloc[i,7])
                    df.iloc[i,8] = t(df.iloc[i,8])
                st.dataframe(df)
            # st.info("No product recommendations available.")
        else:
            st.info("No product recommendations available.")

    # Data Quality Tab
    with tabs[4]:
        st.markdown(t("### Data Quality & Validation"))
        
        ps = safe_get(state_to_show, "prospect", None)
        if ps and getattr(ps, "validation_errors", None):
            st.markdown(t("#### Validation Errors"))
            for error in ps.validation_errors:
                st.markdown(t(f'<div class="info-card" style="border-left-color: #ff6b6b;">⚠ {error}</div>'), unsafe_allow_html=True)
        else:
            st.markdown(t('<div class="info-card" style="border-left-color: #A3B087;">✓ No validation errors found</div>'), unsafe_allow_html=True)
        
        dq = safe_get(state_to_show, "prospect.data_quality_score", None)
        if dq:
            st.markdown(t(f'<div class="metric-card"><h4>Data Quality Score</h4><p style="font-size: 1.8rem; color: #A3B087; font-weight: 600;">{dq:.2%}</p></div>'), unsafe_allow_html=True)

    # Agent Performance Tab
    with tabs[5]:
        display_agent_performance(state_to_show)

    # Chat Tab
    with tabs[6]:
        display_chat_interface(state_to_show)


def display_agent_performance(state: WorkflowState):
    """Display agent execution performance metrics."""
    st.markdown(t("### Agent Execution Performance"))
    
    if isinstance(state, dict):
        try:
            state = WorkflowState(**state)
        except Exception:
            st.warning("Could not reconstruct workflow state from dictionary.")
            return
    
    if not state or not state.agent_executions:
        st.info("No agent execution records available.")
        return
    
    rows = []
    for e in state.agent_executions:
        rows.append({
            "Agent Name": t(e.agent_name),
            "Status": t(e.status),
            "Start Time": getattr(e.start_time, "isoformat", lambda: str(e.start_time))(),
            "End Time": getattr(e.end_time, "isoformat", lambda: str(e.end_time))() if e.end_time else "N/A",
            "Execution Time (s)": f"{e.execution_time:.2f}" if e.execution_time else "N/A",
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, width='stretch')


def display_chat_interface(state_to_show):
    """Display enhanced chat interface with history."""
    st.markdown(t("### RM Chat Assistant"))
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    # Chat history display
    if st.session_state["chat_history"]:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state["chat_history"]:
            st.markdown(t(f'<div class="chat-message user"><strong>You:</strong><br>{msg["user"]}</div>'), unsafe_allow_html=True)
            st.markdown(t(f'<div class="chat-message assistant"><strong>Assistant:</strong><br>{msg["bot"]}</div>'), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(t("Start a conversation by asking a question below or selecting a suggested question."))
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    with col1:
        user_q = st.text_input(t("Ask a question about this prospect or analysis"), key="chat_input", label_visibility="collapsed", placeholder=t("Type your question here..."))
    with col2:
        ask_button = st.button(t("Send"), width='stretch')
    
    if ask_button and user_q:
        with st.spinner(t("Generating response...")):
            reply = generate_chat_response(user_q, state_to_show)
            st.session_state["chat_history"].append({"user": user_q, "bot": reply})
            st.rerun()
    
    # Suggested questions
    st.markdown("---")
    st.markdown(t("#### Suggested Questions"))
    suggestions = get_suggested_questions(state_to_show)
    
    cols = st.columns(len(suggestions))
    for idx, (col, question) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(question, key=f"suggest_{idx}", width='stretch'):
                with st.spinner(t("Generating response...")):
                    reply = generate_chat_response(question, state_to_show)
                    st.session_state["chat_history"].append({"user": question, "bot": reply})
                    st.rerun()
    
    # Clear chat button
    if st.session_state["chat_history"]:
        if st.button(t("Clear Chat History"), type="secondary"):
            st.session_state["chat_history"] = []
            st.rerun()


def generate_chat_response(query: str, analysis_state: WorkflowState) -> str:
    """Generate chat response using RM Assistant or Gemini fallback."""

    nest_asyncio.apply()
    if analysis_state is None:
        logger.error("No analysis_state provided to chat generator")
        return "No analysis available to answer the question"

    if not isinstance(analysis_state,WorkflowState):
        try: 
            if hasattr(analysis_state,"to_dict"):
                as_dict=analysis_state.to_dict()
            else:
                as_dict=dict(analysis_state)
            analysis_state=WorkflowState(**as_dict)
        except Exception as e: 
            logger.debug(f"Could not reconstruct WorkflowState from session object:{e}")
    try:
        if getattr(analysis_state, "chat", None) is None:
            analysis_state.chat = ChatState(
                conversation_history=[],
                current_query=query,
                context="",
                response=None
            )
        else:
            analysis_state.chat.current_query = query
    except Exception:
        try:
            analysis_state["chat"]={
                "converation_history":[],
                "current_query":query,
                "context":"",
                "response":None,
            }
        except Exception:
            pass

    if RMAssistant:
        try:
            assistant = RMAssistant()
            loop= asyncio.get_event_loop()

            handle=assistant.handle_query
            result_or_coro=handle(analysis_state,query)

            if asyncio.iscoroutine(result_or_coro):
                if loop.is_running():
                    result=loop.run_until_complete(result_or_coro)
                else:
                    result=asyncio.run(result_or_coro)
            else: 
                result=result_or_coro
            
            #return asyncio.run(assistant.handle_query(analysis_state, query))
            return result if isinstance(result,str) else str(result)
        except Exception as e:
            logger.warning(f"RMAssistant failed, using Gemini fallback: {e}")

    prospect = safe_get(analysis_state, "prospect.prospect_data", {})
    risk = safe_get(analysis_state, "analysis.risk_assessment", {})
    goal = safe_get(analysis_state, "analysis.goal_prediction", {})
    persona = safe_get(analysis_state, "analysis.persona_classification", {})

    context_summary = {
        "Prospect Overview": prospect,
        "Risk Assessment": {
            "Level": getattr(risk, "risk_level", None),
            "Confidence": getattr(risk, "confidence_score", None),
            "Factors": getattr(risk, "risk_factors", []),
        },
        "Goal Planning": {
            "Feasibility": getattr(goal, "goal_success", None),
            "Probability": getattr(goal, "probability", None),
            "Success Factors": getattr(goal, "success_factors", []),
            "Challenges": getattr(goal, "challenges", []),
        },
        "Persona": {
            "Type": getattr(persona, "persona_type", None),
            "Insights": getattr(persona, "behavioral_insights", []),
        },
    }

    prompt = (
        "You are an intelligent Relationship Manager assistant. "
        "Use the context below to answer the RM's question in a clear and concise way.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{json.dumps(context_summary, default=str, indent=2)}\n\n"
        "Respond in plain, professional English. Avoid repetition and unnecessary detail."
    )

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        loop=asyncio.get_event_loop()
        response=loop.run_until_complete(model.generate_content_async(prompt))

        if hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "content"):
            return response.content.strip()
        elif isinstance(response, dict):
            candidates = response.get("candidates", [])
            if candidates and "content" in candidates[0]:
                return candidates[0]["content"]
        return "Sorry, I couldn't generate a proper response."
    except Exception as e:
        logger.error(f"Gemini chat failed: {e}")
        return generate_fallback_response(query, analysis_state)


def generate_fallback_response(query: str, analysis_state: WorkflowState) -> str:
    """Rule-based fallback for chat."""
    q = query.lower()
    if "risk" in q:
        risk = safe_get(analysis_state, "analysis.risk_assessment.risk_level", "Unknown")
        conf = safe_get(analysis_state, "analysis.risk_assessment.confidence_score", None)
        return f"Risk level: {risk}. Confidence: {conf}" if conf else f"Risk level: {risk}."
    if "goal" in q or "feasible" in q:
        gp = safe_get(analysis_state, "analysis.goal_prediction.goal_success", "Unknown")
        prob = safe_get(analysis_state, "analysis.goal_prediction.probability", None)
        return f"Goal feasibility: {gp}. Probability: {prob}" if prob else f"Goal feasibility: {gp}."
    return "Sorry — I couldn't find a specific answer from the current analysis."


def get_suggested_questions(analysis_state: WorkflowState) -> List[str]:
    """Build suggested follow-up questions."""
    suggestions = []
    risk = safe_get(analysis_state, "analysis.risk_assessment.risk_level", None)
    goal = safe_get(analysis_state, "analysis.goal_prediction.goal_success", None)
    persona = safe_get(analysis_state, "analysis.persona_classification.persona_type", None)

    if risk:
        suggestions.append(t(f"What drives the {risk} risk?"))
    if goal:
        suggestions.append(t("How to improve goal success?"))
    if persona:
        suggestions.append(t(f"Best approach for {persona}?"))
    
    suggestions.append(t("What are the next steps?"))
    return suggestions[:4]  # Limit to 4 suggestions


# ---------------------------
# MAIN UI LAYOUT
# ---------------------------

# --- MAIN TITLE ---
st.markdown(t('<h1>Multi-Agent Financial AI System</h1>'), unsafe_allow_html=True)
st.markdown(t('<p class="subtitle">AI-Powered Intelligent dashboard for financial assessments and investment planning.</p>'), unsafe_allow_html=True)



# --- SESSION STATE ---
if "run_requested" not in st.session_state:
    st.session_state["run_requested"] = False
if "analysis_completed" not in st.session_state:
    st.session_state["analysis_completed"] = False

# --- SIDEBAR ---
# st.sidebar.markdown('<div class="sidebar-heading">Configuration</div>', unsafe_allow_html=True)
st.sidebar.markdown(t(("Configuration").upper()))

with st.sidebar.expander(t("Model Status & Setup"), expanded=True):
    model_ok = ensure_models_trained()
    status = check_model_status()

    status_color = "#A3B087" if model_ok else "#ff6b6b"
    st.markdown(t(f"**Model Artifacts Present:** <span style='color: {status_color}; font-weight: 600;'>{'✓ Yes' if model_ok else '✗ No'}</span>"), unsafe_allow_html=True)

    if isinstance(status, dict):
        st.markdown(t("**Model Load Status:**"))
        rows = []
        for model_name, is_ready in status.items():
            status_icon = "✓" if is_ready else "✗"
            status_text = t("Loaded") if is_ready else "Missing"
            rows.append({
                t("Model"): t(model_name.replace("_", " ").title()),
                t("Status"): f"{status_icon} {status_text}"
            })
        st.table(rows)
    else:
        try:
            parsed = json.loads(status)
            st.json(parsed)
        except Exception:
            st.write(status)

    if not model_ok:
        st.warning("Some models are missing. Fallback methods will be used.")

st.sidebar.markdown("---")

# --- MAIN CONTENT ---
st.markdown(t("## Prospect Selection"))

prospects = load_prospects()
if not prospects:
    st.warning("No prospects found in the dataset.")
else:
    names = [f"{p.get('prospect_id', '-')} | {t(p.get('name', '-'))}" for p in prospects]
    sel = st.selectbox("Select a Prospect", options=names, help="Choose a prospect to analyze")
    selected_idx = names.index(sel) if sel in names else 0
    selected_prospect = prospects[selected_idx]

    with st.expander(t("View Prospect Details"), expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(t("#### Basic Information"))
            st.write(t(f"**Prospect ID:** `{selected_prospect.get('prospect_id', 'N/A')}`"))
            st.write(t(f"**Name:** {selected_prospect.get('name', 'N/A')}"))
            st.write(t(f"**Age:** {selected_prospect.get('age', 'N/A')} years"))
            st.write(t(f"**Annual Income:** ₹{selected_prospect.get('annual_income', 0):,}"))
            st.write(t(f"**Current Savings:** ₹{selected_prospect.get('current_savings', 0):,}"))

        with col2:
            st.markdown(t("#### Investment Profile"))
            st.write(t(f"**Target Goal Amount:** ₹{selected_prospect.get('target_goal_amount', 0):,}"))
            st.write(t(f"**Investment Horizon:** {selected_prospect.get('investment_horizon_years', 'N/A')} years"))
            st.write(t(f"**Number of Dependents:** {selected_prospect.get('number_of_dependents', 'N/A')}"))
            st.write(t(f"**Experience Level:** {selected_prospect.get('investment_experience_level', 'N/A')}"))
            st.write(t(f"**Investment Goal:** {selected_prospect.get('investment_goal', 'N/A')}"))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Left-aligned analyze button
    analyze_button = st.button(t("Analyze Prospect"))
    if analyze_button:
        st.session_state["run_requested"] = True
        st.session_state["analysis_completed"] = False
        st.session_state["selected_prospect"] = selected_prospect

st.markdown("---")

workflow = get_workflow()

# Progress tracking with actual workflow nodes
if st.session_state.get("run_requested", False) and not st.session_state.get("analysis_completed", False):
    #st.info("Starting analysis — tracking workflow execution in real-time...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a callback to update progress
    def update_progress(percent, message):
        # progress_bar.progress(percent)
        progress_bar.progress(percent / 100)  # Streamlit expects 0.0-1.0 range
        status_text.text(message)
    
    # # Track actual workflow execution
    # class ProgressTracker:
    #     def __init__(self):
    #         self.current_step = 0
    #         self.steps = [
    #             (0, "Initializing workflow..."),
    #             (15, "Validating prospect data..."),
    #             (30, "Executing risk assessment agent..."),
    #             (50, "Running goal planning agent..."),
    #             (70, "Classifying persona..."),
    #             (85, "Generating recommendations..."),
    #             (100, "Finalizing analysis...")
    #         ]
        
    #     def update(self, step_name=None):
    #         if self.current_step < len(self.steps):
    #             percent, message = self.steps[self.current_step]
    #             if step_name:
    #                 message = f"{step_name}..."
    #             update_progress(percent, message)
    #             self.current_step += 1
    
    # tracker = ProgressTracker()
    
    # import time
    # # Run analysis with progress tracking
    # for step_percent, step_message in tracker.steps:
    #     update_progress(step_percent, step_message)
    #     time.sleep(0.2)
    
    state_result = run_analysis(workflow, st.session_state["selected_prospect"], update_progress)
    
    # progress_bar.progress(100)
    # Clean up progress UI
    progress_bar.progress(1.0)
    status_text.empty()
    
    # st.success(" Analysis complete!")
    st.success("✓ Analysis complete!")
    st.session_state["latest_state"] = state_result
    st.session_state["analysis_completed"] = True
    st.session_state["run_requested"] = False

    import time
    time.sleep(0.5)
    st.rerun()

st.markdown(t("## Analysis Results"))
state_to_show: Optional[WorkflowState] = st.session_state.get("latest_state", None)
if not state_to_show:
    st.info(t("No analysis available. Click 'Analyze Prospect' above to run."))
else:
    display_analysis_results(state_to_show=state_to_show)

