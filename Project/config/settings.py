# TODO: Import os, lru_cache, Optional, Path from pathlib
import os
from functools import lru_cache
from typing import Optional
from pathlib import Path

# TODO: Import Field from pydantic
from pydantic import Field

# TODO: Import load_dotenv from dotenv
from dotenv import load_dotenv

# TODO: Load .env file from parent directory or current working directory
load_dotenv(dotenv_path = Path.cwd() / ".env")
load_dotenv(dotenv_path = Path.cwd().parent / ".env")

# TODO: Try to import BaseSettings and SettingsConfigDict from pydantic_settings
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

# TODO: Fallback to pydantic if pydantic_settings not available
except ImportError:
    from pydantic import BaseSettings
    SettingsConfigDict = None

# TODO: Create Settings class inheriting from BaseSettings with field definitions:
class Settings(BaseSettings):

    #   - API Keys: gemini_api_key, langchain_api_key
    gemini_api_key: Optional[str] = Field(None, env = "GEMINI_API_KEY_1") 
    langchain_api_key: Optional[str] = Field(None, env = "LANGCHAIN_API_KEY")

    #   - LangSmith: langchain_tracing_v2 (True), langchain_project (rm-agentic-ai)
    langchain_tracing_v2: bool = Field(True, env = "LANGCHAIN_TRACING_V2")
    langchain_project : str = Field("rm-agentic-ai", env = "LANGCHAIN_PROJECT")

    #   - App: log_level (INFO), enable_monitoring (True), debug_mode (False)
    log_level: str = Field("INFO", env = "LOG_LEVEL")
    enable_monitoring: bool = Field(True, env = "ENABLE_MONITORING")
    debug_mode: bool = Field(False, env = "DEBUG_MODE")

    #   - Performance: max_concurrent_agents (5), agent_timeout (300), cache_ttl (3600)
    max_concurrent_agents: int = Field(5, env = "MAX_CONCURRENT_AGENTS")
    agent_timeout: int = Field(300, env = "AGENT_TIMEOUT")
    cache_ttl: int = Field(3600, env = "CACHE_TTL")

    #   - Paths: data_dir, models_dir, output_dir
    data_dir: str = Field("data", env = "DATA_DIR") 
    models_dir: str = Field("ml/models", env = "MODELS_DIR")
    output_dir: str = Field("output", env = "OUTPUT_DIR")

    #   - Model paths: risk_model_path, goal_model_path, encoders paths
    risk_model_path: str = Field("ml/models/risk/risk_profile_model.pkl", env = "RISK_MODEL_PATH")
    goal_model_path: str = Field("ml/models/goal/goal_success_model.pkl", env = "GOAL_MODEL_PATH")
    risk_encoders_path: str = Field("ml/models/risk/label_encoders.pkl", env = "RISK_ENCODERS_PATH")
    goal_encoders_path: str = Field("ml/models/goal/goal_success_label_encoders.pkl", env = "GOAL_ENCODERS_PATH")

    #   - Data files: prospects_csv, products_csv paths
    prospects_csv: str = Field("data/input_data/prospects.csv", env = "PROSPECTS_CSV")
    products_csv: str = Field("data/input_data/products.csv", env = "PRODUCTS_CSV")

    #   - Streamlit: page_title, page_icon, layout
    page_title: str = Field("AI-Powered Investment Analyzer")
    page_icon: str = Field("📈")
    layout: str = Field("wide")

    #   - Agent: default_temperature (0.1), max_tokens (4000)
    default_temperature: float = Field(0.1, env = "DEFAULT_TEMPERATURE")
    max_tokens: int = Field(4000, env = "MAX_TOKENS")

    # database_url: Optional[str] = Field(None, env="DATABASE_URL")
    # secret_key: Optional[str] = Field(None, env="SECRET_KEY")

    if SettingsConfigDict:
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore" )


# TODO: Create get_settings function returning new Settings instance
def get_settings() -> Settings:
    return Settings()

# TODO: Create get_cached_settings with global caching for repeated calls
@lru_cache                                          # least recently used
def get_cached_settings() -> Settings:
    return get_settings()

# TODO: Create global settings instance for backward compatibility
settings = get_cached_settings()
