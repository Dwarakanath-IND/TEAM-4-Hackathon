# TODO: Import logger from loguru, sys, and Optional from typing
from loguru import logger
import sys
from typing import Optional

# TODO: Import get_settings from .settings
from .settings import get_settings

# TODO: Create setup_logging function that accepts optional log_level parameter
def setup_logging(log_level: Optional[str] = None):

    # TODO: Get settings and determine log level (from param or settings.log_level)
    settings = get_settings()
    level = log_level or settings.log_level

    # TODO: Remove default logger handler
    logger.remove()

    # TODO: Add stdout console handler with colored format (time, level, name, function, line, message)
    console_format = (
        "<green>{time:YYYY-MM-DD HH-mm-ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format = console_format,
        level = level,
        colorize = True,
        backtrace = True,
        diagnose = True
    )
    # File format
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # TODO: Add file handler to logs/app.log with rotation (10MB) and retention (30 days)
    logger.add(
        "logs/app.log",
        format = file_format,
        rotation = "10 MB",
        retention = "30 days",
        level = level,
        compression = "zip",
        enqueue = True,
        backtrace = True,
        diagnose = True
    )

    # TODO: Add agent-specific file handler to logs/agents.log with rotation (5MB) and retention (7 days)
    logger.add(
        "logs/agents.log",
        format = file_format,
        rotation = "5 MB",
        retention = "7 days", 
        level = level,
        filter = lambda record: "agent" in record["name"].lower(),
        enqueue = True,
        backtrace = True,
        diagnose = True
    )

# TODO: Create get_logger function that binds logger with module name
def get_logger(module_name: str):
    return logger.bind(module = module_name)
