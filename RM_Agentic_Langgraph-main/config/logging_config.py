"""Logging configuration for the application."""

import sys
from loguru import logger
from typing import Optional

from .settings import get_settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """Setup application logging with loguru."""
    settings = get_settings()
    level = log_level or settings.log_level
    
    # Remove default handler
    logger.remove()
    
    # Console handler with custom format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler for persistent logging
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    logger.add(
        "logs/app.log",
        format=file_format,
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # Agent-specific log file
    logger.add(
        "logs/agents.log",
        format=file_format,
        level=level,
        rotation="5 MB",
        retention="7 days",
        filter=lambda record: "agent" in record["name"].lower(),
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging initialized with level: {level}")


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)