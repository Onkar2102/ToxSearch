"""
custom_logging.py
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import platform
import getpass
import datetime
import json
import sys
import traceback
from typing import Optional

# Global variable to store the current log file path
_CURRENT_LOG_FILE = None

def get_run_id():
    """Generate a unique run ID for the current day"""
    log_index_file = "logs/log_index.json"
    today = datetime.datetime.now().strftime("%Y%m%d")

    if not os.path.exists("logs"):
        os.makedirs("logs")

    if os.path.exists(log_index_file):
        try:
            with open(log_index_file, "r") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            log_data = {}
    else:
        log_data = {}

    today_run_id = log_data.get(today, 0) + 1
    log_data[today] = today_run_id

    with open(log_index_file, "w") as f:
        json.dump(log_data, f, indent=2)

    return today_run_id

def get_log_filename():
    """Generate a detailed log filename with timestamp and system info"""
    global _CURRENT_LOG_FILE
    
    # If we already have a log file for this run, return it
    if _CURRENT_LOG_FILE is not None:
        return _CURRENT_LOG_FILE
    
    # Create a new log file only once per run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = get_run_id()
    user = getpass.getuser()
    machine = platform.node().split('.')[0]
    device_info = f"{user}@{machine}".replace(" ", "_")
    
    # Add Python version and platform info
    python_version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    platform_info = platform.system().lower()

    _CURRENT_LOG_FILE = f"logs/{timestamp}_run{run_id}_{device_info}_{python_version}_{platform_info}.log"
    return _CURRENT_LOG_FILE

def get_detailed_formatter() -> logging.Formatter:
    """Create a detailed formatter with additional context"""
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)-20s] [%(filename)s:%(lineno)d] [%(funcName)s()]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S.%f"[:-3]  # Include milliseconds
    )

def get_simple_formatter() -> logging.Formatter:
    """Create a simple formatter for console output"""
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S"
    )

def setup_exception_logging(logger: logging.Logger):
    """Setup exception logging to capture unhandled exceptions"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default handler for keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception

def get_logger(name: str = "default_logger", log_file: Optional[str] = None) -> logging.Logger:
    """Get a configured logger with detailed formatting and exception handling"""
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger
    
    # Get log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, log_level)
    except AttributeError:
        log_level = logging.INFO
    
    # Use provided log_file, or current log file, or generate new one
    if log_file is None:
        log_file = _CURRENT_LOG_FILE if _CURRENT_LOG_FILE is not None else get_log_filename()
    
    # Create detailed formatter for file logging
    detailed_formatter = get_detailed_formatter()
    simple_formatter = get_simple_formatter()
    
    # Console handler with simple formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)  # Console shows INFO and above
    logger.addHandler(console_handler)
    
    # File handler with detailed formatting
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=100_000_000,  # 100MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)  # File shows all levels
    logger.addHandler(file_handler)
    
    # Set logger level
    logger.setLevel(logging.DEBUG)
    
    # Setup exception logging
    setup_exception_logging(logger)
    
    # Log initialization
    logger.debug("Logger '%s' initialized with file: %s", name, log_file)
    logger.info("Log level set to: %s", logging.getLevelName(log_level))
    
    return logger

def log_system_info(logger: logging.Logger):
    """Log system information for debugging"""
    logger.info("=== System Information ===")
    logger.info("Platform: %s", platform.platform())
    logger.info("Python Version: %s", sys.version)
    logger.info("Machine: %s", platform.machine())
    logger.info("Processor: %s", platform.processor())
    logger.info("User: %s", getpass.getuser())
    logger.info("Working Directory: %s", os.getcwd())
    logger.info("Log File: %s", get_log_filename())
    
    # Log environment variables (excluding sensitive ones)
    sensitive_vars = {'OPENAI_API_KEY', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID', 'PASSWORD', 'SECRET'}
    env_vars = {k: v for k, v in os.environ.items() 
               if k.startswith(('OPENAI_', 'PYTHON_', 'PATH', 'HOME', 'USER')) 
               and k not in sensitive_vars}
    logger.debug("Environment Variables: %s", env_vars)

def log_performance_metrics(logger: logging.Logger, operation: str, start_time: float, 
                          end_time: Optional[float] = None, **kwargs):
    """Log performance metrics for operations"""
    if end_time is None:
        end_time = datetime.datetime.now().timestamp()
    
    duration = end_time - start_time
    logger.info("PERFORMANCE: %s completed in %.3f seconds", operation, duration)
    
    if kwargs:
        logger.debug("PERFORMANCE_DETAILS: %s - %s", operation, kwargs)

class PerformanceLogger:
    """Context manager for logging operation performance"""
    
    def __init__(self, logger: logging.Logger, operation: str, **kwargs):
        self.logger = logger
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.datetime.now().timestamp()
        self.logger.info("STARTING: %s", self.operation)
        if self.kwargs:
            self.logger.debug("PARAMETERS: %s - %s", self.operation, self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.datetime.now().timestamp()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info("COMPLETED: %s in %.3f seconds", self.operation, duration)
        else:
            self.logger.error("FAILED: %s after %.3f seconds - %s: %s", 
                            self.operation, duration, exc_type.__name__, str(exc_val))
            self.logger.debug("Exception traceback: %s", traceback.format_exc())