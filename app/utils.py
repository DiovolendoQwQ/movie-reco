# Utility functions for the FastAPI application
# (e.g., logging setup, helper functions)

import logging

# Basic logging configuration can go here if needed globally
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_logger(name: str):
    """Gets a logger instance."""
    return logging.getLogger(name)
