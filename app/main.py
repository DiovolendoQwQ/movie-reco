# Main entry point for the FastAPI application.
# This file imports the app instance from api.py, allowing Uvicorn to find it.

import logging
from .api import app # Import the FastAPI app instance from api.py
from .utils import get_logger

# Configure logging basic settings if not already done elsewhere
# This ensures logs are captured when running with Uvicorn
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

logger = get_logger(__name__)

# You can add any application-wide startup logic here if needed,
# although @app.on_event("startup") in api.py is often preferred.

logger.info("FastAPI application instance created in main.py")

# Uvicorn will typically be pointed to this file like:
# uvicorn app.main:app --reload
