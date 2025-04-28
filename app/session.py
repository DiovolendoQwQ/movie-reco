import redis
import json
import os
from typing import List, Optional
import logging
from .utils import get_logger

logger = get_logger(__name__)

# --- Redis Configuration ---
# Get Redis connection details from environment variables or use defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", 0))

SESSION_PREFIX = "session:"
MAX_HISTORY_LENGTH = 5 # Keep track of the last 5 movie choices

# --- Initialize Redis Client ---
try:
    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=True # Decode responses from bytes to strings
    )
    redis_client.ping() # Check connection
    logger.info(f"Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}: {e}")
    # Depending on the application's needs, you might exit or handle this differently
    redis_client = None # Set to None to indicate connection failure

def _get_session_key(session_id: str) -> str:
    """Constructs the Redis key for a given session ID."""
    return f"{SESSION_PREFIX}{session_id}"

def get_mem(session_id: str) -> List[int]:
    """
    Retrieves the recent movie choice history (list of movie IDs) for a session.
    Returns an empty list if the session has no history or Redis is unavailable.
    """
    if not redis_client:
        logger.warning("Redis client not available. Cannot get session memory.")
        return []

    key = _get_session_key(session_id)
    try:
        # LRANGE returns list of strings. Convert them to integers.
        history_str = redis_client.lrange(key, 0, -1)
        history_int = [int(movie_id) for movie_id in history_str]
        logger.debug(f"Retrieved history for session {session_id}: {history_int}")
        return history_int
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error getting history for session {session_id}: {e}")
        return []
    except ValueError as e:
        logger.error(f"Error converting history item to int for session {session_id}: {e}")
        # Consider clearing the list or handling corrupted data
        return [] # Return empty list on conversion error

def push_choice(session_id: str, movie_id: int):
    """
    Adds a movie choice to the session's history and trims the history
    to the maximum allowed length (MAX_HISTORY_LENGTH).
    """
    if not redis_client:
        logger.warning("Redis client not available. Cannot push choice to session.")
        return

    key = _get_session_key(session_id)
    try:
        # Add the new movie ID to the beginning of the list
        redis_client.lpush(key, str(movie_id))
        # Trim the list to keep only the most recent MAX_HISTORY_LENGTH items
        redis_client.ltrim(key, 0, MAX_HISTORY_LENGTH - 1)
        logger.debug(f"Pushed movie ID {movie_id} to session {session_id}. History trimmed.")
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error pushing choice for session {session_id}: {e}")

def clear_session(session_id: str):
    """Clears the history for a given session."""
    if not redis_client:
        logger.warning("Redis client not available. Cannot clear session.")
        return

    key = _get_session_key(session_id)
    try:
        redis_client.delete(key)
        logger.info(f"Cleared history for session {session_id}.")
    except redis.exceptions.RedisError as e:
        logger.error(f"Redis error clearing session {session_id}: {e}")
