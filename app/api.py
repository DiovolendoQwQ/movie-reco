from fastapi import FastAPI, HTTPException, Request, Depends, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
import logging

# Import recommendation and session logic from other modules
from . import recommend
from . import session
from .utils import get_logger

logger = get_logger(__name__)

# --- Pydantic Models for Request/Response ---
class Movie(BaseModel):
    movieId: int
    title: str

class GenreListResponse(BaseModel):
    genres: List[str]

class RecommendationResponse(BaseModel):
    recommendations: List[Movie]

class ChoiceRequest(BaseModel):
    movieId: int

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Movie Recommender API",
    description="API for movie recommendations based on Item-to-Item Collaborative Filtering.",
    version="0.1.0"
)

# --- CORS Configuration ---
# Allow requests from any origin for simplicity in this example.
# For production, restrict this to your frontend's domain.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Session ID Management ---
# Use a cookie to manage the session ID
SESSION_COOKIE_NAME = "reco_session_id"

async def get_session_id(
    request: Request,
    response: Response,
    session_id: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)
) -> str:
    """
    Retrieves session ID from cookie or creates a new one if not present.
    Sets the cookie in the response.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
        logger.info(f"New session created: {session_id}")
        # Set the cookie in the response. Max_age=None means it's a session cookie.
        response.set_cookie(key=SESSION_COOKIE_NAME, value=session_id, httponly=True, samesite='lax') # httponly for security
    else:
        logger.debug(f"Existing session found: {session_id}")
    return session_id

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    # Log model loading status on startup
    # Check if the data structures were initialized correctly in recommend.py
    # We set them to empty dict/None on loading failure there.
    models_loaded = bool(recommend.similarity_lookup) and (recommend.genre_map_df is not None)

    if not models_loaded:
        logger.critical("Models did not load correctly on startup. API might not function.")
    else:
        logger.info("API started. Models loaded successfully.")
    if not session.redis_client:
         logger.warning("Redis connection failed on startup. Session features will not work.")

@app.get("/genres", response_model=GenreListResponse)
async def get_genres():
    """Returns a list of all available movie genres."""
    logger.info("Request received for /genres")
    genres = recommend.get_all_genres()
    if not genres:
         # This might happen if model loading failed
         logger.error("Genre list is empty, likely due to model loading issues.")
         raise HTTPException(status_code=500, detail="Failed to load genre data.")
    return {"genres": genres}

@app.get("/random", response_model=RecommendationResponse)
async def get_random_movies(genre: str):
    """Returns 20 random movies for a given genre."""
    logger.info(f"Request received for /random?genre={genre}")
    if not genre:
        raise HTTPException(status_code=400, detail="Genre parameter is required.")

    random_movies = recommend.random_by_genre(genre, n=recommend.RECOMMENDATION_COUNT)
    if not random_movies and genre not in recommend.get_all_genres():
         raise HTTPException(status_code=404, detail=f"Genre '{genre}' not found.")
    elif not random_movies:
         # Genre exists but no movies found (unlikely but possible) or model loading issue
         logger.warning(f"No random movies returned for genre '{genre}'.")
         # Return empty list instead of error? Or 500 if models didn't load?
         # Let's return empty list for now if genre exists but no movies found.
         pass

    return {"recommendations": random_movies}

@app.post("/choice", response_model=RecommendationResponse)
async def post_choice(
    choice: ChoiceRequest,
    session_id: str = Depends(get_session_id)
):
    """
    Receives a user's movie choice, stores it in the session,
    and returns recommendations based on the updated session history.
    """
    logger.info(f"Request received for /choice (Session: {session_id}, MovieID: {choice.movieId})")

    # Validate movie ID exists in our model data
    if choice.movieId not in recommend.movie_id_to_idx:
         logger.warning(f"Movie ID {choice.movieId} not found in model data.")
         raise HTTPException(status_code=404, detail=f"Movie ID {choice.movieId} not found.")

    # Push the choice to the session history (handled by session.py)
    session.push_choice(session_id, choice.movieId)

    # Get the updated history
    history_ids = session.get_mem(session_id)

    # Get recommendations based on history
    recommendations = recommend.recommend_top_k(history_ids, n=recommend.RECOMMENDATION_COUNT)

    return {"recommendations": recommendations}

@app.post("/reset")
async def reset_session(session_id: str = Depends(get_session_id)):
    """Clears the recommendation history for the current session."""
    logger.info(f"Request received for /reset (Session: {session_id})")
    session.clear_session(session_id)
    return {"message": "Session history cleared successfully."}

# --- Main entry point for Uvicorn (if running this file directly) ---
# This part is usually handled by app/main.py
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
