import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import random
import logging
from .utils import get_logger

logger = get_logger(__name__)

# --- Configuration ---
MODEL_DIR = Path("app/models")
SIMILARITY_FILE = MODEL_DIR / "sim.parquet"
GENRE_MAP_FILE = MODEL_DIR / "genre_map.parquet"
TOP_K_SIMILAR = 50 # Must match the K used in 03_compute_sim.py
RECOMMENDATION_COUNT = 20 # Default number of recommendations to return

# --- Load Data ---
try:
    logger.info(f"Loading similarity data from {SIMILARITY_FILE}...")
    sim_df = pl.read_parquet(SIMILARITY_FILE)
    logger.info("Similarity data loaded.")

    logger.info(f"Loading genre map from {GENRE_MAP_FILE}...")
    genre_map_df = pl.read_parquet(GENRE_MAP_FILE)
    logger.info("Genre map loaded.")

    # --- Precompute Lookups for Efficiency ---
    logger.info("Preprocessing data for faster lookups...")

    # 1. Movie ID to internal index mapping (and vice-versa)
    movie_id_to_idx = dict(zip(genre_map_df['movieId'], genre_map_df['item_idx']))
    idx_to_movie_id = dict(zip(genre_map_df['item_idx'], genre_map_df['movieId']))
    idx_to_title = dict(zip(genre_map_df['item_idx'], genre_map_df['title']))

    # 2. Genre to list of item indices mapping
    genre_to_indices: Dict[str, List[int]] = {}
    all_genres: Set[str] = set()
    for row in genre_map_df.iter_rows(named=True):
        genres = row['genres'].split('|')
        item_idx = row['item_idx']
        for genre in genres:
            if genre == "(no genres listed)": continue # Skip this category
            all_genres.add(genre)
            if genre not in genre_to_indices:
                genre_to_indices[genre] = []
            genre_to_indices[genre].append(item_idx)

    # 3. Similarity lookup (item_idx_from -> List[(item_idx_to, similarity)])
    # Group by 'item_idx_from' and aggregate 'item_idx_to' and 'similarity'
    similarity_lookup: Dict[int, List[Tuple[int, float]]] = {}
    # Sort by similarity descending within each group for faster top-k retrieval later
    sim_df_sorted = sim_df.sort("item_idx_from", "similarity", descending=[False, True])
    for group_key, group_df in sim_df_sorted.group_by("item_idx_from"):
         # group_key is a tuple, take the first element
        from_idx = group_key[0]
        # Convert group to list of tuples
        similar_items = list(zip(group_df['item_idx_to'], group_df['similarity']))
        similarity_lookup[from_idx] = similar_items

    logger.info("Preprocessing finished.")

except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}. Ensure sim.parquet and genre_map.parquet are in app/models/")
    # Set data to None or empty structures to indicate failure
    sim_df = None
    genre_map_df = None
    movie_id_to_idx = {}
    idx_to_movie_id = {}
    idx_to_title = {}
    genre_to_indices = {}
    all_genres = set()
    similarity_lookup = {}
except Exception as e:
    logger.error(f"An error occurred during model loading or preprocessing: {e}")
    # Set data to None or empty structures
    sim_df = None
    genre_map_df = None
    movie_id_to_idx = {}
    idx_to_movie_id = {}
    idx_to_title = {}
    genre_to_indices = {}
    all_genres = set()
    similarity_lookup = {}

# --- Recommendation Functions ---

def get_all_genres() -> List[str]:
    """Returns a sorted list of all available movie genres."""
    if not all_genres:
        logger.warning("Genre list is empty. Models might not have loaded correctly.")
    return sorted(list(all_genres))

def _get_movie_details(item_indices: List[int]) -> List[Dict]:
    """Helper to get movie details (id, title) from internal indices."""
    details = []
    for idx in item_indices:
        movie_id = idx_to_movie_id.get(idx)
        title = idx_to_title.get(idx)
        if movie_id and title:
            details.append({"movieId": movie_id, "title": title})
        else:
             logger.warning(f"Could not find details for item index: {idx}")
    return details


def random_by_genre(genre: str, n: int = RECOMMENDATION_COUNT) -> List[Dict]:
    """Returns N random movies from the specified genre."""
    if not genre_to_indices:
         logger.error("Genre to indices mapping not available. Cannot provide random recommendations.")
         return []
    if genre not in genre_to_indices:
        logger.warning(f"Genre '{genre}' not found.")
        return []

    indices_in_genre = genre_to_indices[genre]
    if len(indices_in_genre) <= n:
        # If fewer movies than requested, return all of them shuffled
        random.shuffle(indices_in_genre)
        selected_indices = indices_in_genre
    else:
        selected_indices = random.sample(indices_in_genre, n)

    logger.info(f"Returning {len(selected_indices)} random movies for genre '{genre}'.")
    return _get_movie_details(selected_indices)


def recommend_top_k(history_movie_ids: List[int], n: int = RECOMMENDATION_COUNT) -> List[Dict]:
    """
    Recommends N movies based on the user's recent history (list of movie IDs).
    Uses item-to-item similarity.
    """
    if not similarity_lookup or not movie_id_to_idx:
        logger.error("Similarity lookup or ID mapping not available. Cannot provide recommendations.")
        return []
    if not history_movie_ids:
        logger.warning("Recommendation history is empty.")
        return [] # Or return popular items? For now, return empty.

    # Convert history movie IDs to internal indices
    history_indices = [movie_id_to_idx.get(mid) for mid in history_movie_ids if movie_id_to_idx.get(mid) is not None]
    if not history_indices:
        logger.warning(f"None of the history movie IDs {history_movie_ids} found in the model.")
        return []

    # Aggregate similar items based on history
    candidate_scores: Dict[int, float] = {}
    history_set = set(history_indices) # For quick filtering later

    logger.debug(f"Generating recommendations based on history indices: {history_indices}")

    # Iterate through each item in the user's history
    for idx_from in history_indices:
        # Get precomputed similar items for this history item
        similar_items = similarity_lookup.get(idx_from, [])

        # Add scores of similar items to candidates
        for idx_to, score in similar_items:
            # Ignore items already in the user's history
            if idx_to not in history_set:
                candidate_scores[idx_to] = candidate_scores.get(idx_to, 0.0) + score

    # Sort candidates by aggregated score (descending)
    sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)

    # Get top N recommendations
    recommended_indices = [idx for idx, score in sorted_candidates[:n]]

    logger.info(f"Returning {len(recommended_indices)} recommendations based on history {history_movie_ids}.")
    return _get_movie_details(recommended_indices)
