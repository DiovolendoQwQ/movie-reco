import pytest
import sys
import os
from pathlib import Path

# Add the app directory to the Python path to allow imports like 'from app import recommend'
# This assumes tests are run from the project root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import the module to test
# It will load models upon import, ensure models exist in app/models
try:
    from app import recommend
except ImportError as e:
    pytest.fail(f"Failed to import recommend module, check PYTHONPATH or module structure: {e}")
except Exception as e:
     pytest.fail(f"Failed during recommend module import (model loading?): {e}")


# --- Fixtures (Optional, can add setup/teardown if needed) ---

# --- Test Cases ---

def test_get_all_genres():
    """Tests if get_all_genres returns a non-empty list of strings."""
    genres = recommend.get_all_genres()
    assert isinstance(genres, list)
    assert len(genres) > 0
    assert all(isinstance(g, str) for g in genres)
    # Check if common genres exist (adjust based on dataset)
    assert "Action" in genres
    assert "Drama" in genres
    assert "Comedy" in genres

def test_random_by_genre_valid():
    """Tests random recommendations for a valid genre."""
    genre = "Action" # Assume Action exists
    n = 10
    recommendations = recommend.random_by_genre(genre, n=n)
    assert isinstance(recommendations, list)
    # It might return fewer if the genre has less than n movies
    assert len(recommendations) <= n
    if recommendations: # Only check structure if list is not empty
        assert all(isinstance(m, dict) for m in recommendations)
        assert all("movieId" in m and "title" in m for m in recommendations)
        assert all(isinstance(m["movieId"], int) for m in recommendations)
        assert all(isinstance(m["title"], str) for m in recommendations)

def test_random_by_genre_invalid():
    """Tests random recommendations for an invalid genre."""
    genre = "NonExistentGenre123"
    recommendations = recommend.random_by_genre(genre)
    assert isinstance(recommendations, list)
    assert len(recommendations) == 0

def test_recommend_top_k_single_item():
    """Tests top-k recommendations based on a single known movie ID."""
    # Find a valid movie ID from the loaded data
    if not recommend.movie_id_to_idx:
        pytest.skip("Movie ID mapping not loaded, skipping test.")

    # Get an arbitrary valid movie ID (e.g., the first one)
    test_movie_id = next(iter(recommend.movie_id_to_idx.keys()), None)
    if test_movie_id is None:
         pytest.skip("No movie IDs available in mapping, skipping test.")

    history = [test_movie_id]
    n = 10
    recommendations = recommend.recommend_top_k(history, n=n)

    assert isinstance(recommendations, list)
    assert len(recommendations) <= n # Might return fewer if not enough similar items found

    if recommendations:
        assert all(isinstance(m, dict) for m in recommendations)
        assert all("movieId" in m and "title" in m for m in recommendations)
        # Ensure the original movie is not recommended back
        assert test_movie_id not in [m["movieId"] for m in recommendations]

def test_recommend_top_k_multiple_items():
    """Tests top-k recommendations based on multiple movie IDs."""
    if not recommend.movie_id_to_idx or len(recommend.movie_id_to_idx) < 2:
        pytest.skip("Not enough movie IDs loaded, skipping test.")

    # Get two arbitrary valid movie IDs
    ids_iter = iter(recommend.movie_id_to_idx.keys())
    history = [next(ids_iter), next(ids_iter)]
    n = 10
    recommendations = recommend.recommend_top_k(history, n=n)

    assert isinstance(recommendations, list)
    assert len(recommendations) <= n

    if recommendations:
        assert all(isinstance(m, dict) for m in recommendations)
        assert all("movieId" in m and "title" in m for m in recommendations)
        # Ensure history items are not recommended back
        history_set = set(history)
        assert all(m["movieId"] not in history_set for m in recommendations)

def test_recommend_top_k_invalid_id():
    """Tests top-k recommendations with an invalid movie ID."""
    history = [-1] # Assuming -1 is an invalid ID
    recommendations = recommend.recommend_top_k(history)
    assert isinstance(recommendations, list)
    assert len(recommendations) == 0

def test_recommend_top_k_empty_history():
    """Tests top-k recommendations with empty history."""
    history = []
    recommendations = recommend.recommend_top_k(history)
    assert isinstance(recommendations, list)
    assert len(recommendations) == 0

# Add more tests as needed, e.g., for edge cases in _get_movie_details
