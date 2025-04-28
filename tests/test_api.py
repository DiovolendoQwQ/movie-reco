import pytest
import httpx # Async HTTP client for testing FastAPI
import sys
from pathlib import Path
import random

# Add the app directory to the Python path (optional, but can be useful)
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# --- Configuration ---
# Assumes the application is running via docker-compose on http://localhost
BASE_URL = "http://localhost/api" # Use the Nginx proxy URL

# --- Test Cases ---

@pytest.mark.asyncio # Mark test as async
async def test_get_genres_api():
    """Tests the /genres endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/genres")
            response.raise_for_status() # Raise exception for 4xx or 5xx status codes
            data = response.json()

            assert response.status_code == 200
            assert "genres" in data
            assert isinstance(data["genres"], list)
            assert len(data["genres"]) > 0
            assert "Action" in data["genres"] # Check for expected genre

        except httpx.RequestError as e:
            pytest.fail(f"API request failed: {e}. Is the service running via docker compose?")
        except Exception as e:
             pytest.fail(f"An unexpected error occurred: {e}")


@pytest.mark.asyncio
async def test_get_random_api_valid_genre():
    """Tests the /random endpoint with a valid genre."""
    genre = "Drama" # Assume Drama exists
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/random?genre={genre}")
            response.raise_for_status()
            data = response.json()

            assert response.status_code == 200
            assert "recommendations" in data
            assert isinstance(data["recommendations"], list)
            # API should return RECOMMENDATION_COUNT (20) movies if available
            assert len(data["recommendations"]) <= 20
            if data["recommendations"]:
                 assert "movieId" in data["recommendations"][0]
                 assert "title" in data["recommendations"][0]

        except httpx.RequestError as e:
            pytest.fail(f"API request failed: {e}")
        except Exception as e:
             pytest.fail(f"An unexpected error occurred: {e}")


@pytest.mark.asyncio
async def test_get_random_api_invalid_genre():
    """Tests the /random endpoint with an invalid genre."""
    genre = "InvalidGenreForTesting123"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/random?genre={genre}")
            # Expecting a 404 Not Found error
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
            assert genre in data["detail"] # Check if error message mentions the genre

        except httpx.RequestError as e:
            pytest.fail(f"API request failed: {e}")
        except Exception as e:
             pytest.fail(f"An unexpected error occurred: {e}")


@pytest.mark.asyncio
async def test_post_choice_api_and_session():
    """
    Tests the /choice endpoint and basic session handling.
    Requires Redis to be running and accessible by the API container.
    """
    async with httpx.AsyncClient() as client:
        try:
            # 1. Get initial random recommendations to find a valid movie ID
            genre = "Comedy"
            response_random = await client.get(f"{BASE_URL}/random?genre={genre}")
            response_random.raise_for_status()
            random_movies = response_random.json()["recommendations"]
            if not random_movies:
                pytest.skip(f"No movies found for genre {genre}, cannot proceed with choice test.")
            
            test_movie_id = random_movies[0]["movieId"]
            test_movie_title = random_movies[0]["title"]
            print(f"\nUsing movie for choice test: ID={test_movie_id}, Title='{test_movie_title}'")

            # 2. Post the choice
            response_choice1 = await client.post(f"{BASE_URL}/choice", json={"movieId": test_movie_id})
            response_choice1.raise_for_status()
            data_choice1 = response_choice1.json()

            assert response_choice1.status_code == 200
            assert "recommendations" in data_choice1
            assert isinstance(data_choice1["recommendations"], list)
            # Check if the chosen movie is NOT in the recommendations
            assert test_movie_id not in [m["movieId"] for m in data_choice1["recommendations"]]
            
            # Extract the session cookie set by the server
            session_cookie = response_choice1.cookies.get( "reco_session_id")
            assert session_cookie is not None, "Session cookie was not set in the response"
            print(f"Session cookie received: {session_cookie}")

            # 3. Post another choice using the same session (via cookie)
            # Find another movie ID from the first choice's recommendations
            if not data_choice1["recommendations"]:
                 pytest.skip("First choice returned no recommendations, cannot test second choice.")
            
            second_movie_id = data_choice1["recommendations"][0]["movieId"]
            print(f"Using second movie for choice test: ID={second_movie_id}")

            # Ensure the client sends the cookie back
            cookies = { "reco_session_id": session_cookie }
            response_choice2 = await client.post(f"{BASE_URL}/choice", json={"movieId": second_movie_id}, cookies=cookies)
            response_choice2.raise_for_status()
            data_choice2 = response_choice2.json()

            assert response_choice2.status_code == 200
            assert "recommendations" in data_choice2
            # Check if both chosen movies are NOT in the new recommendations
            history_set = {test_movie_id, second_movie_id}
            assert all(m["movieId"] not in history_set for m in data_choice2["recommendations"])

            # 4. Test reset endpoint
            response_reset = await client.post(f"{BASE_URL}/reset", cookies=cookies)
            response_reset.raise_for_status()
            data_reset = response_reset.json()
            assert response_reset.status_code == 200
            assert "message" in data_reset

            # 5. Post choice again after reset - should be like the first choice
            response_choice3 = await client.post(f"{BASE_URL}/choice", json={"movieId": test_movie_id}, cookies=cookies)
            response_choice3.raise_for_status()
            data_choice3 = response_choice3.json()
            # Compare recommendations with the first choice (might not be identical due to randomness/updates, but structure should match)
            assert len(data_choice3["recommendations"]) == len(data_choice1["recommendations"])


        except httpx.RequestError as e:
            pytest.fail(f"API request failed: {e}. Is Redis running and accessible?")
        except Exception as e:
             pytest.fail(f"An unexpected error occurred: {e}")

# Add more tests, e.g., for missing parameters, invalid request bodies etc.
