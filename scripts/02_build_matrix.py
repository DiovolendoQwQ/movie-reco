import os
import polars as pl
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
RAW_DATA_DIR = Path("data/raw/ml-25m")
PROCESSED_DATA_DIR = Path("data/processed")
RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
OUTPUT_MATRIX_FILE = PROCESSED_DATA_DIR / "user_item.npz"
OUTPUT_GENRE_MAP_FILE = PROCESSED_DATA_DIR / "genre_map.parquet"
RATING_THRESHOLD = 3.5 # Threshold to consider a rating as a positive interaction

# --- Ensure directories exist ---
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Check if input files exist ---
if not RATINGS_FILE.exists() or not MOVIES_FILE.exists():
    logging.error(f"Required input files not found in {RAW_DATA_DIR}. Please run 01_download_data.py first.")
    exit(1)

# --- Load and process ratings ---
logging.info(f"Loading ratings from {RATINGS_FILE}...")
try:
    # Load all ratings for the final run
    ratings_df = pl.read_csv(RATINGS_FILE)
    logging.info(f"Loaded {len(ratings_df)} ratings.")

    # Filter ratings above threshold and create implicit feedback
    logging.info(f"Filtering ratings >= {RATING_THRESHOLD} and creating implicit feedback...")
    implicit_df = ratings_df.filter(pl.col("rating") >= RATING_THRESHOLD)
    logging.info(f"Found {len(implicit_df)} positive interactions.")

    # Create unique user and item IDs (contiguous integers starting from 0)
    logging.info("Mapping user and movie IDs to contiguous integers...")
    unique_users = implicit_df["userId"].unique().sort()
    unique_movies = implicit_df["movieId"].unique().sort()

    user_map = {user_id: i for i, user_id in enumerate(unique_users)}
    movie_map = {movie_id: i for i, movie_id in enumerate(unique_movies)}
    # Create reverse map for later use if needed (e.g., evaluation)
    # movie_reverse_map = {i: movie_id for movie_id, i in movie_map.items()}

    # Map original IDs to new contiguous IDs using replace
    implicit_df = implicit_df.with_columns([
        pl.col("userId").replace(user_map).alias("user_idx"),
        pl.col("movieId").replace(movie_map).alias("item_idx")
    ])

    # --- Build sparse matrix ---
    logging.info("Building user-item sparse matrix (CSR format)...")
    num_users = len(unique_users)
    num_items = len(unique_movies)
    # Use mapped indices and set data to 1 for implicit feedback
    # Explicitly cast indices to int32 to avoid dtype mismatch issues with implicit's C++ backend
    user_indices = implicit_df["user_idx"].to_numpy().astype(np.int32)
    item_indices = implicit_df["item_idx"].to_numpy().astype(np.int32)
    data = np.ones(len(implicit_df), dtype=np.float32) # Implicit feedback is 1

    user_item_matrix = csr_matrix((data, (user_indices, item_indices)), shape=(num_users, num_items))

    # --- Save the matrix ---
    logging.info(f"Saving sparse matrix to {OUTPUT_MATRIX_FILE}...")
    save_npz(OUTPUT_MATRIX_FILE, user_item_matrix)
    logging.info(f"Matrix saved. Shape: {user_item_matrix.shape}, Non-zero elements: {user_item_matrix.nnz}")

    # --- Create and save genre map ---
    logging.info(f"Loading movies from {MOVIES_FILE} to create genre map...")
    movies_df = pl.read_csv(MOVIES_FILE)

    # Filter movies that are actually present in the interaction matrix
    movies_in_matrix = pl.DataFrame({"movieId": list(movie_map.keys())})
    movies_df = movies_df.join(movies_in_matrix, on="movieId", how="inner")

    # Map original movie IDs to the new contiguous item indices using replace
    movies_df = movies_df.with_columns(
        pl.col("movieId").replace(movie_map).alias("item_idx")
    )

    # Select relevant columns and save
    genre_map_df = movies_df.select(["movieId", "item_idx", "title", "genres"])
    logging.info(f"Saving genre map to {OUTPUT_GENRE_MAP_FILE}...")
    genre_map_df.write_parquet(OUTPUT_GENRE_MAP_FILE)
    logging.info("Genre map saved.")

except pl.exceptions.ComputeError as e:
    logging.error(f"Polars computation error: {e}")
    exit(1)
except FileNotFoundError:
    logging.error(f"Could not find input file. Ensure {RATINGS_FILE} and {MOVIES_FILE} exist.")
    exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
    exit(1)

logging.info("Matrix building process finished.")
