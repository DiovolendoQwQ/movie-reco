import os
import polars as pl
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import logging
from pathlib import Path
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Using sklearn for simplicity, though manual split is also possible

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
RAW_DATA_DIR = Path("data/raw/ml-25m") # Need original ratings for timestamp split
INPUT_MATRIX_FILE = PROCESSED_DATA_DIR / "user_item.npz"
INPUT_SIMILARITY_FILE = PROCESSED_DATA_DIR / "sim.parquet"
RATINGS_FILE = RAW_DATA_DIR / "ratings.csv" # For timestamps
K = 10 # Evaluate Hit@K and NDCG@K
TEST_SIZE = 0.2 # Use 20% of interactions for testing

# --- Check if input files exist ---
if not INPUT_MATRIX_FILE.exists() or not INPUT_SIMILARITY_FILE.exists() or not RATINGS_FILE.exists():
    logging.error(f"Required input files not found. Ensure matrix, similarity, and ratings files exist.")
    exit(1)

# --- Load Data ---
logging.info("Loading data for evaluation...")
try:
    user_item_matrix = load_npz(INPUT_MATRIX_FILE)
    sim_df = pl.read_parquet(INPUT_SIMILARITY_FILE)
    ratings_df = pl.read_csv(RATINGS_FILE, columns=['userId', 'movieId', 'rating', 'timestamp'])
    logging.info("Data loaded.")

    # Need the mapping from original IDs to matrix indices used in 02_build_matrix.py
    # Recreate the maps based on the matrix dimensions or load them if saved previously.
    # For simplicity, let's assume the matrix indices correspond directly to sorted unique IDs
    # from the *filtered* ratings used in script 02. We need to reconstruct this mapping.

    logging.info("Reconstructing ID maps based on the matrix...")
    # Filter original ratings like in script 02
    RATING_THRESHOLD = 3.5
    implicit_df_orig = ratings_df.filter(pl.col("rating") >= RATING_THRESHOLD)

    unique_users_orig = implicit_df_orig["userId"].unique().sort()
    unique_movies_orig = implicit_df_orig["movieId"].unique().sort()

    user_map_orig = {user_id: i for i, user_id in enumerate(unique_users_orig)}
    movie_map_orig = {movie_id: i for i, movie_id in enumerate(unique_movies_orig)}
    # Reverse maps
    user_reverse_map = {i: user_id for user_id, i in user_map_orig.items()}
    movie_reverse_map = {i: movie_id for movie_id, i in movie_map_orig.items()}

    num_users = len(user_map_orig)
    num_items = len(movie_map_orig)

    if user_item_matrix.shape != (num_users, num_items):
         logging.warning(f"Matrix shape {user_item_matrix.shape} doesn't match reconstructed maps ({num_users}, {num_items}). Evaluation might be inaccurate.")
         # Adjust num_users/num_items if necessary, though this indicates a potential issue
         num_users, num_items = user_item_matrix.shape


    # Map original ratings df to matrix indices for timestamp-based splitting
    implicit_df_mapped = implicit_df_orig.with_columns([
        pl.col("userId").map_dict(user_map_orig).alias("user_idx"),
        pl.col("movieId").map_dict(movie_map_orig).alias("item_idx")
    ]).select(["user_idx", "item_idx", "timestamp"])

    logging.info("Data mapped for splitting.")

except Exception as e:
    logging.error(f"Error loading or mapping data: {e}")
    exit(1)


# --- Time-based Split ---
logging.info(f"Performing time-based split (test size: {TEST_SIZE})...")
# Sort interactions by timestamp
implicit_df_sorted = implicit_df_mapped.sort("timestamp")

# Group by user and split each user's history
train_indices = []
test_indices = []

# This is a simplified split: takes the latest 20% interactions overall as test
# A more robust user-level split is often preferred but more complex to implement here
n_interactions = len(implicit_df_sorted)
n_test = int(n_interactions * TEST_SIZE)
n_train = n_interactions - n_test

train_df = implicit_df_sorted.slice(0, n_train)
test_df = implicit_df_sorted.slice(n_train, n_test)

# Create sparse matrices for train and test sets
logging.info("Creating train and test sparse matrices...")
train_data = np.ones(len(train_df), dtype=np.float32)
train_matrix = csr_matrix((train_data, (train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy())), shape=(num_users, num_items))

test_data = np.ones(len(test_df), dtype=np.float32)
test_matrix = csr_matrix((test_data, (test_df["user_idx"].to_numpy(), test_df["item_idx"].to_numpy())), shape=(num_users, num_items))

logging.info(f"Train matrix shape: {train_matrix.shape}, nnz: {train_matrix.nnz}")
logging.info(f"Test matrix shape: {test_matrix.shape}, nnz: {test_matrix.nnz}")


# --- Prepare Similarity Data ---
# Convert the similarity DataFrame into a more usable format, e.g., a dictionary or sparse matrix
# For item-to-item, we need quick lookup of similar items for items in user history
logging.info("Preparing similarity lookup...")
# Pivot the table for easier lookup: item_idx_from -> list of (item_idx_to, similarity)
sim_lookup = sim_df.group_by("item_idx_from").agg([
    pl.col("item_idx_to").alias("similar_items"),
    pl.col("similarity").alias("scores")
]).sort("item_idx_from")

# Convert to dictionary for faster access: {item_idx_from: (indices_array, scores_array)}
sim_dict = {
    row["item_idx_from"]: (np.array(row["similar_items"]), np.array(row["scores"]))
    for row in tqdm(sim_lookup.iter_rows(named=True), total=len(sim_lookup), desc="Building lookup dict")
}
logging.info("Similarity lookup ready.")

# --- Generate Recommendations ---
logging.info(f"Generating Top-{K} recommendations for test users...")

def get_recommendations_for_user(user_idx, train_user_items, k):
    """Generates recommendations for a single user based on their training items."""
    candidate_items = {} # {item_idx: score}

    # Iterate through items the user interacted with in the training set
    for item_idx in train_user_items:
        if item_idx in sim_dict:
            similar_items, scores = sim_dict[item_idx]
            for sim_item_idx, score in zip(similar_items, scores):
                # Aggregate scores (simple sum here, could use weighted sum)
                candidate_items[sim_item_idx] = candidate_items.get(sim_item_idx, 0) + score

    # Remove items already interacted with in the training set
    for item_idx in train_user_items:
        if item_idx in candidate_items:
            del candidate_items[item_idx]

    # Sort candidates by score and take top K
    sorted_items = sorted(candidate_items.items(), key=lambda item: item[1], reverse=True)
    recommendations = [item_idx for item_idx, score in sorted_items[:k]]
    return recommendations

# Get users present in the test set
test_user_indices = np.unique(test_matrix.nonzero()[0])
all_recommendations = {}
train_matrix_lil = train_matrix.tolil() # Efficient row slicing
test_matrix_lil = test_matrix.tolil()

for user_idx in tqdm(test_user_indices, desc="Generating recommendations"):
    train_items = train_matrix_lil.rows[user_idx]
    if not train_items: # Skip users with no history in train set
        continue
    recs = get_recommendations_for_user(user_idx, train_items, K)
    all_recommendations[user_idx] = recs


# --- Evaluate Metrics ---
logging.info(f"Calculating Hit@{K} and NDCG@{K}...")
hits = 0
total_relevant_interactions = 0
ndcg_sum = 0.0

for user_idx in tqdm(test_user_indices, desc="Evaluating metrics"):
    if user_idx not in all_recommendations:
        continue # Skip users for whom recommendations couldn't be generated

    recommended_items = all_recommendations[user_idx]
    true_items = test_matrix_lil.rows[user_idx] # Items user interacted with in test set

    if not true_items: # Skip users with no interactions in test set
        continue

    total_relevant_interactions += len(true_items)
    relevant_set = set(true_items)
    recommended_set = set(recommended_items)

    # Hit Rate
    hit = len(relevant_set.intersection(recommended_set)) > 0
    if hit:
        hits += 1

    # NDCG
    dcg = 0.0
    for i, item_idx in enumerate(recommended_items):
        if item_idx in relevant_set:
            dcg += 1.0 / np.log2(i + 2) # i+2 because index is 0-based

    idcg = 0.0
    for i in range(min(len(true_items), K)):
        idcg += 1.0 / np.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg_sum += ndcg

num_test_users_evaluated = len(all_recommendations) # Users for whom recs were generated
hit_rate = hits / num_test_users_evaluated if num_test_users_evaluated > 0 else 0.0
average_ndcg = ndcg_sum / num_test_users_evaluated if num_test_users_evaluated > 0 else 0.0

logging.info("--- Evaluation Results ---")
logging.info(f"Evaluated on {num_test_users_evaluated} users from the test set.")
logging.info(f"Hit@{K}: {hit_rate:.4f}")
logging.info(f"NDCG@{K}: {average_ndcg:.4f}")
logging.info("--------------------------")

logging.info("Evaluation process finished.")
