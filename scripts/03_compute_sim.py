import os
import numpy as np
import polars as pl
from scipy.sparse import load_npz, csr_matrix
import implicit.nearest_neighbours
import logging
from pathlib import Path
import time
import torch # To check for CUDA availability
from tqdm import tqdm # Import tqdm for progress bar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/processed")
INPUT_MATRIX_FILE = PROCESSED_DATA_DIR / "user_item.npz"
OUTPUT_SIMILARITY_FILE = PROCESSED_DATA_DIR / "sim.parquet"
TOP_K = 50 # Number of similar items to store for each item

# --- Ensure directories exist ---
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Check if input file exists ---
if not INPUT_MATRIX_FILE.exists():
    logging.error(f"Input matrix file not found: {INPUT_MATRIX_FILE}. Please run 02_build_matrix.py first.")
    exit(1)

# --- Load the user-item matrix ---
logging.info(f"Loading user-item matrix from {INPUT_MATRIX_FILE}...")
try:
    user_item_matrix = load_npz(INPUT_MATRIX_FILE)
    logging.info(f"Matrix loaded. Shape: {user_item_matrix.shape}")
    # Implicit library expects item-user matrix, so transpose it
    item_user_matrix = user_item_matrix.T.tocsr()
    logging.info(f"Transposed matrix to item-user format (CSR). Shape: {item_user_matrix.shape}")
    # No need for explicit casting with the latest implicit version from source

except FileNotFoundError:
    logging.error(f"Could not find input file: {INPUT_MATRIX_FILE}")
    exit(1)
except Exception as e:
    logging.error(f"An error occurred loading the matrix: {e}")
    exit(1)


# --- Initialize the Cosine Recommender Model ---
model = implicit.nearest_neighbours.CosineRecommender(K=TOP_K)

# Check for GPU availability (still useful info)
use_gpu = torch.cuda.is_available()
if use_gpu:
    logging.info("CUDA is available. Attempting to use GPU for calculation.")
else:
    logging.info("CUDA not available. Using CPU for calculation.")


logging.info(f"Calculating top-{TOP_K} similar items using Cosine Similarity...")
start_time = time.time()
try:
    # Fit the model (should now handle int64 indices correctly)
    logging.info("Calling model.fit...")
    model.fit(item_user_matrix, show_progress=True)
    logging.info("model.fit call completed.")

    # Get similar items for all items using the trained model
    # model.similar_items returns (indices, scores) for a given item_id or all items
    # We want the full similarity matrix or top-K for all items.
    # Let's get the top K for each item.
    all_items = np.arange(item_user_matrix.shape[0])
    similar_items_list = []
    scores_list = []

    # Iterate through items to get similarities (can be memory intensive for large K)
    # A more direct way might exist depending on the implicit version,
    # but this ensures we get the top K for each item.
    # Alternatively, model.similarity matrix might be accessible if needed,
    # but the plan asks for sim.parquet which implies a Top-K structure.

    logging.info(f"Retrieving top-{TOP_K} similar items for each item...")
    # Use recommend_all which is efficient for getting recommendations (similar items here)
    # We need item indices, not user indices. Let's use similar_items batch processing if possible.
    # The `similar_items` method can take an array of item IDs.
    batch_size = 1024 # Process in batches to manage memory
    num_items = item_user_matrix.shape[0]
    all_similar_indices = np.zeros((num_items, TOP_K), dtype=np.int32)
    all_similar_scores = np.zeros((num_items, TOP_K), dtype=np.float32)

    for start_idx in tqdm(range(0, num_items, batch_size), desc="Finding similar items"):
        end_idx = min(start_idx + batch_size, num_items)
        item_ids_batch = np.arange(start_idx, end_idx)
        # N=TOP_K+1 because the item itself is usually the most similar
        # Removed filter_already_liked_items=False as it's not supported for item similarity
        indices, scores = model.similar_items(item_ids_batch, N=TOP_K + 1, filter_items=None)

        # Remove the item itself from the list (usually the first one with score 1.0)
        # Check if the first item is the item itself
        for i in range(len(item_ids_batch)):
            original_item_idx = item_ids_batch[i]
            if indices[i, 0] == original_item_idx:
                all_similar_indices[original_item_idx] = indices[i, 1:]
                all_similar_scores[original_item_idx] = scores[i, 1:]
            else:
                 # If the item itself wasn't the top match (unlikely with cosine), take top K
                all_similar_indices[original_item_idx] = indices[i, :TOP_K]
                all_similar_scores[original_item_idx] = scores[i, :TOP_K]


    end_time = time.time()
    logging.info(f"Similarity calculation finished in {end_time - start_time:.2f} seconds.")

    # --- Format results into a DataFrame ---
    logging.info("Formatting similarity results into a DataFrame...")
    item_from_indices = np.repeat(np.arange(num_items), TOP_K)
    item_to_indices = all_similar_indices.flatten()
    similarity_scores = all_similar_scores.flatten()

    sim_df = pl.DataFrame({
        "item_idx_from": item_from_indices,
        "item_idx_to": item_to_indices,
        "similarity": similarity_scores
    })

    # Filter out zero similarity scores if any (though unlikely with top K)
    sim_df = sim_df.filter(pl.col("similarity") > 0)

    # --- Save the similarity DataFrame ---
    logging.info(f"Saving similarity data to {OUTPUT_SIMILARITY_FILE}...")
    sim_df.write_parquet(OUTPUT_SIMILARITY_FILE)
    logging.info("Similarity data saved.")

except ImportError:
     logging.error("Implicit library not found. Please install it: pip install implicit")
     exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during similarity calculation: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback for debugging
    exit(1)

logging.info("Similarity calculation process finished.")
