import os
import requests
import zipfile
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
RAW_DATA_DIR = "data/raw"
ZIP_FILE_PATH = os.path.join(RAW_DATA_DIR, "ml-25m.zip")
EXTRACTED_DIR_NAME = "ml-25m" # The name of the directory inside the zip file
EXTRACTED_DATA_PATH = os.path.join(RAW_DATA_DIR, EXTRACTED_DIR_NAME)

# --- Ensure directories exist ---
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# --- Download the dataset ---
if not os.path.exists(ZIP_FILE_PATH):
    logging.info(f"Downloading MovieLens 25M dataset from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        with open(ZIP_FILE_PATH, 'wb') as file, tqdm(
            desc=os.path.basename(ZIP_FILE_PATH),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)

        if total_size != 0 and bar.n != total_size:
            logging.error("ERROR, something went wrong during download")
        else:
            logging.info("Download complete.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download dataset: {e}")
        # Optionally remove partially downloaded file
        if os.path.exists(ZIP_FILE_PATH):
            os.remove(ZIP_FILE_PATH)
        exit(1) # Exit if download fails
else:
    logging.info(f"Dataset already downloaded at {ZIP_FILE_PATH}")

# --- Extract the dataset ---
if not os.path.exists(EXTRACTED_DATA_PATH):
    logging.info(f"Extracting {ZIP_FILE_PATH} to {RAW_DATA_DIR}...")
    try:
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            # Get list of files in zip
            file_list = zip_ref.namelist()
            # Extract files with progress
            for file in tqdm(iterable=file_list, total=len(file_list), desc="Extracting files"):
                zip_ref.extract(member=file, path=RAW_DATA_DIR)
        logging.info("Extraction complete.")
    except zipfile.BadZipFile:
        logging.error(f"Error: The downloaded file {ZIP_FILE_PATH} is not a valid zip file or is corrupted.")
        # Optionally remove the corrupted zip file
        # os.remove(ZIP_FILE_PATH)
        exit(1)
    except Exception as e:
        logging.error(f"An error occurred during extraction: {e}")
        exit(1)
else:
    logging.info(f"Dataset already extracted at {EXTRACTED_DATA_PATH}")

logging.info("Data download and extraction process finished.")
