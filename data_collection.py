"""
MHTM Data Collection Script.

Collects Reddit data from HuggingFace, processes it into a binary classification dataset
for the Mental Health Trend Monitor (MHTM) project.
"""

import os
import logging
import pandas as pd
from datasets import load_dataset
from datetime import datetime

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
# HuggingFace dataset candidates (tried in order)
DATASET_CANDIDATES = [
    "vibhorag101/reddit-mental-health-dataset-phr",
    "solomonk/reddit_mental_health_posts",
    "Sharathhebbar24/reddit_mental_health",
    "EmotionDetection/mental_health_reddit"
]

# Target counts and ratios
TOTAL_ROWS = 12000
MINORITY_RATIO = 0.15
DISTRESS_COUNT = int(TOTAL_ROWS * MINORITY_RATIO)  # 1,800
NEUTRAL_COUNT = TOTAL_ROWS - DISTRESS_COUNT        # 10,200
RANDOM_STATE = 42

# Subreddit definitions
SUBREDDITS_DISTRESS = ["depression", "SuicideWatch", "anxiety", "mentalhealth"]
SUBREDDITS_NEUTRAL = ["CasualConversation", "AskReddit", "fitness", "productivity"]

# Output paths
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "collection.log")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "dataset.csv")
FINAL_DATASET_FILE = os.path.join(DATA_DIR, "dataset.csv")
SAMPLE_FILE = os.path.join(DATA_DIR, "sample.csv")

# ==============================================================================
# SETUP LOGGING
# ==============================================================================
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger("DataCollection")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Terminal handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(formatter)
logger.addHandler(fh)


def safe_get(row, keys, default=None):
    """Safely get the first available key from a row dict."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def extract_hour(timestamp):
    """Extract hour (0-23) from a UTC timestamp. Returns -1 if unavailable/invalid."""
    if timestamp in [None, "", "NaN", "nan"]:
        return -1
    try:
        return datetime.utcfromtimestamp(float(timestamp)).hour
    except (ValueError, TypeError, OSError):
        pass
    
    try:
        dt = pd.to_datetime(timestamp, utc=True)
        return dt.hour
    except Exception:
        return -1


def process_row(row, is_distress_val):
    """Extract required features from a single dataset row."""
    timestamp = safe_get(row, ["created_utc", "created"], None)
    hour_of_day = extract_hour(timestamp)

    # Title processing
    title = str(safe_get(row, ["title"], ""))
    title = title if title.lower() != "nan" else ""
    title_length_chars = len(title)
    title_ends_with_question = 1 if title.strip().endswith("?") else 0

    # Body processing
    body = str(safe_get(row, ["selftext", "body", "text"], ""))
    body = body if body.lower() != "nan" else ""
    body_length_chars = len(body)
    has_url_in_body = 1 if ("http://" in body or "https://" in body) else 0

    # NSFW processing
    nsfw_val = safe_get(row, ["over_18", "nsfw"], False)
    is_nsfw = 1 if nsfw_val in [True, "True", "true", 1, "1"] else 0

    # Flair processing
    flair = str(safe_get(row, ["link_flair_text", "flair"], "None"))
    if not flair or flair.lower() in ["nan", "none"]:
        flair = "None"

    # Subreddit processing
    subreddit = str(safe_get(row, ["subreddit"], "Unknown"))
    if subreddit.lower() == "nan":
        subreddit = "Unknown"

    return {
        "hour_of_day": hour_of_day,
        "title_length_chars": title_length_chars,
        "body_length_chars": body_length_chars,
        "title_ends_with_question": title_ends_with_question,
        "is_nsfw": is_nsfw,
        "has_url_in_body": has_url_in_body,
        "author_is_new": -1,
        "author_comment_karma": -1,
        "author_post_karma": -1,
        "flair_text": flair,
        "subreddit": subreddit,
        "is_distress": is_distress_val
    }


def determine_label_and_subreddit(row):
    """
    Determine if a row is distress (1) or neutral (0) based on subreddit or label columns.
    Returns (label, subreddit) or (None, None) if undetermined.
    """
    sub = str(safe_get(row, ["subreddit"], "")).strip()
    
    # Check subreddit logic first
    if sub and sub.lower() != "nan":
        for ds in SUBREDDITS_DISTRESS:
            if ds.lower() == sub.lower():
                return 1, ds
        for ns in SUBREDDITS_NEUTRAL:
            if ns.lower() == sub.lower():
                return 0, ns

    # Fallback to label column
    label_val = safe_get(row, ["label", "class", "category", "target", "is_depression"], None)
    if label_val is not None:
        try:
            l_int = int(float(label_val))
            is_dist = 1 if l_int != 0 else 0
            return is_dist, sub if sub else "Unknown"
        except (ValueError, TypeError):
            l_str = str(label_val).lower()
            if any(d.lower() in l_str for d in SUBREDDITS_DISTRESS) or l_str in ['1', 'true', 'distress', 'depression']:
                return 1, sub if sub else "Unknown"
            elif any(n.lower() in l_str for n in SUBREDDITS_NEUTRAL) or l_str in ['0', 'false', 'neutral', 'safe']:
                return 0, sub if sub else "Unknown"
    
    return None, None


def load_data_from_huggingface():
    """Attempt to load dataset from HF candidates."""
    for ds_name in DATASET_CANDIDATES:
        logger.info(f"Attempting to load dataset: {ds_name}")
        try:
            dataset = load_dataset(ds_name)
            if "train" in dataset:
                ds = dataset["train"]
            else:
                first_split = list(dataset.keys())[0]
                ds = dataset[first_split]
            logger.info(f"Successfully loaded dataset: {ds_name}")
            return ds
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")
    
    raise RuntimeError("Failed to load any candidate datasets. Please check your internet connection.")


def main():
    """Main data collection routine."""
    logger.info("Starting MHTM Data Collection process.")
    
    dataset = load_data_from_huggingface()
    
    distress_rows = []
    neutral_rows = []
    
    # Pass 1: Extract Distress Posts
    logger.info("Extracting distress posts...")
    for row in dataset:
        if len(distress_rows) >= DISTRESS_COUNT:
            break
        label, sub = determine_label_and_subreddit(row)
        if label == 1:
            row['subreddit'] = sub
            distress_rows.append(process_row(row, 1))

    # Save Checkpoint
    df_distress = pd.DataFrame(distress_rows)
    df_distress.to_csv(CHECKPOINT_FILE, index=False)
    logger.info(f"Checkpoint saved: {len(df_distress)} distress posts to {CHECKPOINT_FILE}")

    # Pass 2: Extract Neutral Posts
    logger.info("Extracting neutral posts...")
    for row in dataset:
        if len(neutral_rows) >= NEUTRAL_COUNT:
            break
        label, sub = determine_label_and_subreddit(row)
        if label == 0:
            row['subreddit'] = sub
            neutral_rows.append(process_row(row, 0))

    if len(distress_rows) < DISTRESS_COUNT:
        logger.warning(f"Only found {len(distress_rows)} distress rows (target: {DISTRESS_COUNT})")
    if len(neutral_rows) < NEUTRAL_COUNT:
        logger.warning(f"Only found {len(neutral_rows)} neutral rows (target: {NEUTRAL_COUNT})")
        
    # Combine and Shuffle
    all_rows = distress_rows + neutral_rows
    df = pd.DataFrame(all_rows)
    
    logger.info("Shuffling the dataset...")
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    # Save Final Dataset
    df.to_csv(FINAL_DATASET_FILE, index=False)
    logger.info(f"Final dataset saved to {FINAL_DATASET_FILE} with {len(df)} rows.")
    
    # Save Sample
    if len(df) >= 100:
        df_sample = df.sample(n=100, random_state=RANDOM_STATE)
    else:
        df_sample = df
    df_sample.to_csv(SAMPLE_FILE, index=False)
    logger.info(f"Sample dataset saved to {SAMPLE_FILE}.")

    # Terminal Report & Verification Checks
    actual_total = len(df)
    actual_columns = len(df.columns)
    actual_distress = len(df[df['is_distress'] == 1])
    minority_ratio = actual_distress / actual_total if actual_total > 0 else 0
    
    print("\n" + "="*50)
    print("CLASS DISTRIBUTION REPORT")
    print("="*50)
    print(f"Total Rows: {actual_total}")
    print(f"Distress (1): {actual_distress} ({minority_ratio:.2%})")
    print(f"Neutral  (0): {actual_total - actual_distress} ({(1 - minority_ratio):.2%})")
    
    print("\n" + "="*50)
    print("CONSTRAINT VERIFICATION CHECKLIST")
    print("="*50)
    
    check_total = actual_total >= 10000
    check_cols = actual_columns >= 8
    check_ratio = 0.05 <= minority_ratio <= 0.25
    
    print(f"[{'✓' if check_total else '✗'}] Total rows >= 10,000 (Current: {actual_total})")
    print(f"[{'✓' if check_cols else '✗'}] Columns >= 8 (Current: {actual_columns})")
    print(f"[{'✓' if check_ratio else '✗'}] Minority ratio between 5% and 25% (Current: {minority_ratio:.2%})")
    print("="*50)


if __name__ == "__main__":
    main()
