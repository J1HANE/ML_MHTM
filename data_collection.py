#!/usr/bin/env python3
"""Data collection script for the Mental Health Trend Monitor (MHTM).

Collects Reddit posts from specified subreddits using the Arctic Shift public API,
extracts a fixed set of features, saves checkpoints per subreddit, and builds a
balanced CSV dataset suitable for binary classification.

Requirements:
    pip install requests pandas
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DISTRESS_SUBREDDITS: List[str] = ["depression", "SuicideWatch", "anxiety", "mentalhealth"]
NEUTRAL_SUBREDDITS: List[str] = ["CasualConversation", "AskReddit", "fitness", "productivity"]

# Target number of posts per subreddit
TARGETS: Dict[str, int] = {
    # distress: 450 each → 1,800 total
    "depression": 450,
    "SuicideWatch": 450,
    "anxiety": 450,
    "mentalhealth": 450,
    # neutral: 2,550 each → 10,200 total
    "CasualConversation": 2550,
    "AskReddit": 2550,
    "fitness": 2550,
    "productivity": 2550,
}

PAGE_SIZE: int = 100  # API limit per request
BASE_URL: str = "https://arctic-shift.photon-reddit.com/api/posts/search"
OUTPUT_DIR: Path = Path(__file__).parent / "data"
CHECKPOINT_DIR: Path = OUTPUT_DIR
LOG_FILE: Path = OUTPUT_DIR / "collection.log"
DATASET_PATH: Path = OUTPUT_DIR / "dataset.csv"
SAMPLE_PATH: Path = OUTPUT_DIR / "sample.csv"

# ---------------------------------------------------------------------------
# Logging setup (console + file)
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_checkpoint(subreddit: str) -> List[Dict]:
    """Load a checkpoint JSON file for *subreddit* if it exists.

    Returns an empty list when the file is missing or unreadable.
    """
    cp_path = CHECKPOINT_DIR / f"checkpoint_{subreddit}.json"
    if cp_path.is_file():
        try:
            with cp_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded checkpoint for r/{subreddit} ({len(data)} posts).")
                return data
        except Exception as e:
            logger.warning(f"Failed to read checkpoint for r/{subreddit}: {e}")
    return []


def save_checkpoint(subreddit: str, rows: List[Dict]):
    """Save *rows* to a checkpoint file for *subreddit*.
    """
    cp_path = CHECKPOINT_DIR / f"checkpoint_{subreddit}.json"
    try:
        with cp_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved for r/{subreddit} ({len(rows)} posts).")
    except Exception as e:
        logger.error(f"Unable to write checkpoint for r/{subreddit}: {e}")


def extract_features(post: Dict, subreddit: str) -> Dict:
    """Transform a raw API *post* into the required feature dictionary.
    """
    # Timestamp handling
    created = post.get("created_utc")
    if isinstance(created, (int, float)):
        hour = datetime.utcfromtimestamp(created).hour
    else:
        hour = -1

    title = post.get("title") or ""
    body = post.get("selftext") or post.get("body") or ""
    if body in ["[deleted]", "[removed]"]:
        body = ""

    # Feature construction
    features = {
        "hour_of_day": hour,
        "title_length_chars": len(title),
        "body_length_chars": len(body),
        "title_ends_with_question": int(title.strip().endswith("?")),
        "is_nsfw": int(bool(post.get("over_18", False))),
        "has_url_in_body": int("http://" in body or "https://" in body),
        "author_is_new": -1,
        "author_comment_karma": -1,
        "author_post_karma": -1,
        "flair_text": post.get("link_flair_text") or "None",
        "subreddit": subreddit,
        "is_distress": 1 if subreddit in DISTRESS_SUBREDDITS else 0,
    }
    return features


def fetch_subreddit(subreddit: str, target: int) -> List[Dict]:
    """Retrieve *target* posts from *subreddit* respecting API limits.

    Returns a list of feature dictionaries.
    """
    logger.info(f"Starting collection for r/{subreddit} – target {target} posts.")
    collected: List[Dict] = []
    after_timestamp: int = 0  # start from epoch
    page = 0

    while len(collected) < target:
        page += 1
        # Build request parameters; omit 'after' on the first page (timestamp 0) to avoid API errors.
        params = {"subreddit": subreddit, "limit": PAGE_SIZE}
        if after_timestamp:
            params["after"] = after_timestamp
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            # Determine rows safely – handle both list and dict responses.
            try:
                json_data = response.json()
                if isinstance(json_data, dict) and "data" in json_data:
                    rows_list = json_data["data"]
                else:
                    rows_list = json_data
                rows_count = len(rows_list) if isinstance(rows_list, list) else "?"
            except Exception:
                rows_count = "?"
            logger.info(
                f"API call | r/{subreddit} | page {page} | status {response.status_code} | rows {rows_count}"
            )
            response.raise_for_status()
            # Preserve the parsed rows list for later processing.
            data = rows_list if isinstance(rows_list, list) else []
        except requests.RequestException as e:
            logger.error(f"Request error for r/{subreddit} page {page}: {e}")
            break

        if not data:
            logger.info(f"No more data returned for r/{subreddit} after page {page}.")
            break

        for post in data:
            if len(collected) >= target:
                break
            features = extract_features(post, subreddit)
            collected.append(features)
            created = post.get("created_utc")
            if isinstance(created, (int, float)) and created > after_timestamp:
                after_timestamp = int(created)

        if len(data) < PAGE_SIZE:
            logger.info(f"Reached end of subreddit r/{subreddit} after page {page}.")
            break

        time.sleep(1)  # basic rate‑limit handling

    logger.info(f"Finished r/{subreddit}: collected {len(collected)} posts (target {target}).")
    return collected


def verify_dataset(df: pd.DataFrame):
    """Print a simple verification report for the final dataset.
    Handles the case where the DataFrame may be empty (e.g., no data fetched).
    """
    total = len(df)
    cols = len(df.columns)
    # Ensure the required column exists for reporting; if missing, treat count as 0.
    distress_cnt = int(df["is_distress"].sum()) if "is_distress" in df.columns else 0
    minority_ratio = distress_cnt / total if total else 0
    logger.info("--- Dataset verification report ---")
    logger.info(
        f"Total rows: {total} (✓)" if total >= 10_000 else f"Total rows: {total} (✗) – need >=10k"
    )
    logger.info(
        f"Number of columns: {cols} (✓)" if cols >= 8 else f"Number of columns: {cols} (✗) – need >=8"
    )
    logger.info(
        f"Distress rows: {distress_cnt} ({minority_ratio:.2%}) "
        + (
            "(✓)" if 0.05 <= minority_ratio <= 0.25 else "(✗) – need 5–25%"
        )
    )
    if "is_distress" in df.columns:
        logger.info("Class distribution:")
        logger.info(df["is_distress"].value_counts().to_string())
    else:
        logger.info("Class distribution: column 'is_distress' not present.")


def main():
    all_rows: List[Dict] = []

    # Process distress subreddits first, then neutral ones.
    for subreddit in DISTRESS_SUBREDDITS + NEUTRAL_SUBREDDITS:
        # Skip if checkpoint already exists
        checkpoint = load_checkpoint(subreddit)
        if checkpoint:
            all_rows.extend(checkpoint)
            continue
        target = TARGETS.get(subreddit, 0)
        rows = fetch_subreddit(subreddit, target)
        all_rows.extend(rows)
        save_checkpoint(subreddit, rows)

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    # Shuffle with a deterministic seed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Persist dataset and a sample
    df.to_csv(DATASET_PATH, index=False)
    df.head(100).to_csv(SAMPLE_PATH, index=False)
    logger.info(f"Dataset saved to {DATASET_PATH}")
    logger.info(f"Sample saved to {SAMPLE_PATH}")

    # Verification report
    verify_dataset(df)


if __name__ == "__main__":
    main()
