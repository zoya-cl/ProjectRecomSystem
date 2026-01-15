#!/usr/bin/env python3
"""
Build Pipeline - Precompute and persist recommendation artifacts.

Run this ONCE (or when data changes) to build all artifacts.
After running, use recommender.py to load and query without recomputation.

Usage:
    python build_pipeline.py
    python build_pipeline.py --force  # Rebuild even if artifacts exist
"""

import os
import re
import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION - Change engine here to swap vectorization
# ============================================================
from engines.smart_tfidf_engine import SmartTfidfEngine
from engines.advanced_preprocessor import AdvancedTextPreprocessor

ENGINE_CLASS = SmartTfidfEngine  # <-- Use the smarter engine
PREPROCESSOR = AdvancedTextPreprocessor()  # <-- Use the advanced preprocessor

# Paths
DATA_PATH = Path("data/updated_data.csv")
ARTIFACTS_DIR = Path("artifacts")
ENGINE_PATH = ARTIFACTS_DIR / "engine.pkl"
VECTORS_PATH = ARTIFACTS_DIR / "vectors.npy"
DATAFRAME_PATH = ARTIFACTS_DIR / "processed_data.pkl"


def create_combined_text(row: pd.Series) -> str:
    """
    Combine relevant columns into a single weighted text blob.
    Weighting certain fields (like Title/Keywords) helps the recommender
    prioritize those matches.
    """
    # High Priority (4x)
    title = PREPROCESSOR.preprocess(str(row.get('Title of the Project', '')))
    
    # High Priority (3x)
    keywords = PREPROCESSOR.preprocess(str(row.get('Keywords', '')))
    
    # Medium Priority (2x)
    domain = PREPROCESSOR.preprocess(str(row.get('Domain', '')))
    short_desc = PREPROCESSOR.preprocess(str(row.get('Short Description', '')))
    tools = PREPROCESSOR.preprocess(str(row.get('Toolkits/Libraries', '')))
    
    # Standard Priority (1x)
    titles_alt = ' '.join([str(row.get('Title #1', '')), str(row.get('Title #2', ''))])
    usecases = ' '.join([
        str(row.get('Usecase #1', '')), 
        str(row.get('Usecase #2', '')), 
        str(row.get('Usecase #3', ''))
    ])
    
    # Build weighted string
    parts = [
        (title + " ") * 4,
        (keywords + " ") * 3,
        (domain + " ") * 2,
        (short_desc + " ") * 2,
        (tools + " ") * 2,
        PREPROCESSOR.preprocess(titles_alt),
        PREPROCESSOR.preprocess(usecases)
    ]
    
    return " ".join(parts).strip()


def load_and_preprocess_data() -> pd.DataFrame:
    """Load CSV and preprocess for recommendation."""
    print(f"[Pipeline] Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"[Pipeline] Loaded {len(df)} projects.")
    
    # Create combined text for vectorization
    df['combined_text'] = df.apply(create_combined_text, axis=1)
    
    # Clean individual columns for filtering
    df['domain_clean'] = df['Domain'].fillna('').apply(PREPROCESSOR.preprocess)
    df['language_clean'] = df['Language Used'].fillna('').apply(PREPROCESSOR.preprocess)
    df['toughness_clean'] = df['Toughness Level'].fillna('').apply(PREPROCESSOR.preprocess)
    
    return df


def build_artifacts(force: bool = False):
    """Build and save all artifacts."""
    
    # Check if artifacts already exist
    if not force and ENGINE_PATH.exists() and VECTORS_PATH.exists() and DATAFRAME_PATH.exists():
        print("[Pipeline] Artifacts already exist. Use --force to rebuild.")
        return
    
    # Create artifacts directory
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Initialize and fit engine
    print(f"[Pipeline] Fitting {ENGINE_CLASS.__name__}...")
    engine = ENGINE_CLASS()
    vectors = engine.fit_transform(df['combined_text'].tolist())
    
    print(f"[Pipeline] Vector shape: {vectors.shape}")
    
    # Save engine
    engine.save(str(ENGINE_PATH))
    
    # Save vectors
    np.save(str(VECTORS_PATH), vectors)
    print(f"[Pipeline] Saved vectors to {VECTORS_PATH}")
    
    # Save processed dataframe
    with open(DATAFRAME_PATH, 'wb') as f:
        pickle.dump(df, f)
    print(f"[Pipeline] Saved dataframe to {DATAFRAME_PATH}")
    
    print("\n" + "="*60)
    print("BUILD COMPLETE!")
    print("="*60)
    print(f"  Engine:    {ENGINE_PATH}")
    print(f"  Vectors:   {VECTORS_PATH}")
    print(f"  DataFrame: {DATAFRAME_PATH}")
    print("\nYou can now use recommender.py or app_cli.py to query.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build recommendation artifacts")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if artifacts exist")
    args = parser.parse_args()
    
    build_artifacts(force=args.force)
