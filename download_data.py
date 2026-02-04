import os
from datasets import load_dataset
import pandas as pd

def download_data():
    print("Loading Lichess puzzles dataset from Hugging Face...")
    # Loading just the 'train' split and taking a subset if it's too large
    # For now, let's try to load the whole thing but we will only save a portion for GitHub
    try:
        dataset = load_dataset("Lichess/chess-puzzles", split="train")
        df = dataset.to_pandas()
        
        print(f"Total puzzles loaded: {len(df)}")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Save a small subset (10k puzzles) for a quick test/demo to GitHub
        # We don't want to push GBs of data to GitHub
        demo_df = df.head(10000)
        demo_df.to_csv("data/puzzles_subset.csv", index=False)
        print("Saved 10,000 puzzles to data/puzzles_subset.csv for GitHub.")
        
        # Save the full dataset locally (not for Git)
        # Note: Github will ignore this if we add it to .gitignore
        df.to_csv("data/puzzles_full.csv", index=False)
        print("Saved full dataset to data/puzzles_full.csv (Local only).")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_data()
