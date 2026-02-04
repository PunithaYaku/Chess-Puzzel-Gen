import os
from datasets import load_dataset
import pandas as pd

def download_data():
    print("Loading Lichess puzzles dataset from Hugging Face...")
    try:
        # Check if local file exists to avoid re-downloading huge dataset
        if os.path.exists("data/puzzles_full.csv"):
            print("Loading from local cache...")
            df = pd.read_csv("data/puzzles_full.csv")
        else:
            dataset = load_dataset("Lichess/chess-puzzles", split="train")
            df = dataset.to_pandas()
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/puzzles_full.csv", index=False)
            print(f"Total puzzles loaded and cached: {len(df)}")
        
        # Filter for Mates
        print("Filtering for Mate in 1-5...")
        for n in range(1, 6):
            theme = f"mateIn{n}"
            mate_df = df[df['Themes'].str.contains(theme, na=False)]
            # Take a healthy sample (e.g., 50k if available)
            sample_size = min(len(mate_df), 50000)
            mate_sample = mate_df.sample(n=sample_size, random_state=42)
            
            output_file = f"data/mate_in_{n}.csv"
            mate_sample.to_csv(output_file, index=False)
            print(f"Saved {sample_size} puzzles to {output_file}")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")

if __name__ == "__main__":
    download_data()
