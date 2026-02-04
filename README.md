# AI Chess Puzzle Generator

A generative AI system that creates **completely new** chess puzzles (Mate in 1 to Mate in 5) instead of just retrieving them from a database.

## Features
- **Generative AI**: Uses a Transformer model to generate board positions (FEN strings) that lead to forced checkmates.
- **Mate in N Support**: Can be prompted to generate puzzles ranging from Mate in 1 to Mate in 5.
- **Dataset Driven**: Trained on millions of Lichess puzzles to learn legal piece placements and mating patterns.

## Project Structure
- `download_data.py`: Specialized downloader that filters for specific Mate in N themes.
- `train_gen.py`: Trains a Transformer model to generate FEN strings.
- `generate_puzzles.py`: Prompts the AI to create new puzzles and provides basic validation.
- `train.py` / `main.py`: Baseline model for move prediction (evaluation).

## How to use
1. **Prepare Data**: `python download_data.py`
2. **Train Generator**: `python train_gen.py`
3. **Generate Puzzles**: `python generate_puzzles.py`
