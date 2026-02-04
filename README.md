# Chess Puzzle Generator & AI Trainer

An AI-powered chess puzzle system using Lichess records.

## Features
- **Data Download**: Fetches over 5 million puzzles from the Lichess Hugging Face dataset.
- **Model Training**: A PyTorch-based CNN that learns to predict the best initial move for a given FEN.
- **Inference**: Use the trained model to suggest moves in any position.

## Project Structure
- `download_data.py`: Downloads and prepares the data. Saves a 10k subset for GitHub and the full dataset locally.
- `train.py`: Trains the Neural Network on the downloaded subset.
- `main.py`: Entry point to test the trained model.
- `data/`: Contains the puzzle datasets (puzzles_full.csv is ignored by Git).

## Getting Started
1. **Install Dependencies**:
   ```bash
   pip install datasets pandas chess torch
   ```
2. **Download Data**:
   ```bash
   python download_data.py
   ```
3. **Train Model**:
   ```bash
   python train.py
   ```
4. **Run Application**:
   ```bash
   python main.py
   ```
