# AI Chess Puzzle Generator

A generative AI system that creates **completely new** chess puzzles (Mate in 1 to Mate in 5) instead of just retrieving them from a database.

## 🚀 Features
- **Generative AI**: Uses a Transformer model to generate board positions (FEN strings) that lead to forced checkmates.
- **Mate in N Support**: Can be prompted to generate puzzles ranging from Mate in 1 to Mate in 5.
- **Dataset Driven**: Trained on millions of Lichess puzzles to learn legal piece placements and mating patterns.
- **Web API**: Built-in FastAPI server for easy integration.

## 📁 Project Structure
- `app.py`: FastAPI server for generating puzzles via REST API.
- `download_data.py`: Specialized downloader that filters for specific Mate in N themes.
- `train_gen.py`: Trains a Transformer model to generate FEN strings.
- `generate_puzzles.py`: Prompts the AI to create new puzzles and provides basic validation.
- `train.py` / `main.py`: Baseline model for move prediction (evaluation).
- `static/`: Web frontend for interacting with the generator.

## 🛠️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PunithaYaku/Chess-Puzzel-Gen.git
   cd Chess-Puzzel-Gen
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to use
1. **Prepare Data**: `python download_data.py`
2. **Train Generator**: `python train_gen.py`
3. **Generate Puzzles**: `python generate_puzzles.py`
4. **Run Web App**: `python app.py` (Local server at http://localhost:8000)

## 🐳 Docker Support
Build and run the generator using Docker:
```bash
docker build -t chess-puzzle-gen .
docker run -p 8000:8000 chess-puzzle-gen
```
