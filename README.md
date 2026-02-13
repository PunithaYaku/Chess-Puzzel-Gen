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

## 🧠 Technical Architecture

The system uses a **Causal Transformer** architecture specifically adapted for chess notation.

1.  **Tokenization**: FEN strings are broken down into a vocabulary of 50+ characters (pieces, numbers, separators).
2.  **Context**: The model prepends a prompt like `[3]:` to signify a request for a "Mate in 3" puzzle.
3.  **Generation**: The transformer predicts the next character in the FEN sequence auto-regressively.
4.  **Post-Processing**: A robust repair algorithm ensures the generated string follows legal FEN structural rules (8x8 grid).
5.  **Validation**: `python-chess` validates the final board state before serving it to the user.

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

## 🛠️ Troubleshooting

- **PowerShell Script Execution**: If you get a "running scripts is disabled" error on Windows, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **SRE Module Mismatch**: If you see `AssertionError: SRE module mismatch`, it's likely due to conflicting `PYTHONPATH` or `PYTHONHOME` environment variables. Clear them before running:
  ```powershell
  $env:PYTHONPATH = ""
  $env:PYTHONHOME = ""
  ```
- **Model Weights Missing**: If `fen_generator.pth` is missing, the API will still run but may generate nonsensical FEN strings. Ensure you run `train_gen.py` first.
- **Port Conflicts**: Ensure port 8000 is not being used by another application.
- **Memory Issues**: The model runs on CPU by default. If you encounter crashes, ensure your system has at least 4GB of free RAM.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started, our branching policy, and code style.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

