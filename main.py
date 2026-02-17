import chess
import torch
from pathlib import Path
from train import SimpleChessNet, fen_to_tensor

def main():
    """Entry point for testing the chess model."""
    print("Chess Puzzle Generator / Solver Initialized")
    
    model_path = Path("chess_puzzle_model.pth")
    
    # Check if we have a trained model
    if model_path.exists():
        model = SimpleChessNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Loaded trained model from {model_path}")
        
        # Example: evaluate starting position
        board = chess.Board()
        input_tensor = fen_to_tensor(board.fen())
        
        with torch.no_grad():
            from_logits, to_logits = model(torch.tensor(input_tensor).unsqueeze(0))
        
        from_sq = torch.argmax(from_logits).item()
        to_sq = torch.argmax(to_logits).item()
        
        move = chess.Move(from_sq, to_sq)
        print(f"Model suggests: {move}")
    else:
        print(f"No trained model found at {model_path}. Run 'python train.py' first.")
        board = chess.Board()
        print(f"Starting position:\n{board}")

if __name__ == "__main__":
    main()
