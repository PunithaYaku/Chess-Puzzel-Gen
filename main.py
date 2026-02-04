import chess
import torch
from train import SimpleChessNet, fen_to_tensor

def main():
    print("Chess Puzzle Generator / Solver Initialized")
    
    # Check if we have a trained model
    if torch.os.path.exists("chess_puzzle_model.pth"):
        model = SimpleChessNet()
        model.load_state_dict(torch.load("chess_puzzle_model.pth"))
        model.eval()
        print("Loaded trained model.")
        
        # Example: evaluate starting position
        board = chess.Board()
        input_tensor = fen_to_tensor(board.fen())
        from_logits, to_logits = model(torch.tensor(input_tensor).unsqueeze(0))
        
        from_sq = torch.argmax(from_logits).item()
        to_sq = torch.argmax(to_logits).item()
        
        move = chess.Move(from_sq, to_sq)
        print(f"Model suggests: {move}")
    else:
        print("No trained model found. Run 'python train.py' first.")
        board = chess.Board()
        print(f"Starting position:\n{board}")

if __name__ == "__main__":
    main()
