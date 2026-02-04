import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import chess
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- Constants ---
SQUARES = 64
PIECE_TYPES = 12 # 6 types * 2 colors

# --- Helpers ---
def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Piece type (1-6) + offset for color (0 or 6)
            p_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            row = square // 8
            col = square % 8
            tensor[p_idx, row, col] = 1.0
    return tensor

def move_to_idx(move_uci):
    move = chess.Move.from_uci(move_uci)
    return move.from_square, move.to_square

class ChessPuzzleDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        fen = self.df.iloc[idx]['FEN']
        # The Moves column contains a space-separated list. We take the first one as the target.
        moves = self.df.iloc[idx]['Moves'].split()
        target_move = moves[0]
        
        input_tensor = fen_to_tensor(fen)
        from_idx, to_idx = move_to_idx(target_move)
        
        return torch.tensor(input_tensor), torch.tensor(from_idx), torch.tensor(to_idx)

# --- Model ---
class SimpleChessNet(nn.Module):
    def __init__(self):
        super(SimpleChessNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_from = nn.Linear(128 * 8 * 8, 64)
        self.fc_to = nn.Linear(128 * 8 * 8, 64)
        
    def forward(self, x):
        features = self.conv(x)
        from_logits = self.fc_from(features)
        to_logits = self.fc_to(features)
        return from_logits, to_logits

# --- Training Loop ---
def train():
    device = torch.device("cpu") # use cpu for now
    print(f"Using device: {device}")
    
    dataset = ChessPuzzleDataset("data/puzzles_subset.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SimpleChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(1, 4): # Just 3 epochs for demonstration
        total_loss = 0
        for boards, from_labels, to_labels in dataloader:
            boards = boards.to(device)
            from_labels = from_labels.to(device)
            to_labels = to_labels.to(device)
            
            optimizer.zero_grad()
            from_pred, to_pred = model(boards)
            
            loss = criterion(from_pred, from_labels) + criterion(to_pred, to_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "chess_puzzle_model.pth")
    print("Model saved to chess_puzzle_model.pth")

if __name__ == "__main__":
    import os
    if os.path.exists("data/puzzles_subset.csv"):
        train()
    else:
        print("Data subset not found. Please run download_data.py first.")
