import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys

# --- Configurations ---
CHARS = "12345678/pnbrqkPNBRQKw bkqKQ- :0123456789[]"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
MAX_LEN = 120 

class FENCharset:
    def __init__(self):
        self.chars = CHARS
        self.char_to_idx = CHAR_TO_IDX
        self.idx_to_char = IDX_TO_CHAR
        self.vocab_size = VOCAB_SIZE

    def encode(self, s):
        return [self.char_to_idx.get(c, self.char_to_idx[' ']) for c in s]

    def decode(self, ids):
        return "".join([self.idx_to_char[i] for i in ids])

class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        output = self.transformer_encoder(x, mask=mask, is_causal=True)
        return self.fc_out(output)

# --- Training ---
def train_gen():
    device = torch.device("cpu")
    charset = FENCharset()
    
    all_fens = []
    print("Loading data for training...", flush=True)
    for n in range(1, 6):
        path = f"data/mate_in_{n}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Standard formatting: [N]:FEN
            fens = df['FEN'].apply(lambda x: f"[{n}]:{x} ").tolist()
            all_fens.extend(fens[:15000]) 
    
    print(f"Total labeled FENs: {len(all_fens)}", flush=True)
    
    def get_batch(batch_size=32):
        indices = np.random.choice(len(all_fens), batch_size)
        batch_fens = [all_fens[i] for i in indices]
        
        x_data = []
        y_data = []
        for fen in batch_fens:
            encoded = charset.encode(fen)
            if len(encoded) > MAX_LEN:
                encoded = encoded[:MAX_LEN]
            
            # Predict next char
            x_data.append(encoded[:-1])
            y_data.append(encoded[1:])
        
        # Padding
        max_batch_len = max(len(row) for row in x_data)
        x_padded = [row + [charset.char_to_idx[' ']] * (max_batch_len - len(row)) for row in x_data]
        y_padded = [row + [charset.char_to_idx[' ']] * (max_batch_len - len(row)) for row in y_data]
            
        return torch.tensor(x_padded).to(device), torch.tensor(y_padded).to(device)

    model = CausalTransformer(VOCAB_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    print("Training generator...", flush=True)
    for step in range(1, 1501):
        x, y = get_batch()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}", flush=True)
            
        if step % 500 == 0:
            torch.save(model.state_dict(), "fen_generator.pth")
            print(f"Checkpoint saved at step {step}", flush=True)
            
    torch.save(model.state_dict(), "fen_generator.pth")
    print("Generator model saved to fen_generator.pth", flush=True)

if __name__ == "__main__":
    train_gen()
