from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import chess
import random
import os
from train_gen import CausalTransformer, FENCharset, VOCAB_SIZE
import torch.nn.functional as F

app = FastAPI(title="AI Chess Puzzle Generator API")

# Global model and charset initialization
device = torch.device("cpu")
charset = FENCharset()
model = CausalTransformer(VOCAB_SIZE).to(device)

# Load pre-trained weights if available
if os.path.exists("fen_generator.pth"):
    model.load_state_dict(torch.load("fen_generator.pth", map_location=device))
model.eval()

def sample_with_temperature(logits, temperature=1.0, top_k=10):
    """
    Samples a token from the distribution with temperature scaling and top-k filtering.
    """
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def repair_fen(fen):
    """
    Basic FEN repair logic to ensure the generated string is a valid board state.
    Ensures 8 rows and 8 squares per row, then appends default game state values.
    """
    parts = fen.split(" ")
    pos = parts[0]
    rows = pos.split("/")
    
    # Ensure exactly 8 rows
    if len(rows) > 8:
        rows = rows[:8]
    elif len(rows) < 8:
        rows.extend(['8'] * (8 - len(rows)))
    
    new_rows = []
    for row in rows:
        squares = []
        for char in row:
            if char.isdigit():
                squares.extend(['1'] * int(char))
            else:
                squares.append(char)
        
        # Ensure exactly 8 squares per row
        if len(squares) > 8:
            squares = squares[:8]
        elif len(squares) < 8:
            squares.extend(['1'] * (8 - len(squares)))
        
        # Convert back to shorthand (e.g., 111 -> 3)
        new_row = ""
        count = 0
        for s in squares:
            if s == '1':
                count += 1
            else:
                if count > 0:
                    new_row += str(count)
                    count = 0
                new_row += s
        if count > 0:
            new_row += str(count)
        new_rows.append(new_row)
    
    new_pos = "/".join(new_rows)
    turn = parts[1] if len(parts) > 1 and parts[1] in ['w', 'b'] else 'w'
    castling = parts[2] if len(parts) > 2 else '-'
    ep = parts[3] if len(parts) > 3 else '-'
    return f"{new_pos} {turn} {castling} {ep} 0 1"

@app.get("/generate/{mate_in}")
async def generate_puzzle_api(mate_in: int):
    """
    Generates a new chess puzzle with a target 'mate in N' difficulty.
    Retries up to 20 times to produce a valid chess board.
    """
    max_retries = 20
    for attempt in range(max_retries):
        input_str = f"[{mate_in}]:"
        input_ids = charset.encode(input_str)
        
        with torch.no_grad():
            for _ in range(120): 
                x = torch.tensor([input_ids])
                logits = model(x)
                last_logit = logits[0, -1, :]
                
                # Sample next token
                next_id = sample_with_temperature(last_logit.unsqueeze(0), temperature=0.7, top_k=5).item()
                input_ids.append(next_id)
                
                # Stop if end token (space) is reached
                if charset.idx_to_char[next_id] == ' ' and len(input_ids) > 20:
                    break
                
        full_output = charset.decode(input_ids)
        try:
            parts = full_output.split("]:")
            if len(parts) < 2: continue
            
            raw_fen = parts[1].split(" ")[0].strip()
            repaired_fen = repair_fen(raw_fen)
            
            # Use python-chess to validate the FEN
            board = chess.Board(repaired_fen)
            
            return {
                "fen": repaired_fen,
                "mate_in": mate_in,
                "attempt": attempt + 1
            }
        except Exception:
            # Continue to next retry on validation error
            continue
    
    raise HTTPException(status_code=500, detail="Failed to generate valid FEN after multiple attempts")

# Serve static frontend files
if not os.path.exists("static"):
    os.makedirs("static")

# Mount the static directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start the development server
    uvicorn.run(app, host="0.0.0.0", port=8000)
