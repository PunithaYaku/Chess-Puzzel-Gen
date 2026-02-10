import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
import chess
import random
import os
from train_gen import CausalTransformer, FENCharset, VOCAB_SIZE
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("chess-ai")

app = FastAPI(title="AI Chess Puzzle Generator API")

# Global model and charset initialization
device = torch.device("cpu")
charset = FENCharset()
model = CausalTransformer(VOCAB_SIZE).to(device)

# Load pre-trained weights if available
if os.path.exists("fen_generator.pth"):
    logger.info("Loading pre-trained model weights...")
    model.load_state_dict(torch.load("fen_generator.pth", map_location=device))
else:
    logger.warning("fen_generator.pth not found. Model will start with random weights.")

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

def find_mate_sequence(board, mate_in):
    """
    Finds a forced mate sequence for the current side to move within N moves (2N-1 plies).
    Returns a list of UCI moves if found, else None.
    """
    # If the side not to move is already in check, it's an illegal FEN for a start position
    if board.was_into_check():
        return None
    
    # If already checkmate, return empty (though usually we want to generate a new one)
    if board.is_checkmate():
        return None

    # Mate in N means at most 2*N - 1 plies
    max_plies = 2 * mate_in - 1
    
    def search(b, plies_left):
        if b.is_checkmate():
            return []
        if plies_left <= 0:
            return None

        best_line = None
        
        # Side to move (Player A)
        for move in b.legal_moves:
            b.push(move)
            if b.is_checkmate():
                b.pop()
                return [move.uci()]
            
            if plies_left > 1:
                # Opponent (Player B) moves
                all_responses_mate = True
                shortest_response_line = None
                
                legal_responses = list(b.legal_moves)
                if not legal_responses: # Stalemate
                    all_responses_mate = False
                else:
                    for opp_move in legal_responses:
                        b.push(opp_move)
                        line = search(b, plies_left - 2)
                        b.pop()
                        
                        if line is None:
                            all_responses_mate = False
                            break
                        
                        # We want the shortest mate to be the "main line" shown to user
                        if shortest_response_line is None or len(line) < len(shortest_response_line):
                            shortest_response_line = [opp_move.uci()] + line
                
                if all_responses_mate:
                    current_line = [move.uci()] + (shortest_response_line or [])
                    if best_line is None or len(current_line) < len(best_line):
                        best_line = current_line
            b.pop()
        return best_line

    return search(board, max_plies)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "model_loaded": os.path.exists("fen_generator.pth")}

@app.get("/generate/{mate_in}")
async def generate_puzzle_api(mate_in: int):
    """
    Generates a new chess puzzle with a target 'mate in N' difficulty.
    Retries until a valid puzzle with a guaranteed solution is found.
    """
    logger.info(f"Puzzle generation requested: Mate in {mate_in}")
    max_retries = 100
    for attempt in range(max_retries):
        input_str = f"[{mate_in}]:"
        input_ids = charset.encode(input_str)
        
        with torch.no_grad():
            for _ in range(120): 
                x = torch.tensor([input_ids])
                logits = model(x)
                last_logit = logits[0, -1, :]
                next_id = sample_with_temperature(last_logit.unsqueeze(0), temperature=0.7, top_k=5).item()
                input_ids.append(next_id)
                if charset.idx_to_char[next_id] == ' ' and len(input_ids) > 20:
                    break
                
        full_output = charset.decode(input_ids)
        try:
            parts = full_output.split("]:")
            if len(parts) < 2: 
                continue
            
            raw_fen = parts[1].split(" ")[0].strip()
            repaired_fen = repair_fen(raw_fen)
            board = chess.Board(repaired_fen)
            
            # Verify if it actually has a mate in N
            solution = find_mate_sequence(board, mate_in)
            if solution is None:
                logger.debug(f"Attempt {attempt + 1} failed solver verification. Retrying...")
                continue
            
            logger.info(f"Successfully generated puzzle with solution on attempt {attempt + 1}")
            return {
                "fen": repaired_fen,
                "solution": solution,
                "mate_in": mate_in,
                "attempt": attempt + 1
            }
        except Exception as e:
            logger.debug(f"Attempt {attempt + 1} error: {str(e)}")
            continue
    
    raise HTTPException(status_code=500, detail=f"Failed to produce a Mate in {mate_in} puzzle after {max_retries} attempts.")

# Serve static frontend files
if not os.path.exists("static"):
    os.makedirs("static")

# Mount the static directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Chess Puzzle Generator API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

