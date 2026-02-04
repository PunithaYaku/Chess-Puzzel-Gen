import torch
import torch.nn.functional as F
import chess
import random
from train_gen import CausalTransformer, FENCharset, VOCAB_SIZE

def sample_with_temperature(logits, temperature=1.0, top_k=10):
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def repair_fen(fen):
    parts = fen.split(" ")
    pos = parts[0]
    rows = pos.split("/")
    
    # Force exactly 8 rows
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
        
        if len(squares) > 8:
            squares = squares[:8]
        elif len(squares) < 8:
            squares.extend(['1'] * (8 - len(squares)))
        
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
    hm = '0'
    fm = '1'
    
    return f"{new_pos} {turn} {castling} {ep} {hm} {fm}"

def generate(n=1, max_retries=50):
    charset = FENCharset()
    model = CausalTransformer(VOCAB_SIZE)
    if torch.os.path.exists("fen_generator.pth"):
        model.load_state_dict(torch.load("fen_generator.pth"))
    model.eval()

    for attempt in range(max_retries):
        input_str = f"[{n}]:"
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
            if len(parts) < 2: continue
            
            raw_fen = parts[1].split(" ")[0].strip()
            # Try to repair before validating
            repaired_fen = repair_fen(raw_fen)
            if not repaired_fen: continue
            
            board = chess.Board(repaired_fen)
            print(f"\n[Success] Attempt {attempt+1}")
            print(f"Generated Raw: {raw_fen}")
            print(f"Repaired FEN: {repaired_fen}")
            print(board)
            return repaired_fen
        except:
            continue
    
    print(f"Failed to generate a valid FEN for Mate in {n} after {max_retries} attempts.")
    return None

if __name__ == "__main__":
    for n in range(1, 4):
        print(f"\n--- Searching for Mate in {n} ---")
        generate(n)
