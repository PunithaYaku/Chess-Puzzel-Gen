import pytest
import torch
import chess
from app import repair_fen, sample_with_temperature
from train_gen import FENCharset

def test_repair_fen_valid():
    raw_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    repaired = repair_fen(raw_fen)
    assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1" == repaired
    assert chess.Board(repaired).is_valid()

def test_repair_fen_missing_rows():
    # Only 5 rows provided, should expand to 8
    raw_fen = "4k3/8/8/8/4K3"
    repaired = repair_fen(raw_fen)
    parts = repaired.split(" ")
    pos_parts = parts[0].split("/")
    assert len(pos_parts) == 8
    # python-chess is_valid checks for Kings
    assert chess.Board(repaired).is_valid()

def test_repair_fen_too_many_rows():
    # 10 rows provided, should truncate to 8
    # Has both kings: K (white) and k (black)
    raw_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/8/8"
    repaired = repair_fen(raw_fen)
    parts = repaired.split(" ")
    pos_parts = parts[0].split("/")
    assert len(pos_parts) == 8
    assert chess.Board(repaired).is_valid()

def test_repair_fen_invalid_squares():
    # Row with more than 8 squares (e.g., 9) and less than 8 squares
    # Should be normalized to 8 per row
    raw_fen = "4k4/8/8/8/8/8/8/4K2"
    repaired = repair_fen(raw_fen)
    parts = repaired.split(" ")
    pos_parts = parts[0].split("/")
    for row in pos_parts:
        # Check that each row expands to 8 squares
        board_row = ""
        for char in row:
            if char.isdigit():
                board_row += '1' * int(char)
            else:
                board_row += char
        assert len(board_row) == 8
    assert chess.Board(repaired).is_valid()

def test_sample_with_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sample = sample_with_temperature(logits, temperature=0.1, top_k=2)
    assert sample.item() in [3, 4] # top-k filtering should limit to highest logits

def test_charset_encoding():
    charset = FENCharset()
    original = "rnbqk1nr/pppp1ppp/8/4p3/1b1P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 1 3"
    encoded = charset.encode(original)
    decoded = charset.decode(encoded)
    assert original == decoded
def test_repair_fen_extra_spaces():
    # FEN with extra spaces between parts
    raw_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR   w   KQkq   -   0   1"
    repaired = repair_fen(raw_fen)
    assert "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" == repaired

def test_repair_fen_empty():
    # Empty string should at least return a valid board structure
    raw_fen = ""
    repaired = repair_fen(raw_fen)
    assert chess.Board(repaired).is_valid()
