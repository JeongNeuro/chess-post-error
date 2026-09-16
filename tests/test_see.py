"""Unit tests for SEE.

Giving the king a material value of 0 makes it cheaper than a pawn when the
cheapest attacker is chosen. The king then recaptures even when a pawn could,
and is itself captured as a zero-value piece, producing a bogus exchange
sequence. test_king_recapture_illegal covers that case (-4 before the fix,
+1 correct).
"""

import chess
import pytest

from src.see import see, VALUES, SEE_VALUES


CASES = [
    # (fen, uci, expected, description)
    ("3r3k/8/8/3P4/4K3/8/b7/8 b - - 0 1", "d8d5", 1,
     "Rxd5 - Black Ba2 covers d5, so Kxd5 is illegal. The pawn is free"),
    ("3r3k/8/8/3N4/2P1K3/8/8/8 b - - 0 1", "d8d5", -2,
     "Rxd5 - the pawn recaptures: rook (5) for knight (3)"),
    ("3rr2k/8/8/3N4/2P1K3/8/8/8 b - - 0 1", "d8d5", -2,
     "Rxd5 - two rooks against pawn and king; the pawn must be used first"),
    ("4k3/8/5p2/4n3/8/3N4/8/4K3 w - - 0 1", "d3e5", 0,
     "Nxe5 - an even trade"),
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 1,
     "exd5 - an undefended pawn"),
    ("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1", "e4e5", 0,
     "e5 - not a capture"),
    ("4k3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 1,
     "Rxd5 - a pawn ahead after the rooks come off"),
]


@pytest.mark.parametrize("fen,uci,expected,desc", CASES)
def test_see_values(fen, uci, expected, desc):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert board.is_legal(move), f"illegal move: {uci} in {fen}"
    got = see(board, move)
    assert got == expected, f"{desc}: expected {expected}, got {got}"


def test_king_recapture_illegal():
    """The king must not be used where recapturing with it is illegal."""
    board = chess.Board("3r3k/8/8/3P4/4K3/8/b7/8 b - - 0 1")
    got = see(board, chess.Move.from_uci("d8d5"))
    assert got == 1, (
        f"the king is being chosen as the cheapest attacker again "
        f"(got={got}, expected 1). Check that SEE_VALUES[KING] is large.")


def test_king_is_last_resort_in_see_values():
    """Inside SEE the king must be dearer than any other piece."""
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        assert SEE_VALUES[chess.KING] > SEE_VALUES[pt]


def test_material_values_exclude_king():
    """In VALUES, used for material accounting, the king must be 0.

    max_opponent_capture_see excludes the king by the rule "VALUES == 0
    means exclude from the targets". Fixing SEE must not change this one.
    """
    assert VALUES[chess.KING] == 0
    assert VALUES[chess.PAWN] == 1
    assert VALUES[chess.KNIGHT] == 3
    assert VALUES[chess.BISHOP] == 3
    assert VALUES[chess.ROOK] == 5
    assert VALUES[chess.QUEEN] == 9


def test_en_passant():
    """En passant leaves the target square empty, so an ordinary capture
    search misses it."""
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    mv = chess.Move.from_uci("e5d6")
    assert board.is_en_passant(mv)
    assert see(board, mv) == 1
