"""
Static exchange evaluation (SEE)

Resolves the full sequence of captures on one square and returns the net
material outcome — Michie's (1961) swap-off value, the standard chess-engine
algorithm.

This implements the **preregistered** criterion for what the manuscript calls
Event B. The reported analysis does not use it; it is kept for the
supplementary comparison (see docs/definitions.md). It was chosen originally
because it needs no arbitrary window such as "recaptured within N moves".

Limitation: it does not account for pins, overloaded defenders, or other
indirect effects.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE
#   - This is the preregistered definition, which the manuscript withdrew.
#     It is needed only to reproduce the pre-revision result, and is kept
#     for the supplementary comparison.
#   - Why this criterion fails to exclude even exchanges: docs/definitions.md.
#   - Two properties to confirm: the king is considered LAST as a recapturer,
#     never first, and it cannot recapture into a defended square. Both were
#     wrong originally; tests/test_see.py holds the position that exposed it.
# ───────────────────────────────────────────────────────────

import chess

# Piece values for material accounting. The king is 0 because it is never
# captured: mat_diff and max_opponent_capture_see both rely on the rule
# "VALUES == 0 means exclude from the targets".
VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# Values used inside SEE only. The king is given a large value so that it
# comes last when the cheapest attacker is chosen.
#
# Using VALUES[KING] = 0 here would make the king a cheaper attacker than a
# pawn, producing an exchange sequence in which
#   (1) the king recaptures even when a pawn could, and
#   (2) that king is then captured as a zero-value piece.
# In `3r3k/8/8/3P4/4K3/8/b7/8 b - -`, Rxd5 should be +1 and came out as -4.
# See tests/test_see.py.
SEE_VALUES = dict(VALUES)
SEE_VALUES[chess.KING] = 10_000


def _least_valuable(board, occupied, attackers_mask, color):
    """Square of the cheapest attacker, or None. The king comes last."""
    best, best_v = None, 1 << 30
    for sq in chess.scan_forward(attackers_mask):
        p = board.piece_at(sq)
        if p is None or p.color != color:
            continue
        v = SEE_VALUES[p.piece_type]
        if v < best_v:
            best, best_v = sq, v
    return best


def _attackers_to(board, occupied, square):
    """Every piece attacking `square` given `occupied`, X-rays included."""
    mask = 0
    for color in (chess.WHITE, chess.BLACK):
        mask |= board.attackers_mask(color, square) & occupied
    # Sliding-piece X-ray: recompute attackers behind a removed piece.
    for sq in chess.scan_forward(occupied):
        p = board.piece_at(sq)
        if p is None or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        if sq in chess.scan_forward(mask):
            continue
        between = chess.between(sq, square)
        if between and not (between & occupied):
            if square in board.attacks(sq) or _aligned(sq, square, p.piece_type):
                mask |= chess.BB_SQUARES[sq]
    return mask


def _aligned(a, b, ptype):
    fa, ra = chess.square_file(a), chess.square_rank(a)
    fb, rb = chess.square_file(b), chess.square_rank(b)
    diag = abs(fa - fb) == abs(ra - rb)
    line = (fa == fb) or (ra == rb)
    if ptype == chess.BISHOP:
        return diag
    if ptype == chess.ROOK:
        return line
    return diag or line


def see(board, move):
    """
    Net gain from resolving the exchange sequence on the target square after
    `move`, from the mover's point of view. Positive is a gain, zero an even
    trade, negative a loss of material.
    """
    to_sq = move.to_square
    victim = board.piece_at(to_sq)
    if board.is_en_passant(move):
        gain0 = VALUES[chess.PAWN]
    elif victim is None:
        gain0 = 0
    else:
        gain0 = VALUES[victim.piece_type]

    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return 0
    att_val = VALUES[attacker.piece_type]

    occupied = board.occupied & ~chess.BB_SQUARES[move.from_square]
    if board.is_en_passant(move):
        cap_sq = to_sq + (-8 if attacker.color == chess.WHITE else 8)
        occupied &= ~chess.BB_SQUARES[cap_sq]

    gains = [gain0]
    side = not attacker.color
    cur_att_val = att_val
    d = 0

    while True:
        mask = _attackers_to(board, occupied, to_sq) & occupied
        src = _least_valuable(board, occupied, mask, side)
        if src is None:
            break
        p = board.piece_at(src)
        if p.piece_type == chess.KING:
            # Recapturing with the king requires the square to be undefended.
            # If an enemy attacker remains the capture is illegal, so the
            # exchange ends here.
            rest = occupied & ~chess.BB_SQUARES[src]
            rest_mask = _attackers_to(board, rest, to_sq) & rest
            if _least_valuable(board, rest, rest_mask, not side) is not None:
                break
        d += 1
        gains.append(cur_att_val - gains[d - 1])
        cur_att_val = SEE_VALUES[p.piece_type]
        occupied &= ~chess.BB_SQUARES[src]
        side = not side
        if d > 31:
            break

    # Fold backwards: either side may stop the exchange at any point.
    while d > 0:
        gains[d - 1] = -max(-gains[d - 1], gains[d])
        d -= 1
    return gains[0]


def material_loss_events(mt_sans, color, opening_cut=10):
    """
    Detect Event C as preregistered (the manuscript's Event B, withdrawn).

    Returns [(ply_index, lost_value, san), ...]
      ply_index : 0-based ply
      lost_value: value of the material lost (event size, continuous)

    Rule: if the opponent's reply has SEE > 0 after the player's move, the
          player has lost material on net -- that is, the move left a
          profitable capture available to the opponent.
    """
    board = chess.Board()
    out = []
    for i, san in enumerate(mt_sans):
        try:
            mv = board.parse_san(san)
        except Exception:
            break
        # Immediately after the player's move, is the opponent's best
        # capture profitable?
        if i % 2 == color and i >= opening_cut:
            board.push(mv)
            best = 0
            for opp in board.legal_moves:
                if not board.is_capture(opp):
                    continue
                v = see(board, opp)
                if v > best:
                    best = v
            if best > 0:
                out.append((i, best, san))
            board.pop()
        board.push(mv)
    return out
