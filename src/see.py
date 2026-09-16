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

# 재료 계산용 기물 점수. 킹은 잡히지 않으므로 0 — stage2v2 의 mat_diff 와
# max_opponent_capture_see 가 "VALUES == 0 이면 표적에서 제외" 규칙으로 쓴다.
VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# SEE 내부 전용 점수. 킹은 "가장 값싼 공격자" 선택에서 반드시 마지막에
# 와야 하므로 큰 값을 준다.
#
# VALUES[KING] = 0 을 그대로 쓰면 킹이 폰보다 값싼 공격자로 뽑혀
#   (1) 폰이 있어도 킹이 먼저 되잡고
#   (2) 그 킹이 0점짜리로 다시 잡히는
# 교환 연쇄가 만들어진다. 실제로 `3r3k/8/8/3P4/4K3/8/b7/8 b - -` 의
# Rxd5 는 정답이 +1 인데 −4 가 나왔다. tests/test_see.py 참조.
SEE_VALUES = dict(VALUES)
SEE_VALUES[chess.KING] = 10_000


def _least_valuable(board, occupied, attackers_mask, color):
    """공격자 중 가장 값싼 기물의 square 반환 (없으면 None). 킹은 맨 마지막."""
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
    """occupied 상태에서 square를 공격하는 모든 기물 (X-ray 갱신 포함)"""
    mask = 0
    for color in (chess.WHITE, chess.BLACK):
        mask |= board.attackers_mask(color, square) & occupied
    # 슬라이딩 기물 X-ray: 제거된 기물 뒤의 공격자 재계산
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
    move를 둔 뒤 해당 칸에서의 교환 연쇄를 해소한 순손익 (수를 두는 쪽 관점).
    양수 = 이득, 0 = 균형 교환, 음수 = 재료 손실.
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
            # 킹으로 되잡으려면 그 칸이 더 이상 방어되지 않아야 한다.
            # 상대 공격자가 남아 있으면 불법이므로 교환은 여기서 끝난다.
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

    # 역방향 축약: 각 시점에서 교환을 멈출 수 있음 (stand-pat)
    while d > 0:
        gains[d - 1] = -max(-gains[d - 1], gains[d])
        d -= 1
    return gains[0]


def material_loss_events(mt_sans, color, opening_cut=10):
    """
    사전등록판 사건 C 탐지 (= 논문의 사건 B, 철회된 정의).

    반환: [(ply_index, lost_value, san), ...]
      ply_index : 0-based 플라이
      lost_value: 잃은 기물 가치 (사건 크기, 연속 변수)

    판정: 플레이어가 둔 수 이후, 상대가 두는 수의 SEE > 0 이면
          플레이어가 재료를 순손실한 것.
          즉 "상대가 이득 보는 잡기를 할 수 있는 상태를 만들었다".
    """
    board = chess.Board()
    out = []
    for i, san in enumerate(mt_sans):
        try:
            mv = board.parse_san(san)
        except Exception:
            break
        # 플레이어가 둔 수 직후, 상대의 최선 잡기가 이득인지 확인
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
