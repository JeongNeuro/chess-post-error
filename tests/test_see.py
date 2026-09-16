"""SEE 단위 테스트

킹을 재료 점수 0으로 두면 "가장 값싼 공격자" 선택에서 킹이 폰보다
먼저 뽑힌다. 그러면 폰이 있어도 킹이 되잡고, 그 킹이 0점짜리로 다시
잡히는 교환 연쇄가 만들어진다. test_king_recapture_illegal 이 그 경우다
(수정 전 −4, 정답 +1).
"""

import chess
import pytest

from src.see import see, VALUES, SEE_VALUES


CASES = [
    # (fen, uci, 기대값, 설명)
    ("3r3k/8/8/3P4/4K3/8/b7/8 b - - 0 1", "d8d5", 1,
     "Rxd5 — d5 는 흑 Ba2 가 다시 노리므로 Kxd5 는 불법. 폰 공짜"),
    ("3r3k/8/8/3N4/2P1K3/8/8/8 b - - 0 1", "d8d5", -2,
     "Rxd5 — 폰이 되잡는다. 룩5 주고 나이트3"),
    ("3rr2k/8/8/3N4/2P1K3/8/8/8 b - - 0 1", "d8d5", -2,
     "Rxd5 — 룩 둘, 폰+킹 방어. 폰이 먼저 쓰여야 한다"),
    ("4k3/8/5p2/4n3/8/3N4/8/4K3 w - - 0 1", "d3e5", 0,
     "Nxe5 — 등가 교환"),
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 1,
     "exd5 — 방어 없는 폰"),
    ("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1", "e4e5", 0,
     "e5 — 잡기가 아닌 수"),
    ("4k3/3r4/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 1,
     "Rxd5 — 룩 교환 후 폰 이득"),
]


@pytest.mark.parametrize("fen,uci,expected,desc", CASES)
def test_see_values(fen, uci, expected, desc):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert board.is_legal(move), f"불법 수: {uci} in {fen}"
    got = see(board, move)
    assert got == expected, f"{desc}: 기대 {expected}, 실제 {got}"


def test_king_recapture_illegal():
    """킹 되잡기가 불법인 국면에서 킹이 쓰이면 안 된다 (회귀 테스트)."""
    board = chess.Board("3r3k/8/8/3P4/4K3/8/b7/8 b - - 0 1")
    got = see(board, chess.Move.from_uci("d8d5"))
    assert got == 1, (
        f"킹을 최저가 공격자로 뽑는 버그가 되살아났다 (got={got}, 기대=1). "
        f"SEE_VALUES[KING] 이 큰 값인지 확인할 것.")


def test_king_is_last_resort_in_see_values():
    """SEE 내부에서 킹은 어떤 기물보다 비싸야 한다."""
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        assert SEE_VALUES[chess.KING] > SEE_VALUES[pt]


def test_material_values_exclude_king():
    """재료 계산용 VALUES 에서는 킹이 0이어야 한다.

    stage2v2.max_opponent_capture_see 가 'VALUES == 0 이면 표적에서 제외'
    규칙으로 킹을 거른다. SEE 수정 때 이쪽을 같이 바꾸면 안 된다.
    """
    assert VALUES[chess.KING] == 0
    assert VALUES[chess.PAWN] == 1
    assert VALUES[chess.KNIGHT] == 3
    assert VALUES[chess.BISHOP] == 3
    assert VALUES[chess.ROOK] == 5
    assert VALUES[chess.QUEEN] == 9


def test_en_passant():
    """앙파상은 목표 칸이 비어 있어 일반 잡기 탐색에 안 걸린다."""
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    mv = chess.Move.from_uci("e5d6")
    assert board.is_en_passant(mv)
    assert see(board, mv) == 1
