"""PGN 파싱·승률 변환·매칭 단위 테스트"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import caliper_mask, cluster_stats, window_mean
from src.extract import (K, move_times, parse_eval, parse_movetext,
                        player_winprobs, winprob)

MT = ("1. e4 { [%eval 0.17] [%clk 0:10:00] } "
      "1... e5 { [%eval 0.24] [%clk 0:10:00] } "
      "2. Nf3 { [%eval 0.21] [%clk 0:09:52] } "
      "2... Nc6?! { [%eval 0.55] [%clk 0:09:45] } "
      "3. Bb5 { [%eval 0.48] [%clk 0:09:40] } "
      "3... a6?? { [%eval 2.90] [%clk 0:09:30] }")


def test_parse_movetext_aligns_by_ply():
    clks, evals, sans, nags = parse_movetext(MT)
    assert sans == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert clks == [600, 600, 592, 585, 580, 570]
    assert evals == pytest.approx([17.0, 24.0, 21.0, 55.0, 48.0, 290.0])
    assert nags == [None, None, None, "inaccuracy", None, "blunder"]


def test_move_times_use_same_player_consecutive_clocks():
    clks, _, _, _ = parse_movetext(MT)
    white = move_times(clks, 0)
    black = move_times(clks, 1)
    assert 0 not in white           # 첫 수는 정의 불가
    assert white[2] == 600 - 592    # Nf3
    assert white[4] == 592 - 580    # Bb5
    assert black[3] == 600 - 585    # Nc6
    assert black[5] == 585 - 570    # a6


def test_move_times_drop_negative():
    """음수 소요시간(기록 오류)은 버린다."""
    assert move_times([100, 100, 120, 100], 0) == {}


def test_move_times_skip_missing_clock():
    """시계 하나가 없으면 두 수의 소요시간이 사라진다.

    clk(i) 가 없으면 그 수(diff 계산 불가)와 바로 다음 자기 수
    (이전 clk 가 없음)가 모두 빠진다. 결측이 있는 대국에서 관측수가
    생각보다 많이 줄어드는 이유다.
    """
    assert move_times([100, 100, None, 100, 80, 100], 0) == {}
    # 결측이 없으면 둘 다 살아난다
    assert move_times([100, 100, 90, 100, 80, 100], 0) == {2: 10, 4: 10}


def test_winprob_is_lichess_formula_normalised():
    """docstring 의 0~100 식과 구현의 0~1 식이 같은지."""
    for cp in (-500.0, -1.0, 0.0, 1.0, 500.0):
        lichess = 50 + 50 * (2 / (1 + np.exp(-K * cp)) - 1)
        assert winprob(cp) == pytest.approx(lichess / 100)
    assert winprob(0.0) == pytest.approx(0.5)


def test_mate_scores_saturate():
    assert parse_eval("#3") == 10000.0
    assert parse_eval("#-3") == -10000.0
    assert winprob(parse_eval("#3")) > 0.999
    assert winprob(parse_eval("#-3")) < 0.001


def test_black_perspective_is_flipped():
    """부호 반전을 빠뜨리면 흑의 블런더를 전부 놓친다."""
    evals = [100.0, 100.0]
    w = player_winprobs(evals, 0)
    b = player_winprobs(evals, 1)
    assert w[0] > 0.5 and b[0] < 0.5
    assert w[0] + b[0] == pytest.approx(1.0)


# ── 매칭 ────────────────────────────────────────────────────

class Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_caliper_mask_strict_drops_event_with_missing_var():
    """★ 예전 버그: 결측 변수는 건너뛰고 나머지로 매칭했다."""
    cand = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 1.0]})
    cal = {"a": 0.5, "b": 0.5}
    _, ok = caliper_mask(Row(a=1.0, b=np.nan), cand, cal, strict=True)
    assert ok is False
    m, ok = caliper_mask(Row(a=1.0, b=np.nan), cand, cal, strict=False)
    assert ok is True and m.tolist() == [True, False]


def test_caliper_mask_all_vars_must_be_within():
    cand = pd.DataFrame({"a": [1.0, 1.0], "b": [1.0, 9.0]})
    m, ok = caliper_mask(Row(a=1.0, b=1.0), cand, {"a": 0.5, "b": 0.5})
    assert ok and m.tolist() == [True, False]


def test_window_mean_steps_by_two_plies():
    zmap = {("g", "p", 12): 1.0, ("g", "p", 14): 2.0, ("g", "p", 16): 3.0}
    v, n = window_mean(zmap, "g", "p", 10, window=3, sign=+1)
    assert v == pytest.approx(2.0) and n == 3
    v, n = window_mean(zmap, "g", "p", 18, window=3, sign=-1)
    assert v == pytest.approx(2.0) and n == 3


def test_cluster_se_is_larger_than_event_se():
    """플레이어당 사건이 여럿이면 클러스터 SE 가 사건 단위보다 크다."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(20):
        base = rng.normal(0, 1.0)           # 플레이어 효과
        for _ in range(50):
            rows.append({"player": f"p{p}", "v": base + rng.normal(0, 0.3)})
    df = pd.DataFrame(rows)
    st = cluster_stats(df, "v")
    assert st["n_players"] == 20
    assert st["n_events"] == 1000
    assert st["se"] > df["v"].sem() * 2
