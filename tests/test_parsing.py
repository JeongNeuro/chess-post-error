"""Unit tests for PGN parsing, win-probability conversion, and matching."""

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
    assert 0 not in white           # the first move has no preceding clock
    assert white[2] == 600 - 592    # Nf3
    assert white[4] == 592 - 580    # Bb5
    assert black[3] == 600 - 585    # Nc6
    assert black[5] == 585 - 570    # a6


def test_move_times_drop_negative():
    """A negative move time is a recording error and is dropped."""
    assert move_times([100, 100, 120, 100], 0) == {}


def test_move_times_skip_missing_clock():
    """One missing clock reading costs two move times.

    Without clk(i) both that move (no difference can be taken) and the
    player's next move (no preceding reading) are lost. This is why games
    with missing clocks lose more observations than one would expect.
    """
    assert move_times([100, 100, None, 100, 80, 100], 0) == {}
    # With nothing missing, both survive.
    assert move_times([100, 100, 90, 100, 80, 100], 0) == {2: 10, 4: 10}


def test_winprob_is_lichess_formula_normalised():
    """The 0-100 formula in the docstring and the 0-1 implementation agree."""
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
    """Omitting the sign flip loses every blunder by Black."""
    evals = [100.0, 100.0]
    w = player_winprobs(evals, 0)
    b = player_winprobs(evals, 1)
    assert w[0] > 0.5 and b[0] < 0.5
    assert w[0] + b[0] == pytest.approx(1.0)


# -- Matching ------------------------------------------------

class Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_caliper_mask_strict_drops_event_with_missing_var():
    """★ Former bug: a missing variable was skipped and matching went ahead
    on the rest."""
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
    """With several events per player, the clustered SE exceeds the
    event-level one."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(20):
        base = rng.normal(0, 1.0)           # player effect
        for _ in range(50):
            rows.append({"player": f"p{p}", "v": base + rng.normal(0, 0.3)})
    df = pd.DataFrame(rows)
    st = cluster_stats(df, "v")
    assert st["n_players"] == 20
    assert st["n_events"] == 1000
    assert st["se"] > df["v"].sem() * 2
