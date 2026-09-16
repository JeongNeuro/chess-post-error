"""Unit tests for building the analysis table - net_mat, z_pre3, events.

z_pre3 must be taken **by ply index**. An earlier implementation used
groupby().shift(), which works by row position: when the measurement floor
(move_time < 1s) removes an intervening move, "the preceding three moves"
silently reaches further back than it should.
test_z_pre3_skips_missing_ply covers that case.
"""

import numpy as np
import pandas as pd
import pytest

from src.prepare import add_pre_speed, net_material_change
from src.prepare import events_A, events_B, events_B_see, mask_A, mask_B


def frame(plies, z=None, mat=None, player="p1", game="g1"):
    n = len(plies)
    return pd.DataFrame({
        "game_id": [game] * n,
        "player": [player] * n,
        "ply": plies,
        "z": np.arange(n, dtype=float) if z is None else z,
        "mat_diff": np.zeros(n) if mat is None else mat,
    })


# ── net_mat ─────────────────────────────────────────────────

def test_net_mat_is_next_own_move():
    d = frame([10, 12, 14, 16], mat=[0.0, -1.0, -1.0, -4.0])
    out = net_material_change(d)
    # ply10: mat(12) - mat(10) = -1 ; ply12: mat(14)-mat(12) = 0
    assert out.net_mat.tolist()[:3] == [-1.0, 0.0, -3.0]
    assert np.isnan(out.net_mat.iloc[-1])   # no following move by this player


def test_even_exchange_is_not_an_event():
    """An even trade nets to zero and is not an event. This is where the
    definition differs from the withdrawn SEE one."""
    d = frame([10, 12], mat=[0.0, 0.0])
    out = net_material_change(d)
    assert out.net_mat.iloc[0] == 0.0
    assert not mask_B(out).iloc[0]


def test_net_mat_gap_in_plies_gives_nan():
    """When ply+2 is absent the search does not reach past it."""
    d = frame([10, 14], mat=[0.0, -3.0])
    out = net_material_change(d)
    assert np.isnan(out.net_mat.iloc[0])


def test_net_mat_separates_players_and_games():
    a = frame([10, 12], mat=[0.0, -1.0], player="p1", game="g1")
    b = frame([10, 12], mat=[5.0, 5.0], player="p2", game="g2")
    out = net_material_change(pd.concat([a, b], ignore_index=True))
    assert out.net_mat.iloc[0] == -1.0
    assert out.net_mat.iloc[2] == 0.0


# ── z_pre3 ──────────────────────────────────────────────────

def test_z_pre3_is_mean_of_three_previous_own_moves():
    d = frame([10, 12, 14, 16], z=[1.0, 2.0, 3.0, 9.0])
    out = add_pre_speed(d, k=3)
    assert np.isnan(out.z_pre3.iloc[0])
    assert np.isnan(out.z_pre3.iloc[2])          # only two preceding moves
    assert out.z_pre3.iloc[3] == pytest.approx((1.0 + 2.0 + 3.0) / 3)


def test_z_pre3_skips_missing_ply():
    """z_pre3 must be missing when the measurement floor removes a move.

    Shifting by row position would take ply 8 as the preceding move.
    """
    d = frame([8, 10, 14, 16], z=[5.0, 1.0, 2.0, 3.0])   # ply 12 is absent
    out = add_pre_speed(d, k=3)
    assert np.isnan(out.z_pre3.iloc[3]), (
        "shifting by row position instead of by ply")


def test_z_pre3_rejects_duplicate_keys():
    d = frame([10, 10], z=[1.0, 2.0])
    with pytest.raises(ValueError):
        add_pre_speed(d)


# -- Event definitions ---------------------------------------

def test_events_B_uses_net_mat_not_see():
    d = frame([10, 12], mat=[0.0, -3.0])
    d = net_material_change(d)
    d["see_loss"] = [9.0, 9.0]          # large under SEE
    d["tier"] = "1300-1600"
    assert len(events_B(d)) == 1        # one event on the net definition
    assert len(events_B_see(d)) == 2    # two on the SEE definition


def test_events_B_requires_net_mat_column():
    d = frame([10, 12])
    with pytest.raises(KeyError, match="net_mat"):
        events_B(d)


def test_events_A_bounds_are_inclusive_and_symmetric():
    d = pd.DataFrame({
        "wp_delta": [-0.10, -0.09, -0.50, -0.50],
        "wp_before": [0.50, 0.50, 0.05, 0.95],
    })
    m = mask_A(d)
    assert m.tolist() == [True, False, False, False]
    assert len(events_A(d)) == 1


def test_split_groups_are_exclusive():
    d = frame([10, 12, 14, 16], mat=[0.0, -1.0, -1.0, -1.0])
    d = net_material_change(d)
    d["wp_delta"] = [-0.5, -0.5, 0.0, 0.0]
    d["wp_before"] = [0.5, 0.5, 0.5, 0.5]
    hasA, hasB = mask_A(d), mask_B(d)
    groups = [(hasA & ~hasB), (hasA & hasB), (hasB & ~hasA)]
    total = sum(g.sum() for g in groups)
    assert total == (hasA | hasB).sum()      # no overlap and nothing dropped


# -- Empty event sets (regression) ---------------------------

def test_builders_handle_empty_events():
    """Zero events must give back an empty frame (this used to raise
    AttributeError).

    run.py bins died reaching for z_post on data where it found no events
    in which the opponent lost material.
    """
    from src.analysis import lagwise, build_with_pretrend, build_pairs
    from src.prepare import build_and_match

    d = frame([10, 12, 14, 16], z=[1.0, 2.0, 3.0, 4.0])
    d["tier"] = "1300-1600"
    for c in ("wp_before", "clk_before", "n_legal"):
        d[c] = 1.0
    empty = d.iloc[:0]
    cal = {"wp_before": 0.5}

    r = lagwise(d, empty, cal)
    assert len(r) == 0 and "lag+1" in r.columns

    for fn in (build_with_pretrend, build_and_match):
        r = fn(d, empty, cal)
        assert len(r) == 0 and "effect" in r.columns

    r = build_pairs(d, empty, cal)
    assert len(r) == 0 and "event_condition" in r.columns


# ---------------------------------------------------------------------------
# Caliper widths must come from the frame the analyses actually run on.
#
# _calipers_from_main() used to take standard deviations over the whole
# prepared table, while every analysis first drops rows with no z_pre3. Those
# rows are 9.9% of the table and sit at the start of games -- low ply, high
# clock -- so the two conventions disagreed by up to 4%, and the published
# CSVs were produced under the second one.
# ---------------------------------------------------------------------------

def test_caliper_sd_excludes_missing_prespeed():
    import numpy as np
    import pandas as pd

    # Rows missing z_pre3 are deliberately extreme, as the real ones are.
    d = pd.DataFrame({
        "wp_before": [0.5] * 10 + [0.99] * 10,
        "clk_before": [100.0] * 10 + [600.0] * 10,
        "n_legal": [30] * 10 + [5] * 10,
        "ply": [40] * 10 + [2] * 10,
        "z_pre3": [0.1, -0.1] * 5 + [np.nan] * 10,
    })
    kept = d.dropna(subset=["z_pre3"])

    over_all = {v: d[v].std() for v in d.columns}
    over_kept = {v: kept[v].std() for v in kept.columns}

    # The two conventions must actually differ here, or the test proves nothing.
    assert over_all["wp_before"] != over_kept["wp_before"]

    # Every matching variable is constant once the missing rows are dropped.
    for v in ["wp_before", "clk_before", "n_legal", "ply"]:
        assert over_kept[v] == 0, v
        assert over_all[v] > 0, v


# ---------------------------------------------------------------------------
# A shard containing none of the sampled players writes an empty frame, whose
# columns are all dtype object. Concatenating one of those into the analysis
# table turned move_time into object, and np.log then raised
# "loop of ufunc does not support argument 0 of type float".
#
# The four untitled tiers never triggered it: every shard held one of the
# 1,878 players. The titled tier is sparse enough that 119 of its 2,355
# shards came back empty.
# ---------------------------------------------------------------------------

def test_empty_shard_does_not_poison_dtypes():
    import numpy as np
    import pandas as pd

    real = pd.DataFrame({"player": ["a", "a"], "ply": [10, 12],
                         "move_time": [3, 5]})
    blank = pd.DataFrame(columns=["player", "ply", "move_time"])
    assert blank["move_time"].dtype == object

    naive = pd.concat([real, blank], ignore_index=True)
    assert naive["move_time"].dtype == object, "the hazard must still exist"
    with pytest.raises(TypeError):
        np.log(naive["move_time"] + 1.0)

    # What cmd_prep now does: drop the empty frames before concatenating.
    parts = [f for f in (real, blank) if len(f)]
    fixed = pd.concat(parts, ignore_index=True)
    assert fixed["move_time"].dtype != object
    assert np.isfinite(np.log(fixed["move_time"] + 1.0)).all()
