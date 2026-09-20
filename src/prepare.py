"""
Ply table → analysis table, and the event definitions

  1. net_mat    net material change. The manuscript's Event B is defined here
     net_legal  net change in legal move count (control variable)
  2. z_pre3     pre-event speed control (ply-indexed)
  3. prepare    measurement floor, then within-player z standardisation
  4. events_*   Event A (blunder) and Event B (net material loss)
  5. calipers   matching caliper construction

★ Label warning — the preregistration and the manuscript use different names.

    Registered   Manuscript   Code           Meaning
    A            A            events_A       blunder (win-probability drop)
    B            (dropped)    —              error under time pressure; too
                                             few events survived matching
    C            B            events_B       net material loss

  The manuscript uses two events, A and B. The code follows the manuscript.
  The withdrawn SEE criterion is events_B_see. See the table in
  docs/definitions.md.

Driver: `run.py prep`.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE ★ the revised event definition is built here
#   - net_mat = mat_diff[ply+2] − mat_diff[ply], two consecutive moves by the
#     same player. Indexed by ply, not by row position.
#   - Event condition is net_mat < 0, strictly. Even exchanges give 0 and are
#     therefore excluded.
#   - Confirm that z standardisation is within player.
#   - z_pre3 is **ply-indexed** (ply−2, ply−4, ply−6). The earlier
#     implementation used groupby().shift(), which is row position; when the
#     measurement floor removes an intervening move the two diverge. The lag
#     windows are ply-indexed, so one analysis was mixing both conventions.
#   - events_B uses net_mat (the manuscript); events_B_see uses SEE (withdrawn,
#     supplementary comparison only, never called from the main path).
#   - Order matters: net_mat is computed BEFORE the measurement floor,
#     z_pre3 AFTER standardisation.
# ───────────────────────────────────────────────────────────

import io
import numpy as np
import pandas as pd

from .config import (TAU, EPOCH_WINDOW, LOG_OFFSET, CALIPER_SD, MATCH_VARS,
                     BLUNDER_THRESH, WP_LO, WP_HI, PRESPEED_K)
from .analysis import (caliper_mask, build_zmap, window_mean, control_pool,
                       require_isolated,
                       cluster_stats)

KEY = ["game_id", "player", "ply"]


# ════════════════════════════════════════════════════════════════════
# Derived variables - net_mat and z_pre3
# The pre-trend test showed that event epochs were already slower than their
# controls just before the event (+0.064 in the lower four tiers, +0.079 in
# FM+, averaged over lags -1 to -3 with no pre-speed control). Matching on
# win probability, time, complexity and ply does not control for how fast the
# player happened to be moving just beforehand.
# 
#   z_pre3(t) = mean( z(t-2), z(t-4), z(t-6) )     <- by ply index
# 
# It was not preregistered, so it is reported as a robustness check.
# ════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────
# Which code built this table
# ────────────────────────────────────────────────────────────────────

PREP_STAMP_KEY = b"chess_prep_code"


def preprocessing_hash():
    """
    A hash over the code that decides what the analysis table contains: this
    module, and the constants the preprocessing reads.

    It changes when the preprocessing changes, which is the point. A table
    built before the change then stops being readable instead of quietly
    producing results under a definition nobody is using any more.
    """
    import hashlib
    import os
    h = hashlib.sha256()
    h.update(io.open(os.path.join(os.path.dirname(__file__), "prepare.py"),
                     "rb").read())
    for name in ("TAU", "LOG_OFFSET", "PRESPEED_K", "OPENING_CUT",
                 "BLUNDER_THRESH", "WP_LO", "WP_HI"):
        h.update(f"{name}={globals().get(name)!r};".encode())
    return h.hexdigest()


def write_prepared(d, path):
    """Write the analysis table with the preprocessing hash attached."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(d, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta[PREP_STAMP_KEY] = preprocessing_hash().encode()
    pq.write_table(t.replace_schema_metadata(meta), path)


def prepared_stamp(path):
    """The hash stored in a table, or None if it carries none."""
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata or {}
    v = meta.get(PREP_STAMP_KEY)
    return v.decode() if v else None


def read_prepared(path, columns=None, require_stamp=True):
    """
    Read an analysis table, refusing one that current code did not build.

    A warning would not do. The stale table was used for a full day of
    analysis while every stage ran without complaint; only a comparison
    against freshly built data showed it. So this raises.
    """
    import pandas as pd
    if require_stamp:
        want = preprocessing_hash()
        got = prepared_stamp(path)
        if got is None:
            raise ValueError(
                f"{path} carries no preprocessing stamp, so there is no way "
                f"to tell which code built it. Rebuild it with `run.py prep` "
                f"(or `prep fm`).")
        if got != want:
            raise ValueError(
                f"{path} was built by different preprocessing code "
                f"(stored {got[:12]}, current {want[:12]}). Rebuild it with "
                f"`run.py prep`, or check out the code that made it.")
    return pd.read_parquet(path, columns=columns)


def add_pre_mean(d, src, k=3, col=None):
    """
    Attach to each ply the mean of `src` over the player's preceding k moves,
    found by ply index.

    Consecutive moves by the same player are two plies apart, so this looks
    up ply-2, ply-4, ..., ply-2k. If any one is absent the result is NaN.

    `src` is a column of `d`, so the control it defines is in that column's
    own units. Matching the recent history of one quantity with the recent
    history of another does not control it: the two correlate but are not the
    same, and a quantity defined only inside a window (blunder_q, on
    win probability 10-90%) is missing on different rows than one defined
    everywhere.
    """
    col = col or f"{src}_pre{k}"
    d = d.copy()
    dup = d.duplicated(subset=KEY).sum()
    if dup:
        raise ValueError(
            f"(game_id, player, ply) is not unique ({dup} rows). "
            f"Looking {col} up by ply requires uniqueness.")

    s = d.set_index(KEY)[src]
    acc = np.zeros(len(d), dtype=float)
    for i in range(1, k + 1):
        key = pd.MultiIndex.from_arrays(
            [d["game_id"].values, d["player"].values, d["ply"].values - 2 * i],
            names=KEY)
        acc += s.reindex(key).to_numpy(dtype=float)   # absent -> NaN, propagates
    d[col] = acc / k
    return d


def add_pre_speed(d, k=3, col="z_pre3"):
    """The pre-event speed control: `add_pre_mean` over standardised time."""
    return add_pre_mean(d, "z", k=k, col=col)


def _next_own(d, col):
    """Value of `col` at the player's next move (ply+2), NaN if absent."""
    dup = d.duplicated(subset=KEY).sum()
    if dup:
        raise ValueError(f"(game_id, player, ply) is not unique ({dup} rows).")
    s = d.set_index(KEY)[col]
    key = pd.MultiIndex.from_arrays(
        [d["game_id"].values, d["player"].values, d["ply"].values + 2],
        names=KEY)
    return s.reindex(key).to_numpy(dtype=float)


def net_legal_change(d, col="net_legal"):
    """
    Change in legal move count between two consecutive moves by the same
    player.

        net_legal(t) = n_legal(t+2) − n_legal(t)

    This is the quantity used in the manuscript's "Number of available
    options" section. It moves in opposite directions when the player loses
    material and when the opponent does (-8.36 on average for a nine-point
    own loss, +6.31 for a nine-point opponent loss; four untitled tiers, all
    events - see legal_change.csv), so holding it fixed and
    comparing the effect rules out "the position simply became simpler".
    """
    d = d.copy()
    d[col] = _next_own(d, "n_legal") - d["n_legal"].to_numpy(dtype=float)
    return d


def net_material_change(d, col="net_mat"):
    """
    Change in the material differential between two consecutive moves by the
    same player.

        net_mat(t) = mat_diff(t+2) − mat_diff(t)

    mat_diff is recorded **immediately before the move** and **from the
    mover's point of view**. So net_mat < 0 means the player lost material on
    net between that move and their next turn. An even trade gives 0 and is
    therefore not an event.

    This implements "Event B - Material net loss" in docs/definitions.md.
    """
    d = d.copy()
    d[col] = _next_own(d, "mat_diff") - d["mat_diff"].to_numpy(dtype=float)
    return d

# ════════════════════════════════════════════════════════════════════
# Event definitions and matching
#   Event A    win probability drops 10pp, position between 10% and 90%
#              (sensitivity checks at 20 and 30pp)
#   Event B    material net loss (net_mat < 0); an even trade is not an event
#   Matching   same player, plus a 0.2 SD caliper on each variable
#              (Austin 2011); 1:N, averaged
#   Outcome    z = (ln(move_time+1) - mu_p)/sigma_p
#              effect = z(event) - z(control)
# ════════════════════════════════════════════════════════════════════

def prepare(df):
    """Apply the measurement floor and standardise z within player."""
    d = df[df.move_time >= TAU].copy()
    d["y"] = np.log(d.move_time + LOG_OFFSET)
    g = d.groupby("player")["y"]
    d["z"] = (d.y - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return d.dropna(subset=["z"])


def standardise(d, tau=TAU, k=PRESPEED_K):
    """
    Measurement floor, then z within player, then the preceding-speed control.

    The one implementation. cmd_prep and the robustness specifications both
    call it, because they had a copy each and the copies diverged: the
    robustness one produced 174,515 more missing z_pre3 than the table it was
    being compared against, and the opening-cut row and the measurement-floor
    row -- the same specification under two names -- read -0.6396 and -0.6210.

    The order matters and is fixed here. z_pre3 is taken by ply index, so it
    has to be computed on the rows that survive the floor; computing it first
    would let "the preceding three moves" reach past a move the floor
    removed.
    """
    x = d[d.move_time >= tau].copy()
    x["y"] = np.log(x.move_time + LOG_OFFSET)
    g = x.groupby("player")["y"]
    x["z"] = (x.y - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    x = x.dropna(subset=["z"]).reset_index(drop=True)
    return add_pre_speed(x, k=k)


def calipers(d, sd_mult=CALIPER_SD, vars_=MATCH_VARS):
    """Austin (2011): 0.2 times each variable's SD."""
    return {v: sd_mult * d[v].std() for v in vars_ if v in d.columns}


def expected_quality(d, min_cell=30):
    """
    Loss on a move, in excess of what its position would ordinarily cost.

    How much win probability a move gives up is driven mechanically by where
    the position sits: its win probability, how many legal moves there are,
    and how much clock is left. A position reached by losing material has
    fewer legal moves and a lower win probability, and both make a given
    lapse cost less. Matching cannot remove that, because the quantity is
    measured on the move after the event, by which point the event has
    already changed it.

    So it is removed from the outcome instead. Every evaluated move is placed
    in a cell -- win probability in 20 equal bins, legal-move count in
    deciles, remaining clock in quintiles -- and the cell mean becomes what
    that position ordinarily costs. The outcomes are the excess over it.

    Cells holding fewer than min_cell moves are left as NaN rather than
    fitted on a handful of observations.
    """
    d = d.copy()
    ok = d.wp_before.notna() & d.wp_delta.notna()
    cells = pd.DataFrame({
        "w": pd.cut(d.wp_before, np.linspace(0, 1, 21), include_lowest=True),
        "l": pd.qcut(d.n_legal.rank(method="first"), 10, labels=False),
        "c": pd.qcut(d.clk_before.rank(method="first"), 5, labels=False),
    }, index=d.index)
    key = [cells.w, cells.l, cells.c]

    loss = (-d.wp_delta).where(ok)
    blun = d["blunder_q"]
    size = loss.groupby(key, observed=True).transform("size")
    big = size >= min_cell

    d["wp_loss_exp"] = loss.groupby(key, observed=True).transform("mean").where(big)
    d["blunder_exp"] = blun.groupby(key, observed=True).transform("mean").where(big)
    d["wp_loss_x"] = (loss - d["wp_loss_exp"]).astype(float)
    d["blunder_x"] = (blun - d["blunder_exp"]).astype(float)
    d["_cell_n"] = size
    return d


def next_quality_columns(d, thresh=BLUNDER_THRESH, wp_lo=WP_LO, wp_hi=WP_HI):
    """
    Outcome columns for run.py nextq, one value per move.

    These are properties of the move itself, not of the move that follows it:
    lagwise looks them up at ply + 2k on its own. Only wp_before_next is
    shifted here, because it is a matching variable rather than an outcome.

    blunder_q is forced to float. A boolean column carrying NaN becomes
    object dtype, and a mean over that reduces with `or` rather than by
    addition, returning 1/n for every group instead of the rate.
    """
    d = d.copy()
    inband = d.wp_before.between(wp_lo, wp_hi)
    ok = d.wp_delta.notna()
    d["blunder_q"] = np.where(ok & inband,
                              (d.wp_delta <= -thresh).astype(float),
                              np.nan).astype(float)
    d["wp_loss_q"] = (-d.wp_delta).astype(float)
    d["wp_before_next"] = _next_own(d, "wp_before")
    return d


def events_A(d, thresh=BLUNDER_THRESH, wp_lo=WP_LO, wp_hi=WP_HI):
    """Blunders. A lost position is excluded by the definition itself (0
    observed); the bound is applied symmetrically."""
    m = d.wp_delta.notna() & (d.wp_delta <= -thresh)
    m &= d.wp_before.between(wp_lo, wp_hi)
    return d[m]


def mask_A(d, thresh=BLUNDER_THRESH, wp_lo=WP_LO, wp_hi=WP_HI):
    """The same condition as events_A, as a boolean mask for partitioning."""
    return (d.wp_delta.notna() & (d.wp_delta <= -thresh)
            & d.wp_before.between(wp_lo, wp_hi))


def events_B(d, min_loss=1):
    """
    Material net loss - Event B in the manuscript (labelled C in the
    preregistration).

    net_mat is built from mat_diff by net_material_change. An even trade
    gives net_mat == 0 and so drops out on its own.
    """
    _require_net_mat(d)
    return d[d.net_mat.notna() & (d.net_mat <= -min_loss)]


def mask_B(d, min_loss=1):
    _require_net_mat(d)
    return d.net_mat.notna() & (d.net_mat <= -min_loss)


def mask_any_event(d):
    """
    Every move this study treats as an event, for keeping out of the control
    pool. The Method says controls are windows in which no event occurred.

      - a blunder (identifiable only where an engine evaluation exists)
      - the player losing material on net
      - **the opponent** losing material on net

    The third is not analysed under that name, but it is followed by slower
    moves (+0.219), so a control drawn from one is pulled the opposite way
    from a material-loss event, biasing the comparison towards zero.

    What cannot be removed: a blunder in a game carrying no engine evaluation
    is not identifiable, so some remain. Stated in the limitations.
    """
    m = pd.Series(False, index=d.index)
    if "net_mat" in d.columns:
        m |= d.net_mat.notna() & (d.net_mat != 0)
    if {"wp_delta", "wp_before"} <= set(d.columns):
        m |= mask_A(d)
    return m


def mask_after_event(d):
    """
    The player's own preceding move (two plies back) was an event.

    28.9% of moves are an event of some kind and 27.5% follow one, so a
    control drawn without this condition is often the move straight after
    someone's event, its time already changed by it. 42.9% of events
    themselves follow another event.
    """
    ev = mask_any_event(d)
    keys = set(zip(d.game_id[ev], d.player[ev], d.ply[ev]))
    return pd.Series(
        [(g, p, y - 2) in keys
         for g, p, y in zip(d.game_id, d.player, d.ply)], index=d.index)


def isolated(d, mask):
    """
    Events from `mask` that do not themselves follow an event.

    Every stage must pass its event mask through this. lagwise asserts it,
    so a stage that forgets will stop rather than report the effect of a run
    of events as the effect of one.
    """
    return mask & ~mask_after_event(d)


def events_B_see(d, min_value=1):
    """
    Event C as preregistered = the manuscript's Event B (SEE > 0).
    **This definition was withdrawn.**

    It failed to exclude even trades: 55.8% of its events had zero net
    change. Use it only to regenerate the appendix comparison.
    See docs/definitions.md.
    """
    if "see_loss" not in d.columns:
        raise KeyError("no see_loss column - was stage 2 run with do_see=True?")
    return d[d.see_loss.notna() & (d.see_loss >= min_value)]


def _require_net_mat(d):
    if "net_mat" not in d.columns:
        raise KeyError(
            "no net_mat column - build the analysis table with run.py prep "
            "first. For the preregistered SEE definition use events_B_see().")


def build_and_match(d, events, cal, window=EPOCH_WINDOW, seed=0,
                    max_events=None, strict=True):
    """
    1:N matching - averages the window z over every candidate satisfying the
    caliper.

    There is no cap on the number of candidates. Against a cap of 50 the
    running time was the same -- the bottleneck is filtering candidates, not
    reading windows -- and the effects differed by about 1e-5. Without a cap
    there is no random selection, which also makes the result reproducible.
    """
    if len(events) == 0:
        return pd.DataFrame(columns=[
            "player", "tier", "game_id", "ply", "z_post", "z_ctrl",
            "n_obs", "n_ctrl", "clk_before", "wp_before", "size", "effect"])
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    require_isolated(events, d)
    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)
    by_player = {p: g for p, g in control_pool(d, events, cal).groupby("player")}

    recs = []
    for r in events.itertuples():
        zp, nobs = window_mean(zmap, r.game_id, r.player, r.ply, window, +1)
        g = by_player.get(r.player)
        zc, ncand = np.nan, 0
        if g is not None and len(g):
            m, ok = caliper_mask(r, g, cal, strict=strict)
            if ok:
                c = g[m]
                ncand = len(c)
                if ncand:
                    zs = [window_mean(zmap, x.game_id, x.player, x.ply,
                                      window, +1)[0] for x in c.itertuples()]
                    zs = [v for v in zs if not np.isnan(v)]
                    if zs:
                        zc = float(np.mean(zs))
        recs.append({
            "player": r.player, "tier": r.tier, "game_id": r.game_id, "ply": r.ply,
            "z_post": zp, "z_ctrl": zc, "n_obs": nobs, "n_ctrl": ncand,
            "clk_before": getattr(r, "clk_before", np.nan),
            "wp_before": getattr(r, "wp_before", np.nan),
            "size": getattr(r, "net_mat", np.nan),
        })
    e = pd.DataFrame(recs)
    e["effect"] = e.z_post - e.z_ctrl
    return e


def reliability(epochs, min_events=10):
    """Split-half reliability by game parity, Spearman-Brown corrected,
    with variance components."""
    e = epochs.dropna(subset=["effect"]).copy()
    if len(e) < 50:
        return None
    # Alternate the player's OWN games. pd.factorize over the whole frame
    # numbers games in the order they appear across every player, so other
    # players' games fall between a player's own and the split within that
    # player is arbitrary rather than alternating -- it can even land
    # lopsided. The Method says "split into alternate games", and this is
    # what that means.
    order = e.groupby("player").game_id.transform(
        lambda g: pd.factorize(g)[0])
    e["half"] = order % 2
    piv = e.groupby(["player", "half"])["effect"].agg(["mean", "size"]).unstack()
    ok = (piv["size"].fillna(0) >= min_events / 2).all(axis=1)
    piv = piv[ok]

    per = e.groupby("player")["effect"].agg(["mean", "var", "size"])
    per = per[per["size"] >= min_events]
    out = {"n_players_split": len(piv), "n_players_var": len(per)}
    if len(piv) >= 20:
        a, b = piv["mean"][0], piv["mean"][1]
        r = np.corrcoef(a, b)[0, 1]
        out["r"] = r
        out["sb"] = 2 * r / (1 + r) if r > -1 else np.nan
    if len(per) >= 20:
        var_w = per["var"].mean()
        sb2 = max(per["mean"].var() - var_w / per["size"].mean(), 0)
        out.update(sigma_b=np.sqrt(sb2), sigma_w=np.sqrt(var_w),
                   mean_events=per["size"].mean())
    return out


def moderation_by_time(epochs, bins=(0, 30, 60, 120, 180, 300, 600)):
    """
    Effect by band of remaining time - time pressure is context, not an
    event.

    Note: below 30 seconds the arithmetic constraint is strong enough that
    the band should be read separately. SEs are clustered **by player**. An
    earlier implementation used an event-level .sem(), which underestimates
    them because events from the same player are correlated.
    """
    e = epochs.dropna(subset=["effect"]).copy()
    e["clk_bin"] = pd.cut(e.clk_before, bins)
    rows = []
    for b, g in e.groupby("clk_bin", observed=True):
        st = cluster_stats(g, "effect")
        if st is None:
            continue
        rows.append({"clk_bin": b, **st})
    return pd.DataFrame(rows).set_index("clk_bin")
