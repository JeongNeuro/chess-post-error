"""
Matching, lag decomposition, pre-trend, mixed-effects models

  1. caliper matching and clustered SEs        caliper_mask / cluster_stats
  2. decomposition by lag                      lagwise / summarize
  3. pre-trend and difference-in-differences   build_with_pretrend
  4. preregistered mixed-effects model         build_pairs / fit_mixed

Driver: `run.py`.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE ★ the standard errors come from here
#   - Clustered SEs are computed in exactly one place, cluster_stats:
#     groupby('player').mean() followed by .sem(). Computing them at the
#     event level underestimates them — that was wrong once.
#   - Both effect (event-weighted) and effect_pw (player-weighted) are
#     reported. A large gap between them signals that events per player are
#     badly unbalanced.
#   - caliper_mask DISCARDS an event that is missing any matching variable
#     (strict=True). Previously that variable alone was skipped and the event
#     matched on the rest, so some events entered on looser criteria than the
#     manuscript describes ("all six within the caliper").
#   - The control pool is not "moves with no event"; it is "moves that are not
#     the event currently under analysis". This belongs in the limitations.
#   - Windows count only the player's own moves, hence ply ± 2k.
# ───────────────────────────────────────────────────────────

import warnings

import numpy as np
import pandas as pd

from .config import EPOCH_WINDOW

LAGS = list(range(-6, 4))          # −6 … +3


# ════════════════════════════════════════════════════════════════════
# 1. Caliper matching and clustered standard errors
# lagwise, pretrend and mixed each held their own copy of the same matching
# logic. Three copies can drift apart, so they were merged here.
# 
# Missing values in the caliper (important)
#     With strict=True, an event missing any matching variable is dropped
#     entirely. The control pool is already filtered by dropna, so without
#     this the specification would quietly differ between the two sides.
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# What to check here ★
#   - caliper_mask: the mask that selects control candidates for one event.
#     With strict=True, an event missing any matching variable is dropped.
#   - window_mean: counts only the player's own moves, so it steps by 2k.
# ─────────────────────────────────────────────────────────────


def is_missing(v):
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def caliper_mask(row, cand, cal, strict=True):
    """
    For one event row, which rows of the candidate frame `cand` qualify.

    Returns (mask, ok)
      mask : np.ndarray[bool], the same length as cand
      ok   : False means this event cannot be matched -- under strict, one of
             its matching variables is missing
    """
    m = np.ones(len(cand), bool)
    for v, w in cal.items():
        rv = getattr(row, v, None)
        if is_missing(rv):
            if strict:
                return m, False
            continue
        if v not in cand.columns:
            if strict:
                return m, False
            continue
        m &= np.abs(cand[v].values - rv) <= w
    return m, True


def build_zmap(d):
    """Lookup table from (game_id, player, ply) to z."""
    return dict(zip(zip(d.game_id.values, d.player.values, d.ply.values),
                    d.z.values))


def window_mean(zmap, gid, pl, ply, window, sign=1):
    """
    Mean z over a window of +/-window around the event ply.
    Only the player's own moves count, so the plies are ply + sign*2k.
    Returns (the mean or nan, how many were actually observed).
    """
    zs = []
    for k in range(1, window + 1):
        v = zmap.get((gid, pl, ply + sign * 2 * k))
        if v is not None and not np.isnan(v):
            zs.append(v)
    return (float(np.mean(zs)) if zs else np.nan), len(zs)


def control_pool(d, events, cal):
    """
    The control pool: rows with every matching variable observed, excluding
    the events themselves.

    Note: this is not "moves with no event" but "moves that are not the
    event currently being analysed". Moves belonging to a different kind of
    event (material losses, when blunders are under analysis) or falling
    inside another event's post-window remain candidates. This is stated in
    the limitations section.
    """
    ev_idx = set(zip(events.game_id.values, events.ply.values))
    keep = ~np.fromiter(
        (k in ev_idx for k in zip(d.game_id.values, d.ply.values)),
        dtype=bool, count=len(d))
    cols = [v for v in cal if v in d.columns]
    pool = d[keep]
    if cols:
        pool = pool.dropna(subset=cols)
    return pool


def cluster_stats(df, value_col, player_col="player"):
    """
    Summary with standard errors clustered by player.

    effect      event-weighted mean (the value reported in the manuscript)
    se          SEM of the per-player means (the clustered SE)
    effect_pw   player-weighted mean. effect and se weight the data
                differently, so a large gap between effect and effect_pw
                signals badly unbalanced cluster sizes.
    """
    s = df[[player_col, value_col]].dropna()
    if len(s) == 0:
        return None
    per = s.groupby(player_col)[value_col].mean()
    se = per.sem()
    eff = s[value_col].mean()
    return {
        "effect": eff,
        "se": se,
        "effect_pw": per.mean(),
        "t": eff / se if se and se > 0 else np.nan,
        "n_events": len(s),
        "n_players": len(per),
    }

# ════════════════════════════════════════════════════════════════════
# 2. Decomposition by lag
# An earlier version reported the pre-event window (t-3 to t-1) as a single
# mean. That mean cannot distinguish two situations:
# 
#   plan then execute:  t-3=0, t-2=large positive, t-1=negative -> mean +0.02
#   position difficulty: t-3=+0.02, t-2=+0.02, t-1=+0.02  -> mean +0.02
# 
# Computing the event-control difference at each lag k = -6 ... +3 shows the
# **shape** of the pre-event window rather than its average. A window running
# past the start or end of the game is missing, so n differs by lag.
# ════════════════════════════════════════════════════════════════════



EMPTY_LAG_COLS = ["player", "tier", "n_ctrl", "z_at"] +                  [f"lag{k:+d}" for k in LAGS]
EMPTY_EPOCH_COLS = ["player", "tier", "game_id", "ply", "z_post", "z_ctrl",
                    "z_pre", "z_ctrl_pre", "z_at", "z_ctrl_at",
                    "n_obs", "n_before", "n_ctrl",
                    "effect", "pretrend", "at_event", "did"]


def lagwise(d, events, cal, lags=LAGS, seed=0, max_events=None, strict=True):
    """
    Lag profile for each event.
    Returns a DataFrame with one row per event and columns lag_-6 ... lag_+3.
    """
    if len(events) == 0:
        return pd.DataFrame(columns=["player", "tier", "n_ctrl", "z_at"]
                            + [f"lag{k:+d}" for k in lags])
    rng = np.random.default_rng(seed)
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    # Keep only games belonging to players who have events, to save memory.
    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)

    pool = control_pool(d, events, cal) if cal else d
    by_player = {p: g for p, g in pool.groupby("player")}

    def prof(gid, pl, ply):
        """z at each lag from the reference ply, nan where absent."""
        return [zmap.get((gid, pl, ply + 2 * k), np.nan) for k in lags]

    recs = []
    for r in events.itertuples():
        ze = prof(r.game_id, r.player, r.ply)
        g = by_player.get(r.player)
        zc = [np.nan] * len(lags)
        ncand = 0
        if g is not None and len(g):
            if cal:
                m, ok = caliper_mask(r, g, cal, strict=strict)
                c = g[m] if ok else g.iloc[:0]
            else:
                # Uncontrolled: 20 random plies from the same player
                c = g.iloc[rng.choice(len(g), min(20, len(g)), replace=False)]
            ncand = len(c)
            if ncand:
                mat = np.array([prof(x.game_id, x.player, x.ply)
                                for x in c.itertuples()], dtype=float)
                # At some lags every candidate is missing (the window falls
                # outside the game). Suppress nanmean's "Mean of empty slice".
                with np.errstate(invalid="ignore"),                         warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    zc = np.nanmean(mat, axis=0)
        row = {"player": r.player, "tier": r.tier, "n_ctrl": ncand,
               "z_at": ze[lags.index(0)]}
        for i, k in enumerate(lags):
            row[f"lag{k:+d}"] = ze[i] - zc[i]
        recs.append(row)
    return pd.DataFrame(recs)


def summarize(prof_df, event_type, tier, spec, lags=LAGS, min_n=30):
    """effect, clustered SE and n, by lag."""
    out = []
    for k in lags:
        col = f"lag{k:+d}"
        s = prof_df[["player", col]].dropna()
        if len(s) < min_n:
            continue
        st = cluster_stats(s, col)
        out.append({"event_type": event_type, "tier": tier, "spec": spec,
                    "lag": k, **st})
    return pd.DataFrame(out)


def per_player_at(prof_df, lag=1, label=None):
    """
    Mean effect per player at one lag.
    This is the source of the distribution panel in Fig 2a and of
    data/derived/per_player_t1.csv.
    """
    col = f"lag{lag:+d}"
    s = prof_df[["player", col]].dropna()
    per = s.groupby("player")[col].mean().rename("eff").reset_index()
    if label is not None:
        per.insert(0, "group", label)
    return per

# ════════════════════════════════════════════════════════════════════
# 3. Pre-trend check
# This addresses the pre-error speeding problem raised by Pfister and
# Foerster (2021). A person may already be speeding up before the error, in
# which case a return to their normal pace afterwards reads as slowing.
# 
#   no pre-event difference -> matching is adequate; the result stands
#   a pre-event difference -> matching is incomplete; correction is needed
# ════════════════════════════════════════════════════════════════════

def build_with_pretrend(d, events, cal, window=EPOCH_WINDOW, seed=0,
                        max_events=None, strict=True):
    """
    Window z for event and control epochs, on both sides of the event.

    post : t+1 to t+window   (the outcome variable)
    pre  : t-window to t-1   (for the pre-trend check)

    These are the player's own moves, so the plies step by 2k.
    """
    if len(events) == 0:
        return pd.DataFrame(columns=EMPTY_EPOCH_COLS)
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)
    by_player = {p: g for p, g in control_pool(d, events, cal).groupby("player")}

    recs = []
    for r in events.itertuples():
        zp, nobs = window_mean(zmap, r.game_id, r.player, r.ply, window, +1)
        zb, nbef = window_mean(zmap, r.game_id, r.player, r.ply, window, -1)
        z_at = zmap.get((r.game_id, r.player, r.ply))     # the event move itself

        g = by_player.get(r.player)
        zc = zcb = zc_at = np.nan
        ncand = 0
        if g is not None and len(g):
            m, ok = caliper_mask(r, g, cal, strict=strict)
            c = g[m] if ok else g.iloc[:0]
            ncand = len(c)
            if ncand:
                post, pre, at = [], [], []
                for x in c.itertuples():
                    a, _ = window_mean(zmap, x.game_id, x.player, x.ply,
                                       window, +1)
                    b, _ = window_mean(zmap, x.game_id, x.player, x.ply,
                                       window, -1)
                    if not np.isnan(a):
                        post.append(a)
                    if not np.isnan(b):
                        pre.append(b)
                    v = zmap.get((x.game_id, x.player, x.ply))
                    if v is not None and not np.isnan(v):
                        at.append(v)
                zc = float(np.mean(post)) if post else np.nan
                zcb = float(np.mean(pre)) if pre else np.nan
                zc_at = float(np.mean(at)) if at else np.nan

        recs.append({
            "player": r.player, "tier": r.tier,
            "game_id": r.game_id, "ply": r.ply,
            "z_post": zp, "z_ctrl": zc,
            "z_pre": zb, "z_ctrl_pre": zcb,
            "z_at": z_at, "z_ctrl_at": zc_at,
            "n_obs": nobs, "n_before": nbef, "n_ctrl": ncand,
        })
    e = pd.DataFrame(recs)
    e["effect"] = e.z_post - e.z_ctrl          # post-event: the outcome
    e["pretrend"] = e.z_pre - e.z_ctrl_pre     # pre-event: should be 0
    e["at_event"] = e.z_at - e.z_ctrl_at       # the event move itself
    e["did"] = e.effect - e.pretrend           # difference-in-differences
    return e


def pretrend_report(e, label=""):
    """
    SEs are clustered by player. An earlier version used an event-level .sem().
    """
    ok = e.dropna(subset=["effect", "pretrend"])
    out = {"label": label, "n": len(ok)}
    for k in ["pretrend", "at_event", "effect", "did"]:
        st = cluster_stats(ok, k)
        out[k] = (np.nan, np.nan) if st is None else (st["effect"], st["se"])
    return out

# ════════════════════════════════════════════════════════════════════
# 4. The preregistered mixed-effects model
# As specified under Statistical models in the OSF preregistration:
#     z_DV ~ event_condition + wp_level + remaining_time + legal_move_count
#            + log(total_games) + (event_condition | player)
# 
# Each event epoch produces two rows:
#   event_condition = 1 : mean z over the post-window (t+1 to t+3)
#   event_condition = 0 : mean z over the matched control windows
# ════════════════════════════════════════════════════════════════════

def build_pairs(d, events, cal, seed=0, max_events=None, strict=True,
                window=1):
    """Event and control pairs in long form, with covariates."""
    if len(events) == 0:
        return pd.DataFrame(columns=["player", "tier", "wp_level",
                                     "remaining_time", "legal_moves", "ply",
                                     "pre_speed", "eid",
                                     "z_dv", "event_condition"])
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)
    by_player = {p: g for p, g in control_pool(d, events, cal).groupby("player")}

    def win(gid, pl, ply):
        """
        The dependent variable is the move at t+1.

        It used to be the mean of t+1..t+3. Every other stage reports t+1, and
        because t+2 and t+3 sit near zero the average ran about 0.2 where the
        matched estimate ran about 0.5 -- two numbers for one quantity, side
        by side in the same Results section.

        ps6 and did keep the window: a difference-in-differences needs the
        pre- and post-windows to be symmetric, so there the window is the
        design rather than a choice.
        """
        if window == 1:
            v = zmap.get((gid, pl, ply + 2))
            return np.nan if v is None else v
        return window_mean(zmap, gid, pl, ply, window, +1)[0]

    recs = []
    for r in events.itertuples():
        zp = win(r.game_id, r.player, r.ply)
        if np.isnan(zp):
            continue
        g = by_player.get(r.player)
        if g is None or not len(g):
            continue
        m, ok = caliper_mask(r, g, cal, strict=strict)
        if not ok:
            continue
        c = g[m]
        if not len(c):
            continue
        zs = [win(x.game_id, x.player, x.ply) for x in c.itertuples()]
        zs = [v for v in zs if not np.isnan(v)]
        if not zs:
            continue
        zc = float(np.mean(zs))
        common = {"player": r.player, "tier": r.tier,
                  "wp_level": r.wp_before, "remaining_time": r.clk_before,
                  "legal_moves": r.n_legal, "ply": r.ply,
                  "pre_speed": getattr(r, "z_pre3", np.nan),
                  "eid": len(recs)}
        recs.append({**common, "z_dv": zp, "event_condition": 1})
        recs.append({**common, "z_dv": zc, "event_condition": 0})
    return pd.DataFrame(recs)


def fit_mixed(long_df, activity=None, with_wp=False):
    """
    Random intercept plus random slope.
    If it does not converge, fall back to (1) random intercept only, then
    (2) the paired difference.
    """
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")

    df = long_df.copy()

    # Controls follow config.MATCH_VARS, the main specification.
    #
    # The formula used to name wp_level unconditionally. Win probability is
    # missing for 83% of untitled moves, and statsmodels drops those rows, so
    # the "main" model was silently fitted on 5,426 of 32,782 rows -- too few
    # per player to converge. It then fell back to the paired difference and
    # reported that as the model.
    #
    # Pass with_wp=True for the registered specification, which is only
    # computed on the engine-evaluated subsample where it is present.
    ctrl = ["remaining_time", "legal_moves", "ply", "pre_speed"]
    if with_wp:
        ctrl = ["wp_level"] + ctrl
    ctrl = [c for c in ctrl if c in df.columns and df[c].notna().any()]
    for c in ctrl:
        sd = df[c].std()
        df[c] = (df[c] - df[c].mean()) / (sd if sd else 1.0)
    df = df.dropna(subset=["z_dv"] + ctrl)
    formula = "z_dv ~ event_condition" + "".join(f" + {c}" for c in ctrl)
    if activity is not None:
        df = df.merge(activity, on="player", how="left")
        df["log_games"] = np.log(df["n_games"].fillna(df["n_games"].median()))
        df["log_games"] = (df.log_games - df.log_games.mean()) / df.log_games.std()
        formula += " + log_games"

    out = {"n_obs": len(df), "n_players": df.player.nunique(),
           "controls": " + ".join(ctrl)}
    try:
        md = smf.mixedlm(formula, df, groups=df["player"],
                         re_formula="~event_condition")
        fit = md.fit(method="lbfgs", maxiter=200)
        out["path"] = "random slope"
    except Exception:
        fit = None
    if fit is None or not fit.converged:
        try:
            md = smf.mixedlm(formula, df, groups=df["player"])
            fit = md.fit(method="lbfgs", maxiter=200)
            out["path"] = "random intercept (fallback 1)"
        except Exception:
            fit = None
    if fit is None or not fit.converged:
        # fallback 2: the paired difference
        w = df.pivot_table(index="eid", columns="event_condition",
                           values="z_dv").dropna()
        diff = w[1] - w[0]
        pl = df.drop_duplicates("eid").set_index("eid").player
        per = diff.groupby(pl).mean()
        out.update(path="paired difference (fallback 2)",
                   coef=diff.mean(), se=per.sem(), converged=False)
        return out, None

    out.update(coef=fit.params["event_condition"],
               se=fit.bse["event_condition"],
               z=fit.tvalues["event_condition"],
               p=fit.pvalues["event_condition"],
               converged=fit.converged)
    return out, fit


# ══════════════════════════════════════════════════════════════
# 5. Individual-level reliability and its dependence on observation count
# ══════════════════════════════════════════════════════════════
# Table 2 and Fig 2d/2e come from here.
#
# The claim is that reliability for blunders does not rise as observations
# accumulate, so the comparison has to hold the observation count **fixed**.
# Reliability is therefore remeasured with each player truncated to N events.
#
# Note: the split is **by game**. Events within a game are correlated, so
# splitting by event inflates the reliability.


def restrict_events(epochs, n, seed=0):
    """Truncate each player to n events at random, dropping players with
    fewer than n."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in epochs.groupby("player", sort=False):
        if len(g) < n:
            continue
        keep.append(g.iloc[rng.choice(len(g), n, replace=False)])
    if not keep:
        return epochs.iloc[:0]
    return pd.concat(keep, ignore_index=True)


def reliability_curve(epochs, counts=(10, 20, 30, 40, 50, 60, 80, 100,
                                      150, 200, 237, 300),
                      seed=0, min_players=20, label=""):
    """
    Split-half reliability by observation count, Spearman-Brown corrected.

    Columns: label, n_events_per_player, r, sb, n_players
    Counts leaving fewer than min_players are skipped. This is the situation
    the manuscript describes as
    "beyond that count fewer than 25 players met the criterion, so no stable
    estimate could be obtained".
    """
    from .prepare import reliability
    out = []
    for n in counts:
        sub = restrict_events(epochs, n, seed=seed)
        n_pl = sub.player.nunique()
        if n_pl < min_players:
            out.append({"label": label, "n_events_per_player": n,
                        "r": np.nan, "sb": np.nan, "n_players": n_pl,
                        "note": f"{n_pl} players (<{min_players}) - cannot estimate"})
            continue
        rel = reliability(sub, min_events=max(2, n // 2))
        out.append({"label": label, "n_events_per_player": n,
                    "r": (rel or {}).get("r", np.nan),
                    "sb": (rel or {}).get("sb", np.nan),
                    "n_players": n_pl, "note": ""})
    return pd.DataFrame(out)


def bin_effects(ev, value_col, bins, effect_col="effect", label=""):
    """
    Effect by bin, with standard errors clustered by player.

    Fig 2b: the effect within bins of change in legal move count.
    """
    d = ev.dropna(subset=[value_col, effect_col]).copy()
    d["_bin"] = pd.cut(d[value_col], bins)
    rows = []
    for b, g in d.groupby("_bin", observed=True):
        st = cluster_stats(g, effect_col)
        if st is None:
            continue
        rows.append({"label": label, "bin": str(b),
                     "bin_mid": float(b.mid), **st})
    return pd.DataFrame(rows)
