#!/usr/bin/env python
"""
Single entry point for the pipeline.

    python run.py <stage> [arguments]
    python run.py --help              list every stage
    python run.py <stage> --help      arguments for that stage, and a note on
                                      what to check in it

Full order (approximate running times in brackets)

    scan            stage 1, account metadata over shards 0-421   (~15 min)
    scan-titled     titled accounts, 2024-01 .. 2024-06      <- slowest stage
    sample          fix the player sample -> out/ (not distributed)  (~1 min)
    extract         stage 2, board replay and ply features        (~45 min)
    extract-titled  the same for the titled tier                  (~20 min)
    prep            analysis table (net_mat, z, z_pre3)            (~5 min)
    se              clustered SEs -> data/derived/se_*.csv         (~10 min)
    lag-all         every lag profile -> lag_profiles.csv          (~20 min)
    reliability     per-player reliability (Table 2)               (~15 min)
    bins            effect within legal-move-count bins            (~10 min)
    describe        event descriptives                             (~2 min)
    robustness      the six registered specifications              (~30 min)
    verify-paper    manuscript values vs data/derived/    (instant) <- run
                                                          this before submitting

Supporting stages: prelim, lag, lag-split, ps6, did, mixed,
                   cx-fens, cx, cx-model, aggregate, anonymize
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE
#   - The whole pipeline is in this one file; stages correspond 1:1 to the
#     commands in the README.
#   - Each stage skips output that already exists, so an interrupted run can
#     be resumed.
#   - scan and extract take a shard range and can be split across sessions.
#   - Every path is in src/config.py. Do not write path strings here.
# ───────────────────────────────────────────────────────────

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src import config as C
from src.analysis import lagwise, summarize, per_player_at, cluster_stats
from src.analysis import build_with_pretrend, build_pairs, fit_mixed
from src.extract import (download, shard_url, scan_shard, scan_titled_shard,
                         extract_shard, sample_fens, build_all)
from src.prepare import (add_pre_speed, net_material_change, events_A, events_B,
                         events_B_see, mask_A, mask_B, calipers)

SORT_KEY = ["player", "game_id", "ply"]


# ══════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════

def _parallel(fn, jobs, workers, label):
    t = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        r = list(ex.map(fn, jobs))
    counts = {k: r.count(k) for k in ("ok", "skip", "fail", "excl") if r.count(k)}
    print(f"{label}: {len(jobs)} jobs / {time.time()-t:.0f}s  "
          + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return r


def _need(path, hint):
    if not os.path.exists(path):
        sys.exit(f"missing file: {path}\n{hint}")


def _calipers_from_main(with_prespeed=True, with_wp=False):
    """
    The SDs defining the calipers are taken once, over the lower four tiers,
    and used for every tier.

    ★ The SDs are computed **after dropping rows where z_pre3 is missing**.
      Every analysis runs on a frame that has had
      `.dropna(subset=[PRESPEED_VAR])` applied, but this function used to take
      its SDs over the whole frame. z_pre3 is missing for 9.9% of rows, and
      those rows cluster at the start of games (low ply, plenty of clock), so
      the two bases disagreed by up to 4%.

          variable      after dropna (paper)   whole frame (old code)
          wp_before            0.055514               0.053049
          clk_before          30.935437              31.491718
          n_legal              2.545109               2.483746
          ply                  5.958813               6.157324

      The published CSVs were produced on the dropna basis.
      tests/test_prep.py fixes this ordering.
    """
    base_vars = C.MATCH_VARS_WP if with_wp else C.MATCH_VARS
    cols = list(base_vars) + ([C.PRESPEED_VAR] if with_prespeed else [])
    base = pd.read_parquet(C.PREPARED, columns=cols)
    if with_prespeed:
        base = base.dropna(subset=[C.PRESPEED_VAR])
    sd = {v: base[v].std() for v in base.columns}
    del base
    cal = {v: C.CALIPER_SD * sd[v] for v in base_vars}
    if with_prespeed:
        cal[C.PRESPEED_VAR] = C.PRESPEED_CALIPER_SD * sd[C.PRESPEED_VAR]
    return cal, sd


def _events(d, which, use_see=False):
    if which.endswith("A"):
        return events_A(d), "blunder"
    if use_see:
        return events_B_see(d), "material_see"
    return events_B(d), "material"


def _slice(ev, a, b):
    ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
    return ev.iloc[a:b] if b is not None else ev.iloc[a:]


# ══════════════════════════════════════════════════════════════
# Stage 1 - account metadata
# ══════════════════════════════════════════════════════════════

def cmd_scan(args):
    """Aggregate account metadata over shards 0-421, without parsing games."""
    C.ensure_dirs(C.STAGE1, C.WORK)

    def work(i):
        p = f"{C.STAGE1}/s{i:05d}.parquet"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return "skip"
        tmp = os.path.join(C.WORK, f"_a{i}.parquet")
        if not download(shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i,
                                  C.MAIN_N_SHARDS), tmp):
            return "fail"
        try:
            scan_shard(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return "ok"

    todo = [i for i in range(args.start, args.end) if i not in C.PRELIM_SHARDS]
    _parallel(work, todo, 6, "scan")
    print("next: python run.py scan-titled <month> / python run.py sample")


def cmd_scan_titled(args):
    """
    Aggregate titled accounts only. With no month given, runs January to June.

    One month yields only 26 titled accounts with at least 30 games, which is
    too few to form a tier. The 136 in Table 1 come from six months. Finding
    titled accounts means reading every shard of each month, which makes this
    the slowest stage in the pipeline. Shards already downloaded are skipped,
    so it can be split across sessions.
    """
    C.ensure_dirs(C.STAGE1_FM, C.WORK)
    months = sorted(C.TITLED_MONTHS) if args.month is None else [args.month]
    for m in months:
        n = C.TITLED_MONTHS[m]
        end = args.end if args.end is not None else n
        jobs = [(m, i, n) for i in range(args.start, min(end, n))
                if not (m == 1 and i in C.PRELIM_SHARDS)]
        _parallel(scan_titled_shard, jobs, 5, f"scan-titled m{m:02d}")
    print("next: python run.py sample")


def cmd_sample(args):
    """Combine the stage 1 output into the final list of sampled players."""
    build_all()
    if os.path.exists(C.SAMPLE):
        print()
        _check_sample(pd.read_parquet(C.SAMPLE))


# ══════════════════════════════════════════════════════════════
# Stage 2 - replaying games
# ══════════════════════════════════════════════════════════════

def _check_sample(samp):
    """
    Check that the sample list is the one the manuscript reports.

    Aggregating the titled tier over one month gives 26 players; over six
    months it gives 136. Running stage 2 from the 26-player file produces a
    different sample from the paper's, and that file was in fact the one
    present in the repository. This makes that impossible to miss.
    """
    from src.paper_check import TABLE1, N_PLAYERS
    n_fm = int((samp.tier == C.TITLE_TIER).sum())
    want_fm = TABLE1[C.TITLE_TIER][3]
    if len(samp) != N_PLAYERS or n_fm != want_fm:
        print(f"  warning: sample differs from the manuscript - "
              f"{len(samp):,} players (paper: {N_PLAYERS:,}), "
              f"{n_fm} titled (paper: {want_fm}).", flush=True)
        if n_fm and n_fm < want_fm / 2:
            print(f"         the titled tier looks like a one-month "
                  f"aggregate. Run run.py scan-titled for all six months, "
                  f"then run.py sample again.", flush=True)
        print("         pass --force to proceed anyway.", flush=True)
        return False
    return True


def _extract_common(sample_path, out_dir, tier_filter, url_fn, jobs, label,
                    force=False):
    _need(sample_path, "run python run.py scan and sample first.")
    C.ensure_dirs(out_dir, C.WORK)
    samp = pd.read_parquet(sample_path)
    if not _check_sample(samp) and not force:
        sys.exit(1)
    keep = set(tier_filter(samp).player)
    tier_map = dict(zip(samp.player, samp.tier))
    print(f"{len(keep):,} players selected", flush=True)

    def work(job):
        out, url, tag = url_fn(job)
        if tag == "excl":
            return "excl"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return "skip"
        tmp = os.path.join(C.WORK, f"_x{tag}.parquet")
        if not download(url, tmp):
            return "fail"
        try:
            extract_shard(tmp, out, keep_players=keep, eval_only=False,
                          do_see=True, tier_map=tier_map)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return "ok"

    _parallel(work, jobs, 5, label)


def cmd_extract(args):
    """The lower four tiers. mat_diff is only filled when do_see=True."""
    def url_fn(i):
        return (f"{C.STAGE2}/s{i:05d}.parquet",
                shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i, C.MAIN_N_SHARDS),
                f"b{i}")
    jobs = [i for i in range(args.start, args.end) if i not in C.PRELIM_SHARDS]
    _extract_common(C.SAMPLE, C.STAGE2, lambda s: s[s.tier != C.TITLE_TIER],
                    url_fn, jobs, "extract", force=args.force)


def cmd_extract_titled(args):
    """The titled tier. All of January to June by default."""
    def url_fn(job):
        m, i = job
        if m == 1 and i in C.PRELIM_SHARDS:
            return None, None, "excl"
        return (f"{C.STAGE2_FM}/m{m:02d}_s{i:05d}.parquet",
                C.ARCHIVE_2024_URL.format(m=m, i=i, n=C.TITLED_MONTHS[m]),
                f"f{m}_{i}")

    if args.month is None:
        jobs = [(m, i) for m in sorted(C.TITLED_MONTHS)
                for i in range(C.TITLED_MONTHS[m])]
    else:
        end = args.end if args.end is not None else C.TITLED_MONTHS[args.month]
        jobs = [(args.month, i)
                for i in range(args.start, min(end, C.TITLED_MONTHS[args.month]))]
    _extract_common(C.SAMPLE_FM, C.STAGE2_FM, lambda s: s, url_fn, jobs,
                    "extract-titled", force=True)   # the FM file holds one tier


def cmd_prelim(args):
    """Preliminary extraction. Uses shards 0-23 only, so it does not overlap
    the main sample."""
    C.ensure_dirs(C.PRELIM, C.WORK)

    def work(i):
        p = f"{C.PRELIM}/s{i:05d}.parquet"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return "skip"
        tmp = os.path.join(C.WORK, f"_p{i}.parquet")
        if not download(shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i,
                                  C.MAIN_N_SHARDS), tmp):
            return "fail"
        try:
            extract_shard(tmp, p, keep_players=None, eval_only=False,
                          do_see=False)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return "ok"

    _parallel(work, list(range(args.start, args.end)), 4, "prelim")


# ══════════════════════════════════════════════════════════════
# Stage 3 - the analysis table
# ══════════════════════════════════════════════════════════════

KEEP_COLS = ["game_id", "player", "color", "tier", "ply", "move_time",
             "clk_before", "clk_after", "wp_before", "wp_after", "wp_delta",
             "see_loss", "n_legal", "mat_diff", "max_see_mine", "n_checks",
             "nag", "is_capture", "ply_total"]
KEY = ["game_id", "player", "ply"]


def cmd_prep(args):
    """
    plies -> the analysis table.

    The order matters:
      1. net_mat   <- **before** the measurement floor. mat_diff is position
                      information and has nothing to do with the clock, so
                      removing floor-excluded moves first would needlessly
                      lose ply+2 partners.
      2. the measurement floor (move_time >= TAU)
      3. standardise z within player
      4. z_pre3    <- can only be computed once z is defined
    """
    src_dir = C.STAGE2_FM if args.fm else C.STAGE2
    out = C.PREPARED_FM if args.fm else C.PREPARED
    C.ensure_dirs(out)

    files = sorted(glob.glob(f"{src_dir}/*.parquet"))
    if not files:
        sys.exit(f"no plies found in {src_dir}\n"
                 f"run python run.py extract first.")
    print(f"reading {len(files)} shards...", flush=True)
    parts, blank = [], 0
    for f in files:
        dd = pd.read_parquet(f)
        if len(dd) == 0:
            # A shard with no wanted game writes an empty frame, and an empty
            # frame's columns are all dtype object. Concatenating one of those
            # turns move_time into object for the whole table, and np.log then
            # fails with "loop of ufunc does not support argument 0 of type
            # float". Dropping them changes nothing and avoids the cast.
            #
            # The four untitled tiers never hit this -- every shard contained
            # one of the 1,878 players. The titled tier is sparse enough that
            # 119 of its 2,355 shards are empty.
            blank += 1
            continue
        parts.append(dd[[c for c in KEEP_COLS if c in dd.columns]])
    if not parts:
        sys.exit(f"all {len(files)} shards are empty: {src_dir}")
    if blank:
        print(f"  skipped {blank} empty shards", flush=True)
    d = pd.concat(parts, ignore_index=True)
    print(f"  {len(d):,} rows / {d.player.nunique():,} players / "
          f"{d.game_id.nunique():,} games", flush=True)

    t = time.time()
    dup = d.duplicated(subset=KEY).sum()
    if dup:
        print(f"  warning: removed {dup:,} duplicate rows", flush=True)
        d = d.drop_duplicates(subset=KEY)
    d = d.sort_values(KEY).reset_index(drop=True)

    if "mat_diff" not in d.columns or d.mat_diff.isna().all():
        sys.exit("mat_diff is empty - rerun extract with do_see=True.")
    d = net_material_change(d)
    n_ev = int((d.net_mat < 0).sum())
    print(f"  net_mat - {n_ev:,} loss events "
          f"({n_ev / max(d.game_id.nunique(), 1):.2f} per game)", flush=True)

    before = len(d)
    d = d[d.move_time >= C.TAU].copy()
    print(f"  measurement floor {C.TAU}s: {before:,} -> {len(d):,} rows "
          f"({100 * (before - len(d)) / before:.1f}% removed)", flush=True)

    d["y"] = np.log(d.move_time + C.LOG_OFFSET)
    g = d.groupby("player")["y"]
    d["z"] = (d.y - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    d = d.dropna(subset=["z"]).reset_index(drop=True)

    d = add_pre_speed(d, k=C.PRESPEED_K)
    print(f"  z_pre3 missing in {d.z_pre3.isna().sum():,} rows "
          f"({100 * d.z_pre3.isna().mean():.1f}%) - undefined early in a game",
          flush=True)

    d.to_parquet(out, index=False)
    print(f"\n{'FM+' if args.fm else 'lower four tiers'}: {len(d):,} rows -> {out}  "
          f"[{time.time()-t:.0f}s]")
    print("next: python run.py se split")


# ══════════════════════════════════════════════════════════════
# Main analysis - standard errors
# ══════════════════════════════════════════════════════════════

SE_LAGS = range(-3, 4)


def cmd_se(args):
    """
    Every standard error reported in the manuscript comes from here.

    Columns:
      e{k}    event-weighted mean (the value reported in the paper)
      se{k}   standard error clustered by player
      epw{k}  player-weighted mean
      np{k}   number of clusters (players)

    A large gap between e and epw means events are badly unbalanced across
    players.
    """
    _need(C.PREPARED, "run python run.py prep first.")
    d = pd.read_parquet(C.PREPARED).dropna(subset=[C.PRESPEED_VAR])

    # The titled tier joins every mode, not just the tier breakdown.
    #
    # Adding it cannot disturb the four untitled tiers: lagwise draws controls
    # only from the same player (by_player[r.player]) and discards players
    # with no events. Caliper widths still come from the untitled sample, so
    # every tier is matched on one ruler.
    #
    # --tiers four reproduces the untitled-only version for comparison.
    if args.tiers == "five" and os.path.exists(C.PREPARED_FM):
        fm = pd.read_parquet(C.PREPARED_FM).dropna(subset=[C.PRESPEED_VAR])
        keep = [c for c in d.columns if c in fm.columns]
        d = pd.concat([d[keep], fm[keep]], ignore_index=True)
        print(f"  titled tier included: +{len(fm):,} rows "
              f"({fm.player.nunique()} players)", flush=True)
    elif args.tiers == "four":
        print("  lower four tiers only (--tiers four)", flush=True)

    cal, _ = _calipers_from_main()
    hasA, hasB = mask_A(d), mask_B(d)

    def run(lab, ev, want_per_player=False):
        ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > C.MAX_EVENTS_SE:
            ev = (ev.sample(C.MAX_EVENTS_SE, random_state=C.EVENT_SAMPLE_SEED)
                    .sort_values(SORT_KEY).reset_index(drop=True))
        if len(ev) < C.MIN_EVENTS:
            return None, None
        p = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        out = {"label": lab, "n": len(ev)}
        for k in SE_LAGS:
            c = f"lag{k:+d}"
            st = cluster_stats(p[["player", c]].dropna(), c)
            if st is None:
                continue
            out[f"e{k}"], out[f"se{k}"] = st["effect"], st["se"]
            out[f"epw{k}"], out[f"np{k}"] = st["effect_pw"], st["n_players"]
        per = per_player_at(p, lag=1, label=lab) if want_per_player else None
        return out, per

    rows, pers = [], []
    if args.mode == "tier":
        for t in sorted(d.tier.unique()):
            dd = d[d.tier == t]
            for tag, m in [(f"B_{t}", mask_B(dd)), (f"A_{t}", mask_A(dd))]:
                r, _ = run(tag, dd[m])
                if r:
                    rows.append(r)
    elif args.mode == "split":
        for nm, m in [("blunder_only", hasA & ~hasB),
                      ("blunder_loss", hasA & hasB),
                      ("loss_only", hasB & ~hasA)]:
            r, per = run(nm, d[m], want_per_player=True)
            if r:
                rows.append(r)
            if per is not None:
                pers.append(per)
    else:  # dose - direction x size
        # Labels are self_N (the player lost) and opp_N (the opponent lost).
        # The direction and size columns are written out alongside so the
        # rows can be matched to the figures.
        for v in [1, 3, 5, 9]:
            for tag, direction, m in [(f"self_{v}", "self", d.net_mat == -v),
                                      (f"opp_{v}", "opp", d.net_mat == v)]:
                r, _ = run(tag, d[m])
                if r:
                    r["direction"], r["size"] = direction, v
                    rows.append(r)

    C.ensure_dirs(C.DERIVED)
    x = pd.DataFrame(rows)
    lead = [c for c in ["label", "direction", "size", "n"] if c in x.columns]
    x = x[lead + [c for c in x.columns if c not in lead]]
    path = os.path.join(C.DERIVED, f"se_{args.mode}.csv")
    x.to_csv(path, index=False)
    print(f"→ {path}")
    if pers:
        pp = os.path.join(C.DERIVED, "per_player_t1.csv")
        pd.concat(pers, ignore_index=True).to_csv(pp, index=False)
        print(f"→ {pp}")
    for r in x.itertuples():
        e1, se1 = getattr(r, "e1"), getattr(r, "se1")
        print("{:>18} n={:>6,}  t+1 {:+.4f} +/- {:.4f}  t={:>7.1f}  "
              "(player-weighted {:+.4f}, clusters {:,})".format(
                  r.label, r.n, e1, se1, e1 / se1,
                  getattr(r, "epw1"), int(getattr(r, "np1"))))


# ══════════════════════════════════════════════════════════════
# Decomposition by lag
# ══════════════════════════════════════════════════════════════

SPECS = ("none", "nlegal", "nlegal_cx", "zpre")
LAG_COLS = ["game_id", "player", "tier", "ply", "z", "wp_before", "clk_before",
            "n_legal", "see_loss", "wp_delta", "net_mat", "z_pre3", "cx_pred"]


def _load_for_lag(is_fm, spec, need_cx=False):
    import pyarrow.parquet as pq
    src = C.prepared_path(is_fm)
    _need(src, "run python run.py prep first.")
    avail = set(pq.ParquetFile(src).schema.names)
    d = pd.read_parquet(src, columns=[c for c in LAG_COLS if c in avail])
    if spec == "zpre":
        d = d.dropna(subset=[C.PRESPEED_VAR])
    if need_cx and "cx_pred" not in d.columns:
        sys.exit("no cx_pred column - run python run.py cx-model first.\n"
                 "(without it the specification silently becomes nlegal)")
    return d


def cmd_lag(args):
    """Lag profiles by group of tiers. Panels (a) and (b) of Figure 1."""
    if args.spec not in SPECS:
        sys.exit(f"spec must be one of {SPECS} (got {args.spec!r})")
    is_fm = args.which.startswith("fm")
    d = _load_for_lag(is_fm, args.spec, need_cx=(args.spec == "nlegal_cx"))

    _, sd = _calipers_from_main()
    if "cx_pred" in d.columns:
        sd["cx_pred"] = d.cx_pred.std()
    # The core set follows config.MATCH_VARS minus n_legal, which each spec
    # adds for itself. It used to name wp_before, clk_before and ply directly,
    # which broke the moment the main specification dropped win probability.
    core = {v: C.CALIPER_SD * sd[v]
            for v in C.MATCH_VARS if v != "n_legal"}
    if args.spec == "none":
        cal = {}
    elif args.spec == "nlegal":
        cal = {**core, "n_legal": C.CALIPER_SD * sd["n_legal"]}
    elif args.spec == "nlegal_cx":
        cal = {**core, "n_legal": C.CALIPER_SD * sd["n_legal"],
               "cx_pred": C.CALIPER_SD * sd["cx_pred"]}
    else:
        cal = {**core, "n_legal": C.CALIPER_SD * sd["n_legal"],
               C.PRESPEED_VAR: C.PRESPEED_CALIPER_SD * sd[C.PRESPEED_VAR]}

    ev_all, et = _events(d, args.which, args.see)

    # With --by-tier each tier is run separately. Fig 1a/1b draw one curve
    # per tier for all five, but an earlier version only ever pooled the lower
    # four into a single "lower" group.
    if args.by_tier:
        tiers = [t for t in sorted(ev_all.tier.dropna().unique())]
    else:
        tiers = [None]

    C.ensure_dirs(C.WORK)
    parts = []
    for tier in tiers:
        ev = ev_all if tier is None else ev_all[ev_all.tier == tier]
        ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > args.nmax:
            ev = (ev.sample(args.nmax, random_state=C.EVENT_SAMPLE_SEED)
                    .sort_values(SORT_KEY).reset_index(drop=True))
        if len(ev) < C.MIN_EVENTS:
            print(f"  skipping {tier}: only {len(ev)} events")
            continue
        dd = d[d.player.isin(set(ev.player))]
        tier_label = tier if tier is not None else (
            "fm_plus" if is_fm else "lower")
        print(f"  {tier_label}: {len(dd):,} rows / {len(ev):,} events",
              flush=True)
        t = time.time()
        part = summarize(lagwise(dd, ev, cal, seed=C.MATCH_SEED),
                         et, tier_label, args.spec)
        if part.empty:
            # summarize drops any lag with fewer than min_n observations.
            # If every lag goes, the frame is empty and must not be concatenated.
            print(f"    too few observations - skipped", flush=True)
            continue
        parts.append(part)
        print(f"    [{time.time()-t:.0f}s]", flush=True)

    if not parts:
        print("nothing produced (too few observations in every tier)")
        return
    res = pd.concat(parts, ignore_index=True)
    out = (f"{C.WORK}/lag_{args.which}_{args.spec}"
           f"{'_bytier' if args.by_tier else ''}"
           f"{'_see' if args.see else ''}.parquet")
    res.to_parquet(out, index=False)
    print(f"{et}/{args.spec} → {out}")
    cols = ["tier", "lag", "effect", "se", "t", "n_events", "n_players"]
    print(res[cols].to_string(index=False))


def cmd_lag_all(args):
    """Run all 2 events x 2 tier groups x 4 specifications, then combine
    them into one CSV."""
    import pyarrow.parquet as pq
    failed = []
    for which in ["A", "B", "fmA", "fmB"]:
        src = C.prepared_path(which.startswith("fm"))
        if not os.path.exists(src):
            print(f"skipping {which}: {src} is absent")
            continue
        has_cx = "cx_pred" in set(pq.ParquetFile(src).schema.names)
        for spec in SPECS:
            if spec == "nlegal_cx" and not has_cx:
                print(f"skipping {which}/{spec}: no cx_pred (run run.py cx-model)")
                continue
            print(f"\n--- {which} / {spec}", flush=True)
            try:
                cmd_lag(argparse.Namespace(which=which, spec=spec,
                                           nmax=C.MAX_EVENTS_LAG, see=False,
                                           by_tier=False))
            except SystemExit as e:
                if e.code:
                    failed.append(f"{which}/{spec}")
    # Tier-wise profiles (Fig 1a/1b), once each under the main specification
    for which in ["A", "B"]:
        if not os.path.exists(C.prepared_path(False)):
            break
        print(f"\n--- {which} / zpre / by tier", flush=True)
        try:
            cmd_lag(argparse.Namespace(which=which, spec="zpre",
                                       nmax=C.MAX_EVENTS_LAG, see=False,
                                       by_tier=True))
        except SystemExit as e:
            if e.code:
                failed.append(f"{which}/zpre/by-tier")

    cmd_aggregate(args)
    if failed:
        print(f"\nfailed combinations: {', '.join(failed)}")


def cmd_lag_split(args):
    """
    Lag decomposition split into A-only, B-only and A-and-B. Panel (c) of
    Figure 1.

    ★ Note on the specification. An earlier version ran under a **different**
      specification from the main result (run.py se split): a four-variable
      caliper with no z_pre3, and players truncated to 110 per tier. The two
      outputs covered the same three groups yet could not be compared, which
      is why the manuscript's effects (+0.123 and so on) and standard errors
      (0.026 and so on) came from different runs.

      The default now matches the main result: player plus four variables plus
      z_pre3. Pass --no-prespeed to reproduce the old specification.
    """
    is_fm = args.which == "fm"
    spec = "nlegal" if args.no_prespeed else "zpre"
    d = _load_for_lag(is_fm, spec)
    cal, _ = _calipers_from_main(with_prespeed=not args.no_prespeed)

    hasA, hasB = mask_A(d), mask_B(d)
    grp = {"Aonly": (hasA & ~hasB, "blunder_only"),
           "Bonly": (~hasA & hasB, "material_only"),
           "AB": (hasA & hasB, "both")}
    if args.grp not in grp:
        sys.exit("grp must be one of Aonly, Bonly, AB")
    m, et = grp[args.grp]
    ev = d[m].sort_values(SORT_KEY).reset_index(drop=True)

    # Cap memory by sampling players
    rng = np.random.default_rng(C.EVENT_SAMPLE_SEED)
    plt_n = int(os.environ.get(
        "PLT", 136 if is_fm else C.PLAYERS_PER_TIER_LAGSPLIT))
    keep = set()
    for _, g in ev.groupby("tier"):
        u = g.player.unique()
        keep |= set(rng.choice(u, min(plt_n, len(u)), replace=False))
    ev = ev[ev.player.isin(keep)].reset_index(drop=True)
    d = d[d.player.isin(keep)]

    C.ensure_dirs(C.WORK)
    out = (f"{C.WORK}/lagsplit_{args.which}_{args.grp}"
           f"{'_nopre' if args.no_prespeed else ''}.parquet")
    if os.path.exists(out):
        print("skip")
        return
    t = time.time()
    s = summarize(lagwise(d, ev, cal, seed=C.MATCH_SEED), et,
                  "fm_plus" if is_fm else "lower", spec)
    s.to_parquet(out, index=False)
    print(f"{et}/{args.which} [{time.time()-t:.0f}s] n={len(ev):,} → {out}")
    print(" ".join(f"{r.lag:+d}:{r.effect:+.3f}({r.t:.1f})"
                   for r in s.itertuples()))


# ══════════════════════════════════════════════════════════════
# Supporting analyses
# ══════════════════════════════════════════════════════════════

def cmd_ps6(args):
    """Per-event pre, at and post values under the main specification
    (pre-event speed matched within 0.6 SD)."""
    is_fm = args.which.startswith("fm")
    src = C.prepared_path(is_fm)
    _need(src, "run python run.py prep first.")
    d = pd.read_parquet(src).dropna(subset=[C.PRESPEED_VAR])
    cal, _ = _calipers_from_main()
    ev, _et = _events(d, args.which)
    ev = _slice(ev, args.start, args.end)
    if len(ev) == 0:
        print("empty")
        return

    C.ensure_dirs(C.WORK)
    out = f"{C.WORK}/ps6_{args.which}_{args.start:07d}.parquet"
    if os.path.exists(out):
        print("skip")
        return
    t = time.time()
    e = build_with_pretrend(d, ev, cal, seed=C.MATCH_SEED).reset_index(drop=True)
    e["size"] = ev["net_mat"].values
    e["wp_delta"] = ev["wp_delta"].values
    e.to_parquet(out, index=False)
    ok = e.dropna(subset=["effect"])
    print("{} {:,} events {:.0f}s  matched {:.0f}%  pre{:+.4f}  eff{:+.4f}".format(
        args.which, len(ev), time.time() - t,
        100 * len(ok) / len(e), ok.pretrend.mean(), ok.effect.mean()))


def cmd_did(args):
    """Difference-in-differences. Leaving z_pre3 out of the caliper is the
    point of this specification."""
    is_fm = args.which.startswith("fm")
    src = C.prepared_path(is_fm)
    _need(src, "run python run.py prep first.")
    cols = [c for c in LAG_COLS if c != "cx_pred"]
    d = pd.read_parquet(src, columns=cols)
    if is_fm:
        main = pd.read_parquet(C.PREPARED, columns=list(C.MATCH_VARS))
        cal = calipers(main)
        del main
    else:
        cal = calipers(d)

    ev, _et = _events(d, args.which)
    ev = _slice(ev, args.start, args.end)
    if len(ev) == 0:
        print("empty")
        return

    C.ensure_dirs(C.WORK)
    out = f"{C.WORK}/did_{args.which}_{args.start:07d}.parquet"
    if os.path.exists(out):
        print("skip")
        return
    t = time.time()
    e = build_with_pretrend(d, ev, cal, seed=C.MATCH_SEED)
    e.to_parquet(out, index=False)
    ok = e.dropna(subset=["effect", "pretrend"])
    print("{} {:,} events {:.0f}s  pre{:+.4f} post{:+.4f} did{:+.4f}".format(
        args.which, len(ev), time.time() - t,
        ok.pretrend.mean(), ok.effect.mean(), ok.did.mean()))


def cmd_mixed(args):
    """
    Preregistered mixed-effects model.

    Its job is to show that the effect does not depend on the matching: the
    same quantity, estimated with player random effects instead. So it uses
    the same dependent variable as everything else -- the move at t+1 -- and
    the same five matching variables as controls.

    Three rows:
      mixed model          random intercept and slope by player
      paired difference    the same sample, event minus its matched controls
      with wp (robustness)  the registered specification, on the engine-
                            evaluated subsample where it can be computed

    The earlier version reported "with wp" and "without wp" as two headline
    numbers. That split existed because win probability sat in the caliper;
    once it moved to robustness the split had nothing left to mean.

    ★ Always report which path the fit took. A model that falls back has not
      converged, and the number is a paired difference wearing its name.
    """
    is_fm = args.which.startswith("fm")
    src = C.prepared_path(is_fm)
    _need(src, "run python run.py prep first.")
    d = pd.read_parquet(src).dropna(subset=[C.PRESPEED_VAR])
    ev_all, _et = _events(d, args.which)
    ev_all = ev_all.sort_values(SORT_KEY).reset_index(drop=True)
    print(f"{len(ev_all):,} events", flush=True)

    activity = (pd.read_parquet(C.SAMPLE)[["player", "n_games"]]
                if os.path.exists(C.SAMPLE) else None)

    rows = []

    def one(tag, frame, events, cal, fit_model=True, with_wp=False):
        t = time.time()
        long_df = build_pairs(frame, events, cal, seed=C.EVENT_SAMPLE_SEED,
                              max_events=args.max_events)
        n_pairs = len(long_df) // 2
        if n_pairs == 0:
            print(f"  skipping {tag}: no pairs", flush=True)
            return
        w = long_df.pivot_table(index="eid", columns="event_condition",
                                values="z_dv").dropna()
        diff = w[1] - w[0]
        pl = long_df.drop_duplicates("eid").set_index("eid").player
        per = diff.groupby(pl).mean()
        rows.append({"spec": f"paired difference ({tag})",
                     "coefficient": float(diff.mean()),
                     "se": float(per.sem()), "convergence": "n/a",
                     "n_obs": len(diff), "n_players": int(per.size)})
        print(f"  paired difference ({tag}) {diff.mean():+.4f} +/- {per.sem():.4f}  "
              f"{len(diff):,} pairs / {per.size:,} players  "
              f"[{time.time()-t:.0f}s]", flush=True)

        if not fit_model:
            return
        out, _fit = fit_mixed(long_df, activity=activity,
                              with_wp=with_wp)
        rows.append({"spec": f"mixed model ({tag})",
                     "coefficient": float(out.get("coef", float("nan"))),
                     "se": float(out.get("se", float("nan"))),
                     "convergence": out.get("path", "random slope")
                     if out.get("converged") else
                     f"NOT CONVERGED -- {out.get('path')}",
                     "n_obs": int(out.get("n_obs", 0)),
                     "n_players": int(out.get("n_players", 0))})
        print(f"  mixed model ({tag}) {out.get('coef', float('nan')):+.4f} "
              f"+/- {out.get('se', float('nan')):.4f}  "
              f"path {out.get('path')}  converged {out.get('converged')}",
              flush=True)

    cal5, _ = _calipers_from_main()
    one("main", d, ev_all, cal5)

    # Registered specification, where it exists.
    cal6, _ = _calipers_from_main(with_wp=True)
    sub = d.dropna(subset=["wp_before"])
    ev_sub = ev_all[ev_all.player.isin(sub.player.unique())]
    ev_sub = ev_sub.dropna(subset=["wp_before"]).reset_index(drop=True)
    one("with wp, engine-eval subset", sub, ev_sub, cal6, with_wp=True)

    C.ensure_dirs(C.DERIVED)
    res = pd.DataFrame(rows)
    path = os.path.join(C.DERIVED, f"mixed_{args.which}.csv")
    res.to_csv(path, index=False)
    print(f"\n→ {path}")
    print(res.to_string(index=False))


def cmd_cx_fens(args):
    """The plies tables hold no FENs, so games are replayed to recover the
    positions."""
    C.ensure_dirs(C.WORK, C.CX_FENS)
    keep = tier_map = None
    if os.path.exists(C.SAMPLE):
        samp = pd.read_parquet(C.SAMPLE)
        keep = set(samp[samp.tier != C.TITLE_TIER].player)
        tier_map = dict(zip(samp.player, samp.tier))

    parts, t = [], time.time()
    for i in range(args.start, args.end):
        tmp = os.path.join(C.WORK, f"_cx{i}.parquet")
        if not download(shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i,
                                  C.MAIN_N_SHARDS), tmp):
            print(f"  [{i}] download FAILED", flush=True)
            continue
        try:
            part = sample_fens(tmp, n=2000, seed=i,
                               keep_players=keep, tier_map=tier_map)
            parts.append(part)
            print(f"  [{i}] {len(part):,} positions", flush=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if not parts:
        sys.exit("no positions were sampled.")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(C.CX_FENS, index=False)
    print(f"{len(out):,} positions / {time.time()-t:.0f}s -> {C.CX_FENS}")


def cmd_cx(args):
    """Measure position complexity with Stockfish multi-PV. Requires the
    engine."""
    from src.complexity import Engine, complexity, gap_top2
    _need(C.CX_FENS, "run python run.py cx-fens first.")
    s = pd.read_parquet(C.CX_FENS)
    end = args.end if args.end is not None else len(s)
    s = s.iloc[args.start:end]

    C.ensure_dirs(C.WORK)
    out = f"{C.WORK}/cx_{args.start:05d}.parquet"
    if os.path.exists(out):
        print("skip")
        return
    recs, t = [], time.time()
    with Engine() as e:
        for r in s.itertuples():
            sc = e.analyse(r.fen, depth=args.depth)
            recs.append({"tier": r.tier, "ply": r.ply, "n_legal": r.n_legal,
                         "max_see_mine": getattr(r, "max_see_mine", None),
                         "mat_diff": getattr(r, "mat_diff", None),
                         "n_checks": getattr(r, "n_checks", None),
                         "sd": complexity(sc), "gap": gap_top2(sc),
                         "n_pv": len(sc)})
    pd.DataFrame(recs).to_parquet(out, index=False)
    print(f"[{args.start}:{end}] {len(recs)} positions {time.time()-t:.0f}s -> {out}")


def cmd_cx_model(args):
    """
    Fit the complexity prediction model and attach cx_pred to
    prepared*.parquet.

    ★ This model was missing from the original repository. It has been
      reconstructed from the four features stage 2 happened to record, so the
      output below must be checked against the r = 0.47 the paper reports.
    """
    from src.complexity import (CX_FEATURES, fit_cx_model, predict_cx,
                                save_model)
    files = sorted(glob.glob(f"{C.WORK}/cx_*.parquet"))
    if not files:
        sys.exit(f"no complexity measurements: {C.WORK}/cx_*.parquet\n"
                 f"run run.py cx-fens then run.py cx first.")
    cx = pd.concat([pd.read_parquet(f) for f in files],
                   ignore_index=True).dropna(subset=["sd"])
    print(f"{len(cx):,} positions measured")
    ok = cx.dropna(subset=["n_legal", "sd"])
    print(f"  n_legal ~ complexity   r = {np.corrcoef(ok.n_legal, ok.sd)[0,1]:+.3f}"
          f"   (paper reports -0.16)")
    model = fit_cx_model(cx)
    print(f"  prediction model       r = {model['r']:+.3f}   (paper reports +0.47)")
    print(f"  features: {model['features']}")
    print(f"  coefficients: {[round(c, 5) for c in model['coef']]}  "
          f"intercept {model['intercept']:.4f}  n={model['n']:,}")

    C.ensure_dirs(C.CX_MODEL)
    save_model(model, C.CX_MODEL)
    print(f"→ {C.CX_MODEL}")
    for path in (C.PREPARED, C.PREPARED_FM):
        if not os.path.exists(path):
            continue
        d = pd.read_parquet(path)
        missing = [f for f in CX_FEATURES if f not in d.columns]
        if missing:
            print(f"  skipping {os.path.basename(path)}: missing features {missing}")
            continue
        d["cx_pred"] = predict_cx(d, model)
        d.to_parquet(path, index=False)
        print(f"  cx_pred added -> {path} "
              f"({100 * d.cx_pred.isna().mean():.1f}% missing)")


# ══════════════════════════════════════════════════════════════
# Folding per-run output into data/derived/
# ══════════════════════════════════════════════════════════════

def cmd_aggregate(args):
    """
    Combine the per-run parquet files in {WORK} into one CSV under
    data/derived/.

    event_type in lag_profiles.csv distinguishes material from material_see.
    An earlier version wrote both as material, which made results under the
    withdrawn SEE definition look like results under the paper's definition
    (-0.205 at t+1, against the paper's -0.587).
    """
    C.ensure_dirs(C.DERIVED)
    jobs = [(f"{C.WORK}/lag_*.parquet", "lag_profiles.csv"),
            (f"{C.WORK}/lagsplit_*.parquet", "lagsplit_profiles.csv")]
    for pattern, name in jobs:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"skipping {name}: no files match {pattern}")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        cols = [c for c in ["event_type", "tier", "spec", "lag"]
                if c in df.columns]
        if cols:
            df = df.sort_values(cols).reset_index(drop=True)
        path = os.path.join(C.DERIVED, name)
        df.to_csv(path, index=False)
        print(f"-> {path}  ({len(files)} files, {len(df):,} rows)")
        if cols:
            for k, v in df.groupby(cols[:-1]).size().items():
                print(f"     {k}  {v} rows")


def cmd_anonymize(args):
    """
    Replace any remaining Lichess account names with salted hashes.

    Account names are public under CC0, so redistributing them is lawful in
    itself. Publishing per-person behavioural estimates attached to
    identifiable accounts is a separate question. Whether to do that is the
    researcher's call, so this never runs automatically.
    """
    C.ensure_dirs(C.WORK)
    salt_file = os.path.join(C.WORK, "anon_salt.txt")
    if os.path.exists(salt_file):
        salt = open(salt_file, encoding="utf-8").read().strip()
    else:
        salt = os.urandom(16).hex()
        open(salt_file, "w", encoding="utf-8").write(salt)
        print(f"new salt written to {salt_file}  (losing it makes the mapping\n"
              f"irreversible)")

    map_file = os.path.join(C.WORK, "player_mapping.json")
    mapping = (json.load(open(map_file, encoding="utf-8"))
               if os.path.exists(map_file) else {})

    targets = [(C.WORK, "sample.parquet"), (C.WORK, "sample_fm.parquet"),
               (C.DERIVED, "per_player_t1.csv")]
    for base, name in targets:
        path = os.path.join(base, name)
        if not os.path.exists(path):
            print(f"skipping {name}: file not found")
            continue
        read = pd.read_parquet if name.endswith(".parquet") else pd.read_csv
        d = read(path)
        if "player" not in d.columns:
            print(f"skipping {name}: no player column")
            continue
        names = d["player"].astype(str)
        if names.str.fullmatch(r"P[0-9a-f]{12}").all():
            print(f"skipping {name}: already anonymised")
            continue
        for n in names.unique():
            mapping.setdefault(n, "P" + hashlib.sha256(
                (salt + n).encode("utf-8")).hexdigest()[:12])
        print(f"{name}: {len(d):,} rows / {names.nunique():,} accounts "
              f"(e.g. {names.iloc[0]} -> {mapping[names.iloc[0]]})")
        if args.dry_run:
            continue
        d["player"] = names.map(mapping)
        (d.to_parquet if name.endswith(".parquet") else d.to_csv)(
            path, index=False)

    if args.dry_run:
        print("\n--dry-run: nothing was changed.")
        return
    json.dump(mapping, open(map_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n{len(mapping):,} mappings -> {map_file}")
    print("This file and the salt are in .gitignore. Never commit them to a\n"
          "public repository.")


# ══════════════════════════════════════════════════════════════
# The remaining analyses the manuscript reports
# ══════════════════════════════════════════════════════════════
# The five stages below did not exist in the original repository. The paper
# reported their results with no code that produced them, so those numbers
# survived only as transcriptions in a reference table. That table is now
# src/external_values.py. See docs/corrections.md.


def _events_with_effect(d, cal, ev, seed=None):
    """Per-event effects, carrying game_id and ply, which the reliability
    split-half needs."""
    from src.analysis import build_with_pretrend
    return build_with_pretrend(d, ev, cal, seed=seed or C.MATCH_SEED)


def _select_and_truncate(ev, threshold, seed=0):
    """
    Keep players with at least `threshold` events, then cut every one of them
    to exactly that many.

    A global cap cannot be used here. It divides a fixed budget among however
    many players qualify, so the observations per person -- the thing
    reliability is a function of -- depend on how many players there are. The
    published table came from a run that had quietly subsampled the player
    list (every third player, first 230), which left 443 events each; running
    the same command on all 607 players left 148 each and a different answer.

    Truncating per player fixes the observations per person regardless of how
    many qualify, so the number is reproducible from the rule alone.
    """
    rng = np.random.default_rng(seed)
    cnt = ev.groupby("player").size()
    sel = sorted(cnt[cnt >= threshold].index)
    if not sel:
        return ev.iloc[:0], 0, len(cnt)
    sub = ev[ev.player.isin(sel)].reset_index(drop=True)
    idx = []
    for _, g in sub.groupby("player", sort=True):
        idx.extend(rng.choice(g.index.values, threshold, replace=False))
    out = sub.loc[sorted(idx)].sort_values(SORT_KEY).reset_index(drop=True)
    return out, len(sel), len(cnt)


def cmd_reliability(args):
    """
    Individual-level reliability - Table 2 and Fig 2d/2e.

    The split is by game, Spearman-Brown corrected. Splitting by event would
    inflate reliability, because events within a game are correlated.

    The curve by observation count remeasures reliability with each player
    truncated to N events. The paper's claim -- that reliability for blunders
    does not rise as observations accumulate -- only holds as a comparison at
    equal counts.

    Writes: data/derived/reliability.csv        (Table 2)
          data/derived/reliability_curve.csv  (Fig 2d)
    """
    from src.analysis import reliability_curve, lagwise
    from src.prepare import reliability

    rows, curves = [], []
    for tier_label, is_fm in [("lower", False), ("titled", True)]:
        src = C.prepared_path(is_fm)
        if not os.path.exists(src):
            print(f"skipping {tier_label}: {src} is absent")
            continue
        d = pd.read_parquet(src).dropna(subset=[C.PRESPEED_VAR])
        cal, _ = _calipers_from_main()
        for ev_label, which in [("Material loss", "B"), ("Blunder", "A")]:
            ev, _et = _events(d, which)
            thr = (args.select if args.select is not None
                   else C.RELIABILITY_SELECT[which])
            ev, n_sel, n_all = _select_and_truncate(
                ev, thr, seed=C.EVENT_SAMPLE_SEED)
            print(f"  {ev_label}, {tier_label}: {n_sel} players with >={thr} "
                  f"events (of {n_all}) -> truncated to {thr} each, "
                  f"{len(ev):,} events", flush=True)
            if len(ev) < C.MIN_EVENTS:
                print(f"skipping {ev_label}/{tier_label}: only {len(ev)} events")
                continue
            print(f"  {ev_label}, {tier_label}: processing {len(ev):,} events...",
                  flush=True)
            # The per-event effect is the t+1 difference, as everywhere else.
            #
            # This used to take build_with_pretrend's z_post, the mean of
            # t+1..t+3. Averaging cuts measurement error but dilutes the
            # signal too, and reliability is their ratio: material loss read
            # .393 that way against .685 on t+1, and titled blunders +0.283
            # against -0.032. Table 2's sigma_b is discussed beside the
            # tier-wise t+1 effects, so it has to be the same quantity.
            #
            # (Either window gives the same answer on the tier comparison:
            # between-player SD is within 7% across tiers in both.)
            prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
            e = prof[["player", "lag+1"]].copy()
            e.columns = ["player", "effect"]
            e["game_id"] = ev["game_id"].values
            e["ply"] = ev["ply"].values
            e = e.dropna(subset=["effect"])
            rel = reliability(e, min_events=args.min_events) or {}
            per = e.groupby("player")["effect"].size()
            rows.append({
                "label": f"{ev_label}, {tier_label}",
                "reliability": rel.get("sb", np.nan),
                "r_halves": rel.get("r", np.nan),
                "sigma_b": rel.get("sigma_b", np.nan),
                "sigma_w": rel.get("sigma_w", np.nan),
                "events_per_player": float(per.mean()) if len(per) else np.nan,
                "n_players": int(e.player.nunique()),
                "n_events": int(len(e)),
            })
            curves.append(reliability_curve(
                e, seed=C.EVENT_SAMPLE_SEED,
                label=f"{ev_label}, {tier_label}"))

    C.ensure_dirs(C.DERIVED)
    if rows:
        t2 = pd.DataFrame(rows)
        path = os.path.join(C.DERIVED, "reliability.csv")
        t2.to_csv(path, index=False)
        print(f"\n→ {path}")
        print(t2[["label", "reliability", "sigma_b", "events_per_player",
                  "n_players"]].to_string(index=False))
    if curves:
        cv = pd.concat(curves, ignore_index=True)
        path = os.path.join(C.DERIVED, "reliability_curve.csv")
        cv.to_csv(path, index=False)
        print(f"→ {path}")


def cmd_bins(args):
    """
    Effect within bins of change in legal move count - Fig 2b.

    Legal move count moves in opposite directions depending on whose material
    disappeared. Comparing the two directions inside a bin that holds that
    change fixed therefore rules out "the position simply got simpler, so the
    player moved faster".

    --size 9 keeps only nine-point (queen) losses. The manuscript's "widens
    the separation to -1.04 at its maximum" is a value from that run, and no
    stage in the repository produced it: legal_bins_queen.csv was a file with
    no source.

    Writes: data/derived/legal_bins.csv          (without --size)
          data/derived/legal_bins_queen.csv    (--size 9)
    """
    from src.analysis import bin_effects, lagwise
    from src.prepare import net_legal_change

    _need(C.PREPARED, "run python run.py prep first.")
    d = pd.read_parquet(C.PREPARED).dropna(subset=[C.PRESPEED_VAR])
    d = net_legal_change(d)
    cal, _ = _calipers_from_main()

    bins = [-np.inf, -12, -8, -4, -1, 1, 4, 8, np.inf]
    sz = getattr(args, "size", None)
    if sz:
        masks = [("self_lost", d.net_mat == -sz), ("opp_lost", d.net_mat == sz)]
        print(f"  {sz}-point losses only", flush=True)
    else:
        masks = [("self_lost", d.net_mat < 0), ("opp_lost", d.net_mat > 0)]
    out = []
    for label, m in masks:
        ev = d[m].sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > args.max_events:
            ev = ev.sample(args.max_events, random_state=C.EVENT_SAMPLE_SEED)
            ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) == 0:
            print(f"  skipping {label}: no such events")
            continue
        print(f"  {label}: processing {len(ev):,} events...", flush=True)

        # The effect is the t+1 difference, as in every other stage.
        #
        # This used to take build_with_pretrend's z_post, which averages
        # t+1..t+3. Because t+2 and t+3 sit near zero, that divides the
        # effect by roughly three: the same events gave -0.19 here and -0.51
        # under se split. Figure 2b shares its axis with the other panels, so
        # it has to be the same quantity.
        prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        e = prof[["player", "lag+1"]].copy()
        e.columns = ["player", "effect"]
        e["net_legal"] = ev["net_legal"].values
        out.append(bin_effects(e, "net_legal", bins, label=label))

        mean_change = ev.groupby(ev.net_mat.abs())["net_legal"].mean()
        for v in (1, 3, 5, 9):
            if v in mean_change.index:
                print(f"     mean change in legal moves after a {v}-point loss: "
                      f"{mean_change.loc[v]:+.2f}")

    if not out:
        sys.exit("no bins produced - not enough events.")
    C.ensure_dirs(C.DERIVED)
    res = pd.concat(out, ignore_index=True)
    path = os.path.join(C.DERIVED,
                        "legal_bins_queen.csv" if sz else "legal_bins.csv")
    res.to_csv(path, index=False)
    print(f"\n→ {path}")
    print(res[["label", "bin", "effect", "se", "n_events"]].to_string(index=False))


def cmd_describe(args):
    """
    Event characteristics - the descriptive statistics in Results.

      - win-probability drop and material lost, for each of the three groups
      - the share of losses of nine points or more
      - how often a blunder is followed by another blunder
        (the Discussion's "followed by further blunders at an elevated rate";
         the manuscript does not report this value in Results)

    Writes: data/derived/event_characteristics.csv
    """
    _need(C.PREPARED, "run python run.py prep first.")
    d = pd.read_parquet(C.PREPARED).dropna(subset=[C.PRESPEED_VAR])
    hasA, hasB = mask_A(d), mask_B(d)
    groups = {"blunder_only": hasA & ~hasB,
              "blunder_loss": hasA & hasB,
              "loss_only": hasB & ~hasA}

    # Is the player's next move a blunder?
    nxt_blunder = pd.Series(
        dict(zip(zip(d.game_id, d.player, d.ply), hasA.values)))
    key_next = list(zip(d.game_id, d.player, d.ply + 2))
    d = d.assign(next_is_blunder=[nxt_blunder.get(k, np.nan) for k in key_next])

    rows = []
    for name, m in groups.items():
        g = d[m]
        loss = g.net_mat[g.net_mat < 0].abs()
        nb = g.next_is_blunder.dropna()
        rows.append({
            "group": name,
            "n_events": int(len(g)),
            "wp_delta_mean": float(g.wp_delta.mean()),
            "material_lost_mean": float(loss.mean()) if len(loss) else np.nan,
            "material_lost_median": float(loss.median()) if len(loss) else np.nan,
            "share_ge_9": float((loss >= 9).mean()) if len(loss) else np.nan,
            "next_blunder_rate": float(nb.mean()) if len(nb) else np.nan,
            "n_next_obs": int(len(nb)),
        })
    base = d.next_is_blunder.dropna()
    rows.append({"group": "(all moves)", "n_events": int(len(d)),
                 "wp_delta_mean": float(d.wp_delta.mean()),
                 "material_lost_mean": np.nan, "material_lost_median": np.nan,
                 "share_ge_9": np.nan,
                 "next_blunder_rate": float(base.mean()) if len(base) else np.nan,
                 "n_next_obs": int(len(base))})

    C.ensure_dirs(C.DERIVED)
    res = pd.DataFrame(rows)
    path = os.path.join(C.DERIVED, "event_characteristics.csv")
    res.to_csv(path, index=False)
    print(f"→ {path}")
    print(res.to_string(index=False))
    print("\nNote: next_blunder_rate is defined only on moves that carry an "
          "engine evaluation. Read it beside the base rate (all moves).")


def cmd_robustness(args):
    """
    Rerun the six preregistered specifications at alternative levels -
    the Robustness section.

      opening cut         10 / 14 / 20 plies
      measurement floor   1 / 2 / 3 seconds   <- z is restandardised
      blunder threshold   10 / 20 / 30 pp
      matching caliper    0.1 / 0.2 / 0.4 / 0.8 SD
      complexity control  read from run.py lag none / nlegal / nlegal_cx
      pre-event speed     read from run.py lag zpre and run.py did

    Writes: data/derived/robustness.csv
    """
    _need(C.PREPARED, "run python run.py prep first.")
    base = pd.read_parquet(C.PREPARED)

    def effect_at_t1(d, cal, which, n=None):
        d = d.dropna(subset=[C.PRESPEED_VAR])
        ev, _ = _events(d, which)
        ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        n = n or args.max_events
        if len(ev) > n:
            ev = ev.sample(n, random_state=C.EVENT_SAMPLE_SEED)
            ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) < C.MIN_EVENTS:
            return None
        prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        return cluster_stats(prof[["player", "lag+1"]].dropna(), "lag+1")

    def restandardise(d, tau):
        """Changing the measurement floor requires recomputing z."""
        from src.prepare import add_pre_speed
        x = d[d.move_time >= tau].copy()
        x["y"] = np.log(x.move_time + C.LOG_OFFSET)
        g = x.groupby("player")["y"]
        x["z"] = (x.y - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        x = x.dropna(subset=["z"]).reset_index(drop=True)
        return add_pre_speed(x, k=C.PRESPEED_K)

    rows = []

    def record(spec, level, which, st):
        if st is None:
            return
        rows.append({"spec": spec, "level": level, "event": which,
                     "effect": st["effect"], "se": st["se"],
                     "effect_pw": st["effect_pw"],
                     "n_events": st["n_events"], "n_players": st["n_players"]})
        print(f"  {spec:<18}{str(level):<8}{which:<4}"
              f"{st['effect']:+.4f} ± {st['se']:.4f}", flush=True)

    print("opening cut")
    for cut in C.SENS_OPENING_CUT:
        d = base[base.ply >= cut]
        cal, _ = _calipers_from_main()
        record("opening_cut", cut, "B", effect_at_t1(d, cal, "B"))

    print("measurement floor")
    for tau in C.SENS_TAU:
        d = base if tau == C.TAU else restandardise(base, tau)
        cal, _ = _calipers_from_main()
        record("measurement_floor", tau, "B", effect_at_t1(d, cal, "B"))

    print("blunder threshold")
    d = base.dropna(subset=[C.PRESPEED_VAR])
    cal, _ = _calipers_from_main()
    for th in C.GRID_BLUNDER_THRESH:
        ev = events_A(d, thresh=th).sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > args.max_events:
            ev = ev.sample(args.max_events, random_state=C.EVENT_SAMPLE_SEED)
            ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) < C.MIN_EVENTS:
            continue
        prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        record("blunder_threshold", th, "A",
               cluster_stats(prof[["player", "lag+1"]].dropna(), "lag+1"))

    print("matching caliper")
    _, sd = _calipers_from_main()
    for mult in C.SENS_CALIPER:
        cal = {v: mult * sd[v] for v in C.MATCH_VARS}
        cal[C.PRESPEED_VAR] = C.PRESPEED_CALIPER_SD * sd[C.PRESPEED_VAR]
        record("caliper", mult, "B", effect_at_t1(base, cal, "B"))

    # The registered six-variable caliper, reported as a deviation.
    #
    # Computed on the engine-evaluated subsample, where both specifications
    # exist, so the comparison is within one sample rather than across two.
    print("registered specification (win probability in the caliper)")
    cal_wp, _ = _calipers_from_main(with_wp=True)
    cal_5, _ = _calipers_from_main()
    sub = base.dropna(subset=["wp_before"])
    for tag, cal in [("5 vars (main)", cal_5), ("6 vars (registered)", cal_wp)]:
        st = effect_at_t1(sub, cal, "B")
        record("caliper_vars", tag, "B", st)
        if st:
            print(f"  {tag:<22} {st['effect']:+.4f} +/- {st['se']:.4f}  "
                  f"{st['n_players']:,} players", flush=True)

    C.ensure_dirs(C.DERIVED)
    res = pd.DataFrame(rows)
    path = os.path.join(C.DERIVED, "robustness.csv")
    res.to_csv(path, index=False)
    print(f"\n→ {path}")
    print("Complexity control and pre-event speed are read from "
          "run.py lag <B> <none|nlegal|nlegal_cx|zpre> and run.py did.")


def cmd_verify_paper(args):
    """
    Compare the values printed in the manuscript against data/derived/.

    The transcription lives in src/paper_check.py. When the manuscript
    changes, change it there too. A mismatch is labelled DATA (the repository
    needs regenerating) or PAPER (check what the manuscript says).
    """
    from src.paper_check import run as verify
    rep = verify()
    sys.exit(1 if rep.bad else 0)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="run.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<stage>")

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_, description=fn.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        s.set_defaults(fn=fn)
        return s

    s = add("scan", cmd_scan, "stage 1 - account metadata")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=C.MAIN_N_SHARDS)

    s = add("scan-titled", cmd_scan_titled,
            "stage 1, titled accounts (all six months if omitted)")
    s.add_argument("month", nargs="?", type=int, default=None,
                   choices=sorted(C.TITLED_MONTHS))
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=None)

    add("sample", cmd_sample, "fix the player sample -> out/ (not distributed)")

    s = add("extract", cmd_extract, "stage 2 - board replay (slowest stage)")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=C.MAIN_N_SHARDS)
    s.add_argument("--force", action="store_true",
                   help="proceed even when the sample differs from the paper")

    s = add("extract-titled", cmd_extract_titled, "stage 2, titled tier, 2024-01 .. 2024-06")
    s.add_argument("month", nargs="?", type=int, default=None)
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=None)

    s = add("prelim", cmd_prelim, "preliminary extraction (shards 0-23)")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=24)

    s = add("prep", cmd_prep, "analysis table (net_mat, z, z_pre3)")
    s.add_argument("fm", nargs="?", default=False,
                   type=lambda v: str(v).lower().startswith("fm"))

    s = add("se", cmd_se, "clustered SEs -> data/derived/se_*.csv")
    s.add_argument("--tiers", choices=["five", "four"], default="five",
                   help="five: include the titled tier (default); "
                        "four: lower four tiers only")
    s.add_argument("mode", choices=["split", "dose", "tier"])

    s = add("lag", cmd_lag, "one lag profile")
    s.add_argument("which", choices=["A", "B", "fmA", "fmB"])
    s.add_argument("spec", choices=list(SPECS))
    s.add_argument("--nmax", type=int, default=C.MAX_EVENTS_LAG)
    s.add_argument("--see", action="store_true",
                   help="define Event B by the withdrawn SEE rule "
                        "(appendix comparison only)")
    s.add_argument("--by-tier", action="store_true", dest="by_tier",
                   help="one run per tier (the five curves in Fig 1a/1b)")

    add("lag-all", cmd_lag_all, "every lag profile, then aggregate")

    s = add("lag-split", cmd_lag_split, "three-group lag decomposition")
    s.add_argument("which", choices=["lower", "fm"])
    s.add_argument("grp", choices=["Aonly", "Bonly", "AB"])
    s.add_argument("--no-prespeed", action="store_true", dest="no_prespeed",
                   help="drop z_pre3 from the caliper, reproducing the old "
                        "specification")

    for name, fn, help_ in [("ps6", cmd_ps6, "pre-speed matched, with pre-trend"),
                            ("did", cmd_did, "difference-in-differences")]:
        s = add(name, fn, help_)
        s.add_argument("which", nargs="?", default="B",
                       choices=["A", "B", "fmA", "fmB"])
        s.add_argument("start", nargs="?", type=int, default=0)
        s.add_argument("end", nargs="?", type=int, default=None)

    s = add("mixed", cmd_mixed, "preregistered mixed-effects model")
    s.add_argument("which", nargs="?", default="B",
                   choices=["A", "B", "fmA", "fmB"])
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_MIXED,
                   dest="max_events")

    s = add("cx-fens", cmd_cx_fens, "sample positions for complexity validation")
    s.add_argument("start", nargs="?", type=int, default=24)
    s.add_argument("end", nargs="?", type=int, default=34)

    s = add("cx", cmd_cx, "measure complexity with Stockfish")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=None)
    s.add_argument("--depth", type=int,
                   default=int(os.environ.get("CX_DEPTH", 12)))

    add("cx-model", cmd_cx_model, "complexity prediction model, adds cx_pred")
    add("aggregate", cmd_aggregate, "fold per-run output into data/derived/*.csv")

    s = add("anonymize", cmd_anonymize, "replace account names with salted hashes")
    s.add_argument("--dry-run", action="store_true", dest="dry_run")

    s = add("reliability", cmd_reliability, "per-player reliability (Table 2)")
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_SE,
                   dest="max_events")
    s.add_argument("--min-events", type=int, default=10, dest="min_events")
    s.add_argument("--select", type=int, default=None,
                   help="select players by event count before matching. "
                        "Defaults to config.RELIABILITY_SELECT "
                        "(material 237, blunder 40)")

    s = add("bins", cmd_bins, "effect within legal-move-count bins")
    s.add_argument("--size", type=int, default=None,
                   help="keep only losses of this size (9 = a queen); "
                        "all sizes when omitted")
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_SE,
                   dest="max_events")

    add("describe", cmd_describe, "event descriptives")

    s = add("robustness", cmd_robustness, "the six registered specifications")
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_SE,
                   dest="max_events")

    add("verify-paper", cmd_verify_paper, "manuscript values vs data/derived/")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
