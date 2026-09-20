"""
Raw archives → player sample → ply-level features

Four stages in pipeline order:

  1. PGN movetext parsing and win-probability transform
     parse_movetext / move_times / winprob
  2. Stage 1   account metadata     scan_shard / scan_titled_shard
  3. Stage 1b  stratified sampling  aggregate_players / build_sample
  4. Stage 2   board replay and feature extraction   extract_shard  ← the core

Driver: `run.py`. There is no entry point in this file.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE ★ the reported numbers originate in this file
#
#   [parsing]
#   - Does the win-probability transform and its coefficient (0.00368208)
#     match the equation in the Method section?
#   - Perspective normalisation: the sign of the centipawn score is flipped
#     for Black. Omitting this loses every blunder by Black and keeps only
#     White's.
#   - Handling of mate scores (described in the manuscript as 100%).
#   - move_time = clk(i−2) − clk(i). Increment is zero, so the plain
#     difference is correct. A single missing clock removes two move times.
#
#   [sampling]
#   - Tier definitions: 500 players from each of the four untitled tiers,
#     all titled accounts that qualify.
#   - Where the minimum of 30 games is applied.
#   - That the seed is fixed (reproducibility).
#   - The titled tier is scanned across 2024-01 to 2024-06; one month yields
#     only 26 accounts with 30 or more games.
#
#   [extraction] ★★
#   - mat_diff: piece values (pawn 1, knight 3, bishop 3, rook 5, queen 9),
#     king excluded; confirm the sign is from the moving player's
#     perspective. All of Event B rests on this.
#   - mat_diff is filled only when do_see=True.
#   - Which point in the move n_legal, ply, wp_before and wp_delta refer to.
#   - The surest check is to follow one game by hand and count mat_diff.
#   - This is the slowest stage (about 85 minutes over 397 shards).
# ───────────────────────────────────────────────────────────

import hashlib
import os
import re
import subprocess
import time

import chess
import duckdb
import numpy as np
import pandas as pd

from .config import *          # noqa: F403  thresholds, paths, tier definitions
from .see import see, VALUES


# ════════════════════════════════════════════════════════════════════
# 1. Parsing PGN movetext and converting to win probability
# Lichess win-probability function:
#     winprob(cp) = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
#     -> 0-100, from White's point of view. Normalised to 0-1 here.
# 
# Point of view: when the player is Black the sign of cp is flipped before
# converting. Omitting this loses every blunder by Black and keeps only
# White's.
# ════════════════════════════════════════════════════════════════════

K = 0.00368208

CLK_RE  = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+)\]")
EVAL_RE = re.compile(r"\[%eval\s+(#?-?[\d.]+)\]")
# Capture each move together with the comment block that follows it.
MOVE_RE = re.compile(
    r"(?:\d+\.+\s*)?"                     # move number (optional)
    r"((?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?)"
    r"([?!]*)"                            # NAG: Lichess writes ?/??/!/!?
    r"\s*(\{[^}]*\})?"                    # comment block
)
NAG_MAP = {"?!": "inaccuracy", "?": "mistake", "??": "blunder"}

MATE_CP = 10000   # mate scores are clipped to this


def winprob(cp):
    """Centipawns (White's view) -> win probability 0-1 (White's view)."""
    return 1.0 / (1.0 + np.exp(-K * np.asarray(cp, dtype=float)))


def parse_eval(tok):
    """'#-3' or '0.25' -> centipawns as a float, from White's view."""
    if tok.startswith("#"):
        v = tok[1:]
        sign = -1.0 if v.startswith("-") else 1.0
        return sign * MATE_CP
    return float(tok) * 100.0


def parse_movetext(mt):
    """
    movetext → (clks, evals, sans, nags)
    All three arrays are indexed by ply. A missing value is None.
    clk is in whole seconds: Lichess records only H:MM:SS.
    """
    clks, evals, sans, nags = [], [], [], []
    for m in MOVE_RE.finditer(mt):
        san, nag, comment = m.group(1), m.group(2), m.group(3)
        sans.append(san)
        nags.append(NAG_MAP.get(nag))          # Lichess's own labels, for cross-checking
        c = e = None
        if comment:
            cm = CLK_RE.search(comment)
            if cm:
                h, mi, s = cm.groups()
                c = int(h) * 3600 + int(mi) * 60 + int(s)
            em = EVAL_RE.search(comment)
            if em:
                e = parse_eval(em.group(1))
        clks.append(c)
        evals.append(e)
    return clks, evals, sans, nags


def move_times(clks, color):
    """
    Move times for one player.
    color: 0 = White (even ply indices), 1 = Black (odd)
    time(i) = clk(i-2) - clk(i), two consecutive readings for the same player.
    The increment is zero, so no correction term is needed.
    A player's first move has no preceding reading and is undefined -> None.
    Returns {ply_index: seconds}
    """
    out = {}
    idxs = list(range(color, len(clks), 2))
    for j, i in enumerate(idxs):
        if j == 0:
            continue                       # first move: undefined
        prev = idxs[j - 1]
        if clks[i] is None or clks[prev] is None:
            continue
        d = clks[prev] - clks[i]
        if d < 0:
            continue                       # negative: recording error (0.011% observed)
        out[i] = d
    return out


def player_winprobs(evals, color):
    """
    Win probabilities from the player's point of view.
    evals are centipawns from White's view; the sign is flipped for Black
    before converting.
    Returns {ply_index: winprob (0-1)} for the position just after that move.
    """
    out = {}
    sign = 1.0 if color == 0 else -1.0
    for i, e in enumerate(evals):
        if e is None:
            continue
        out[i] = float(winprob(sign * e))
    return out

# ════════════════════════════════════════════════════════════════════
# 2. Stage 1 - account metadata, without reading movetext
# Assigns tiers and fixes the pool of candidates for sampling. Game content
# is not parsed. Each shard is downloaded, aggregated and deleted at once, so
# the stage can be restarted.
# Writes: {WORK}/players/sNNNNN.parquet, {WORK}/titled/mMM_sNNNNN.parquet
# ════════════════════════════════════════════════════════════════════

def shard_url(year, month, i, n):
    return ARCHIVE_URL.format(y=year, m=month, i=i, n=n)


def parquet_ok(path):
    """
    A complete parquet file ends with the four bytes PAR1. A download that was
    cut off does not, whatever its size.

    Size was the test before, `> 1_000_000`. Seven truncated shards of 5 MB to
    168 MB passed it, were kept in the cache as if finished, and brought down
    `extract` hours later with "No magic bytes found at end of file".
    """
    try:
        if os.path.getsize(path) < 12:
            return False
        with open(path, "rb") as fh:
            fh.seek(-4, os.SEEK_END)
            return fh.read(4) == b"PAR1"
    except OSError:
        return False


DOWNLOAD_LOG = os.environ.get("CHESS_DOWNLOAD_LOG")   # a path, or unset to stay quiet


def _note(msg):
    if DOWNLOAD_LOG:
        try:
            with open(DOWNLOAD_LOG, "a", encoding="utf-8") as fh:
                fh.write("%s  %s" % (time.strftime("%H:%M:%S"), msg) + chr(10))
        except OSError:
            pass


def download(url, dst, tries=4):
    """
    Download to `dst`, atomically and verified.

    The old version wrote straight to `dst`, checked only the return code and
    the file size, and said nothing about why it failed. A scan of 397 shards
    reported `fail=273` with no record of a single reason, and left seven
    half-written files in the cache that later passed the size test.

    Now: write to `dst.part`, require curl to succeed *and* the footer to be
    intact, then rename. A partial file is removed, never promoted. Each
    attempt's curl error is recorded.
    """
    part = dst + ".part"
    for attempt in range(1, tries + 1):
        # No -C - : curl combines resume badly with --retry and returns 23
        # ("failed writing body") with nothing written. Each attempt starts
        # the file again; a shard is ~170 MB and takes about a minute.
        if os.path.exists(part):
            os.remove(part)
        r = subprocess.run(
            ["curl", "-sS", "-L", "--fail", "--connect-timeout", "30",
             "--max-time", "1800", "--speed-limit", "10000", "--speed-time", "120",
             "--retry", "3", "--retry-delay", "5", "--retry-all-errors",
             "-o", part, url],
            capture_output=True, text=True)
        if r.returncode == 0 and parquet_ok(part):
            os.replace(part, dst)
            return True
        why = (r.stderr or "").strip().splitlines()
        why = why[-1][:120] if why else (
            "footer missing" if r.returncode == 0 else "curl rc=%d" % r.returncode)
        size = os.path.getsize(part) if os.path.exists(part) else 0
        _note("attempt %d/%d  %-11s %12s bytes  %s"
              % (attempt, tries, os.path.basename(dst), f"{size:,}", why))
        if os.path.exists(part):
            os.remove(part)
        if attempt < tries:
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    return False


SQL_PLAYERS = f"""
WITH g AS (
  SELECT White AS player, WhiteTitle AS title, WhiteElo AS elo,
         (movetext LIKE '%eval%') AS has_eval
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}'
    AND Event LIKE 'Rated%'
    AND (WhiteTitle IS NULL OR WhiteTitle <> 'BOT')
    AND (BlackTitle IS NULL OR BlackTitle <> 'BOT')
  UNION ALL
  SELECT Black, BlackTitle, BlackElo, (movetext LIKE '%eval%')
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}'
    AND Event LIKE 'Rated%'
    AND (WhiteTitle IS NULL OR WhiteTitle <> 'BOT')
    AND (BlackTitle IS NULL OR BlackTitle <> 'BOT')
)
SELECT player,
       any_value(title)            AS title,
       count(*)                    AS n_games,
       sum(has_eval::INT)          AS n_eval_games,
       median(elo)                 AS elo_median,
       min(elo) AS elo_min, max(elo) AS elo_max
FROM g
WHERE elo IS NOT NULL
GROUP BY player
"""


def scan_shard(path, out_path):
    con = duckdb.connect()
    con.execute(f"COPY ({SQL_PLAYERS.format(f=path)}) TO '{out_path}' (FORMAT PARQUET)")
    con.close()




STAGE1_FM = STAGE1_FM

TITLED_MONTHS = TITLED_MONTHS          # {1: 421, 2: 388, ... 6: 379}

SQL_TITLED = f"""
WITH g AS (
  SELECT White AS player, WhiteTitle AS title, WhiteElo AS elo,
         (movetext LIKE '%eval%') AS has_eval
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}' AND Event LIKE 'Rated%'
    AND WhiteTitle IN ('FM','IM','GM','WFM','WIM','WGM')
    AND (BlackTitle IS NULL OR BlackTitle <> 'BOT')
  UNION ALL
  SELECT Black, BlackTitle, BlackElo, (movetext LIKE '%eval%')
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}' AND Event LIKE 'Rated%'
    AND BlackTitle IN ('FM','IM','GM','WFM','WIM','WGM')
    AND (WhiteTitle IS NULL OR WhiteTitle <> 'BOT')
)
SELECT player, any_value(title) AS title, count(*) AS n_games,
       sum(has_eval::INT) AS n_eval_games,
       median(elo) AS elo_median, min(elo) AS elo_min, max(elo) AS elo_max
FROM g WHERE elo IS NOT NULL GROUP BY player
"""


def scan_titled_shard(args):
    """
    Aggregate one shard's titled accounts.

    With keep=True the downloaded shard is left in SHARD_CACHE instead of
    being deleted, and extract-titled reads it from there. The two stages
    otherwise fetch the same 2,355 shards twice over: about seven and a half
    hours for the titled tier, half of it spent downloading bytes that were
    already on disk. Keeping them costs roughly 400 GB, which is why it is
    not the default.
    """
    m, i, n = args[0], args[1], args[2]
    keep = args[3] if len(args) > 3 else False
    p = f"{STAGE1_FM}/m{m:02d}_s{i:05d}.parquet"
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return "skip"
    cached = shard_cache_path(m, i)
    have_cached = parquet_ok(cached)
    tmp = cached if keep else os.path.join(WORK, f"_t{m}_{i}.parquet")
    if keep:
        os.makedirs(SHARD_CACHE, exist_ok=True)
    if not have_cached:
        url = ARCHIVE_2024_URL.format(m=m, i=i, n=n)
        if not download(url, tmp):
            return "fail"
    else:
        tmp = cached
    try:
        con = duckdb.connect()
        con.execute(f"COPY ({SQL_TITLED.format(f=tmp)}) TO '{p}' (FORMAT PARQUET)")
        con.close()
    finally:
        if not keep and os.path.exists(tmp):
            os.remove(tmp)
    return "ok"

# ════════════════════════════════════════════════════════════════════
# 3. Stage 1b - aggregating players and drawing the stratified sample
# Sampling rule (fixed in Phase 1):
#   - at least 30 games, the same in every tier
#   - tier assigned by the player's median rating over the period
#   - 500 drawn at random from each of the lower four tiers; FM+ taken whole
#   - below 1300 excluded, bots excluded (already handled in Stage 1)
# ════════════════════════════════════════════════════════════════════

SEED = 20260816

_SQL_AGG = """
    SELECT player,
           max(title)                       AS title,
           sum(n_games)                     AS n_games,
           sum(n_eval_games)                AS n_eval_games,
           -- approximates the period median by weighting the per-shard medians
           sum(elo_median * n_games) / sum(n_games) AS elo_median,
           min(elo_min) AS elo_min, max(elo_max) AS elo_max
    FROM read_parquet('{glob}')
    GROUP BY player
    ORDER BY player          -- the sample draw depends on row order;
                             -- DuckDB does not promise one without this
"""


def aggregate_players(players_glob=None):
    """Sum the per-shard aggregates to one row per player."""
    players_glob = players_glob or f"{STAGE1}/*.parquet"
    con = duckdb.connect()
    df = con.execute(_SQL_AGG.format(glob=players_glob)).df()
    con.close()
    return df


def aggregate_titled(players_glob=None):
    """Aggregate the titled-only output of scan-titled."""
    players_glob = players_glob or f"{STAGE1_FM}/*.parquet"
    return aggregate_players(players_glob)


def assign_tier_row(row):
    t = row["title"]
    if t in TITLES_DROP:
        return None
    if t in TITLES_KEEP:
        return TITLE_TIER
    e = row["elo_median"]
    if pd.isna(e) or e < 1300:
        return None
    for name, lo, hi in TIERS:
        if lo <= e < hi:
            return name
    return None


def build_sample(df, min_games=MIN_GAMES, per_tier=PER_TIER, seed=SEED):
    """
    Draw the stratified sample.

    Sorted by player first. pool.sample(n, random_state=seed) picks by
    position, so a fixed seed over differently ordered rows returns
    different accounts. The aggregation feeding this is a DuckDB GROUP BY,
    which promises no order, and it had none: the published sample was
    drawn from an order nobody recorded and could not be reproduced even
    from the same archive. Sorting here rather than only in the SQL means
    every caller gets the same answer.
    """
    df = df.sort_values("player").reset_index(drop=True)
    df["tier"] = df.apply(assign_tier_row, axis=1)
    df = df[df.tier.notna()]

    elig = df[df.n_games >= min_games]
    out = []
    for name, _, _ in TIERS:
        pool = elig[elig.tier == name]
        n = min(per_tier, len(pool))
        out.append(pool.sample(n=n, random_state=seed))
    tpool = elig[elig.tier == TITLE_TIER]
    out.append(tpool)                     # the titled tier is taken whole
    return df, elig, pd.concat(out, ignore_index=True)


def build_titled_sample(df_titled, min_games=MIN_GAMES):
    """The titled tier, taken whole. Membership follows the title, not
    rating."""
    d = df_titled.copy()
    d = d[d.title.isin(TITLES_KEEP)]
    d = d[d.n_games >= min_games]
    d["tier"] = TITLE_TIER
    return d.reset_index(drop=True)


def sample_report(df, elig, samp):
    lines = []
    lines.append(f"players with a tier assigned: {len(df):,}")
    lines.append(f"passing >={MIN_GAMES} games:      {len(elig):,}  ({100*len(elig)/len(df):.1f}%)")
    lines.append("")
    lines.append(f"{'tier':>12} {'total':>9} {'>=30':>8} {'pass%':>7} {'drawn':>7} {'eval>=30':>8}")
    lines.append("-" * 58)
    for name in [t[0] for t in TIERS] + [TITLE_TIER]:
        a = df[df.tier == name]
        e = elig[elig.tier == name]
        s = samp[samp.tier == name]
        ev = a[a.n_eval_games >= MIN_GAMES]
        rate = 100 * len(e) / len(a) if len(a) else 0
        lines.append(f"{name:>12} {len(a):>9,} {len(e):>8,} {rate:>6.1f}% {len(s):>7,} {len(ev):>8,}")
    lines.append("-" * 58)
    lines.append(f"{'total':>12} {len(df):>9,} {len(elig):>8,} {'':>7} {len(samp):>7,}")
    return "\n".join(lines)


def build_all():
    """
    Build the sample from whatever stage 1 output exists.

    The titled sample is derived from the titled aggregates alone and does not
    depend on the untitled ones. This used to call aggregate_players() first
    and die on a missing players/ directory, taking the titled sample with it,
    so anyone who had run only scan-titled could not go on to extract-titled.
    """
    ensure_dirs(SAMPLE, SAMPLE_FM, PLAYERS_ALL)
    has_untitled = os.path.isdir(STAGE1) and any(
        f.endswith(".parquet") for f in os.listdir(STAGE1))
    if has_untitled:
        df = aggregate_players()
        df, elig, samp = build_sample(df)
    else:
        print(f"note: {STAGE1} holds no aggregates, so only the titled sample "
              f"is built. Run `run.py scan` for the rest.")
        df = elig = samp = None

    # Titled tier: replace it with the separate aggregate when one exists.
    fm = None
    if os.path.isdir(STAGE1_FM) and any(
            f.endswith(".parquet") for f in os.listdir(STAGE1_FM)):
        fm = build_titled_sample(aggregate_titled())
        fm.to_parquet(SAMPLE_FM, index=False)
        if samp is not None:
            samp = pd.concat([samp[samp.tier != TITLE_TIER], fm],
                             ignore_index=True)
        print(f"titled tier: {len(fm):,} players -> {SAMPLE_FM}")
    else:
        print(f"warning: {STAGE1_FM} is empty. Run `run.py scan-titled` "
              f"first or the titled tier will be missing.")

    if samp is None:
        return
    print(sample_report(df, elig, samp))
    samp.to_parquet(SAMPLE, index=False)
    df.to_parquet(PLAYERS_ALL, index=False)
    print(f"\nsample of {len(samp):,} players -> {SAMPLE}")

# ════════════════════════════════════════════════════════════════════
# 4. Stage 2 - replaying games and computing per-ply features
# Columns produced:
#   game_id, player, color, tier, ply, move_time, clk_before, clk_after,
#   wp_before, wp_after, wp_delta,   <- Event A; games with an evaluation only
#   see_loss,                        <- the preregistered (withdrawn) definition
#   mat_diff,                        <- Event B: the material differential
#                                      immediately before the move, from the
#                                      mover's point of view
#   n_legal, max_see_mine, n_checks, <- complexity proxies and model inputs
#   nag, is_castle, is_capture, ply_total, termination
# ════════════════════════════════════════════════════════════════════

GAME_COLS = ["Site", "White", "Black", "WhiteTitle", "BlackTitle",
        "WhiteElo", "BlackElo", "Termination", "movetext"]

SCHEMA = ["game_id", "player", "color", "tier", "ply", "move_time",
          "clk_before", "clk_after", "wp_before", "wp_after", "wp_delta",
          "see_loss", "n_legal", "nag", "is_castle", "is_capture",
          "ply_total", "termination",
          "max_see_mine", "mat_diff", "n_checks"]


def assign_tier_from_game(elo, title):
    if title in TITLES_DROP:
        return None
    if title in TITLES_KEEP:
        return TITLE_TIER
    if elo is None or elo < 1300:
        return None
    for name, lo, hi in TIERS:
        if lo <= elo < hi:
            return name
    return None


def pick_side(site):
    h = int(hashlib.md5(str(site).encode()).hexdigest()[:8], 16)
    return h % 2


def max_opponent_capture_see(board):
    """
    The largest SEE available to the side to move (the opponent).
    A value > 0 means the player who has just moved lost material on net.

    SEE is computed once per target square. Several pieces may be able to
    capture on the same square, but SEE uses the cheapest attacker first
    internally, so the result is the same either way. Targets are examined in
    descending order of value so the search can stop early.
    """
    me = board.turn
    targets = []
    for sq in chess.scan_forward(board.occupied_co[not me]):
        p = board.piece_at(sq)
        if p is None or VALUES[p.piece_type] == 0:
            continue
        if board.attackers_mask(me, sq):
            targets.append((VALUES[p.piece_type], sq))
    if not targets:
        best = 0
    else:
        targets.sort(reverse=True)
        best = 0
        for v, sq in targets:
            if v <= best:
                break                   # no remaining target can beat the best so far
            for mv in board.generate_legal_moves(chess.BB_ALL, chess.BB_SQUARES[sq]):
                if not board.is_capture(mv):
                    continue            # skip non-captures such as promotions
                s = see(board, mv)
                if s > best:
                    best = s

    # En passant leaves the target square empty, so the search above misses it.
    if board.ep_square is not None:
        for mv in board.generate_legal_moves(chess.BB_ALL,
                                             chess.BB_SQUARES[board.ep_square]):
            if board.is_en_passant(mv):
                s = see(board, mv)
                if s > best:
                    best = s
    return best


def extract_game(site, white, black, wt, bt, we, be, term, mt,
                 keep_players=None, do_see=True, tier_map=None):
    color = pick_side(site)
    player = white if color == 0 else black
    if keep_players is not None and player not in keep_players:
        other = black if color == 0 else white
        if other in keep_players:
            color, player = 1 - color, other
        else:
            return None

    # Tier assignment uses the tier fixed at sampling time (the player's
    # median rating over the period). Assigning per game would put the same
    # player in several tiers as their rating moves, which contaminates the
    # between-tier comparison. Per-game assignment is used only when no
    # tier_map is available.
    if tier_map is not None:
        tier = tier_map.get(player)
    else:
        title = wt if color == 0 else bt
        elo = we if color == 0 else be
        tier = assign_tier_from_game(elo, title)
    if tier is None:
        return None

    clks, evals, sans, nags = parse_movetext(mt)
    if len(sans) < OPENING_CUT + 4:
        return None

    mtimes = move_times(clks, color)
    wps = player_winprobs(evals, color)
    if not mtimes:
        return None

    # Board tracking: SEE, legal move count, and the complexity predictors.
    see_by_ply, legal_by_ply, cap_by_ply = {}, {}, {}
    feat_by_ply = {}
    if do_see:
        board = chess.Board()
        for i, san in enumerate(sans):
            try:
                mv = board.parse_san(san)
            except Exception:
                break
            if i % 2 == color and i >= OPENING_CUT and i in mtimes:
                legal_by_ply[i] = board.legal_moves.count()
                cap_by_ply[i] = board.is_capture(mv)
                # Complexity predictors: robustness checks only, not preregistered.
                caps = [m for m in board.legal_moves if board.is_capture(m)]
                best_mine = 0
                for m in caps:
                    v = see(board, m)
                    if v > best_mine:
                        best_mine = v
                me = board.turn
                mat = {}
                for c in (True, False):
                    mat[c] = sum(VALUES[p] * len(board.pieces(p, c))
                                 for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                           chess.ROOK, chess.QUEEN))
                feat_by_ply[i] = (
                    best_mine,
                    mat[me] - mat[not me],
                    sum(1 for m in board.legal_moves if board.gives_check(m)),
                )
                board.push(mv)
                see_by_ply[i] = max_opponent_capture_see(board)
                board.pop()
            board.push(mv)

    rows = []
    for ply, sec in mtimes.items():
        if ply < OPENING_CUT:
            continue
        wp_a, wp_b = wps.get(ply), wps.get(ply - 1)
        clk_a = clks[ply]
        rows.append((
            str(site), player, color, tier, ply, sec,
            (clk_a + sec) if clk_a is not None else None, clk_a,
            wp_b, wp_a,
            (wp_a - wp_b) if (wp_a is not None and wp_b is not None) else None,
            see_by_ply.get(ply), legal_by_ply.get(ply),
            nags[ply], sans[ply].startswith("O-O"), cap_by_ply.get(ply),
            len(sans), term,
            *(feat_by_ply.get(ply) or (None, None, None)),
        ))
    return rows


def sample_fens(path, n=2000, seed=0, keep_players=None, tier_map=None):
    """
    Sample positions for the complexity validation.

    The plies tables do not store FENs (they would be too large), so the
    games are replayed here to recover them.
    Columns: game_id, player, tier, ply, fen, n_legal, max_see_mine,
             mat_diff, n_checks
    """
    rng = np.random.default_rng(seed)
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT {','.join(GAME_COLS)} FROM read_parquet('{path}') "
        f"WHERE TimeControl='{TIME_CONTROL}' AND Event LIKE 'Rated%' "
        f"AND (WhiteTitle IS NULL OR WhiteTitle<>'BOT') "
        f"AND (BlackTitle IS NULL OR BlackTitle<>'BOT')").fetchall()
    con.close()

    out = []
    for r in rows:
        site, white, black, wt, bt, we, be, term, mt = r
        color = pick_side(site)
        player = white if color == 0 else black
        if keep_players is not None and player not in keep_players:
            other = black if color == 0 else white
            if other not in keep_players:
                continue
            color, player = 1 - color, other
        tier = (tier_map or {}).get(player) or assign_tier_from_game(
            we if color == 0 else be, wt if color == 0 else bt)
        if tier is None:
            continue
        clks, evals, sans, nags = parse_movetext(mt)
        if len(sans) < OPENING_CUT + 4:
            continue
        mtimes = move_times(clks, color)
        board = chess.Board()
        for i, san in enumerate(sans):
            try:
                mv = board.parse_san(san)
            except Exception:
                break
            if i % 2 == color and i >= OPENING_CUT and i in mtimes                     and rng.random() < 0.02:
                caps = [m for m in board.legal_moves if board.is_capture(m)]
                best_mine = max((see(board, m) for m in caps), default=0)
                me = board.turn
                mat = {c: sum(VALUES[pt] * len(board.pieces(pt, c))
                              for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                         chess.ROOK, chess.QUEEN))
                       for c in (True, False)}
                out.append({
                    "game_id": str(site), "player": player, "tier": tier,
                    "ply": i, "fen": board.fen(),
                    "n_legal": board.legal_moves.count(),
                    "max_see_mine": best_mine,
                    "mat_diff": mat[me] - mat[not me],
                    "n_checks": sum(1 for m in board.legal_moves
                                    if board.gives_check(m)),
                })
            board.push(mv)
        if len(out) >= n:
            break
    return pd.DataFrame(out[:n])


def extract_shard(path, out_path, keep_players=None, eval_only=False, do_see=True,
              tier_map=None):
    con = duckdb.connect()
    q = (f"SELECT {','.join(GAME_COLS)} FROM read_parquet('{path}') "
         f"WHERE TimeControl='{TIME_CONTROL}' AND Event LIKE 'Rated%' "
         f"AND (WhiteTitle IS NULL OR WhiteTitle<>'BOT') "
         f"AND (BlackTitle IS NULL OR BlackTitle<>'BOT')")
    if eval_only:
        q += " AND movetext LIKE '%eval%'"
    rows = con.execute(q).fetchall()
    con.close()

    out = []
    for r in rows:
        g = extract_game(*r, keep_players=keep_players, do_see=do_see,
                         tier_map=tier_map)
        if g:
            out.extend(g)
    df = pd.DataFrame(out, columns=SCHEMA)
    df.to_parquet(out_path, index=False)
    return len(df), df["game_id"].nunique()
