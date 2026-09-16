# ───────────────────────────────────────────────────────────
# REVIEW NOTE
#   - Every threshold lives here. Check each against the Method section:
#       OPENING_CUT = 10            opening cut (plies)
#       TAU = 1                     measurement floor (seconds)
#       EPOCH_WINDOW = 3            post-event window, t+1 .. t+3
#       LOG_OFFSET = 1.0            ln(t + 1)
#       CALIPER_SD = 0.2            matching caliper
#       PRESPEED_CALIPER_SD = 0.6   pre-event speed caliper
#   - Names beginning SENS_ are the alternative levels used for robustness.
#   - Intermediate output goes to CHESS_WORK; default <repo>/out.
#   - Every filename the pipeline reads or writes is at the bottom of this
#     file. Do not write path strings anywhere else.
# ───────────────────────────────────────────────────────────

"""
Pipeline configuration — fixed parameters and paths.

Path conventions
    WORK        root for intermediate output (default <repo>/out,
                overridden by the CHESS_WORK environment variable)
    DERIVED     versioned output (<repo>/data/derived)
    STAGE1/2/3  per-stage intermediate directories
    PREPARED*   analysis tables, written by `run.py prep`
"""

import os
from pathlib import Path

# -- Fixed (settled before data collection; do not change) -------------
TIME_CONTROL   = "600+0"     # 10+0 rapid only
RATED_ONLY     = True        # Event LIKE 'Rated%'
EXCLUDE_BOT    = True        # exclude accounts tagged BOT
MIN_GAMES      = 30          # minimum games, same in every tier
OPENING_CUT    = 10          # opening cut, in plies: first 10 excluded
TAU            = 1           # measurement floor: move_time >= 1 s
EPOCH_WINDOW   = 3           # post-event window, t+1 .. t+3
LOG_OFFSET     = 1.0         # ln(t + 1)

# -- Matching ----------------------------------------------------------
CALIPER_SD          = 0.2    # Austin (2011): 0.2 SD of each variable
PRESPEED_CALIPER_SD = 0.6    # z_pre3 only; not preregistered
                             # (docs/definitions.md)
# Matching variables.
#
# The registered specification matched on win probability as well. Material
# loss is defined without reference to the engine, so requiring an evaluation
# confines those events to the games a user submitted for analysis -- 12.7%
# of games in the lowest tier against 93.9% in the titled tier, a subset that
# is neither random nor comparable across tiers.
#
# On the engine-evaluated subsample, where both specifications can be
# computed, matching on win probability moves the estimate from -0.552 to
# -0.575: less than either standard error. It costs most of the sample and
# buys almost nothing, so the five-variable set is the main specification and
# the registered one is reported under robustness.
#
# Five variables: player (matched exactly inside lagwise), the three below,
# and z_pre3.
MATCH_VARS          = ["clk_before", "n_legal", "ply"]
MATCH_VARS_WP       = ["wp_before"] + MATCH_VARS   # registered; robustness
PRESPEED_VAR        = "z_pre3"

# -- Subsampling caps and seeds ----------------------------------------
# Analyses with more events than are affordable to process in full are
# subsampled. Different caps give different precision, so standard errors
# from two analyses cannot be placed side by side unless the caps match.
#
# There must be exactly one seed. Previously `se` used 1 and `lag` used 0,
# so the two analyses were not working on the same set of events.
EVENT_SAMPLE_SEED = 0        # event subsampling (all analyses)
MATCH_SEED        = 0        # inside matching (all analyses)
MAX_EVENTS_SE     = 16000    # run.py se
MAX_EVENTS_LAG    = 25000    # run.py lag (ten lags, so a larger cap)
MAX_EVENTS_MIXED  = 20000    # run.py mixed
MIN_EVENTS        = 300      # below this, no estimate is produced

# Player selection for the reliability table (Table 2).
#
# A player enters only with at least this many events BEFORE matching. The
# manuscript states the rule; the code did not apply it, so every player was
# included -- most with one or two events -- and the split-half correlation
# collapsed. Matching then keeps about 63% of the selected events, which is
# why 237 becomes a median of ~149 per player in the published table.
RELIABILITY_SELECT = {"B": 237, "A": 40}
PLAYERS_PER_TIER_LAGSPLIT = 110   # player cap per tier in run.py lag-split

# -- Event definitions -------------------------------------------------
BLUNDER_THRESH = 0.10        # win-probability drop (proportion)
WP_LO, WP_HI   = 0.10, 0.90  # win-probability band before an Event A
PRESPEED_K     = 3           # z_pre3: how many preceding own-moves

# Sampling window
MAIN_YEAR       = 2024
MAIN_MONTH      = 1
MAIN_N_SHARDS   = 421
PRELIM_SHARDS   = set(range(0, 24))   # preliminary only; excluded from
                                      # the main analysis
# The titled tier spans six months: one month yields only 26 accounts with
# 30 or more games.
TITLED_MONTHS   = {1: 421, 2: 388, 3: 405, 4: 386, 5: 400, 6: 379}

# Tier boundaries (Lichess rapid rating). Only the top tier is defined by
# FIDE title rather than by rating.
TIERS = [
    ("1300-1600", 1300, 1600),
    ("1600-1900", 1600, 1900),
    ("1900-2100", 1900, 2100),
    ("2100+",     2100, 4000),
]
TITLE_TIER   = "FM+"
TITLES_KEEP  = {"FM", "IM", "GM", "WFM", "WIM", "WGM"}   # CM and NM excluded
TITLES_DROP  = {"CM", "NM", "BOT", "LM"}
PER_TIER     = 500           # players sampled from each untitled tier

# -- Grid search (settled empirically before preregistration; recorded) --
GRID_BLUNDER_THRESH = [0.10, 0.20, 0.30]   # win-probability drop
GRID_POST_BAND      = [None, (0.15, 0.85), (0.10, 0.90)]
GRID_TIME_FLOOR     = [None, 10, 20]       # floor on remaining clock (s)

# -- Sensitivity analyses (specified in the preregistration) ------------
SENS_OPENING_CUT = [10, 14, 20]
SENS_TAU         = [1, 2, 3]
SENS_CALIPER     = [0.1, 0.2, 0.4, 0.8]

# -- Raw archives ------------------------------------------------------
_HF = ("https://huggingface.co/datasets/Lichess/standard-chess-games/"
       "resolve/main/data")
ARCHIVE_URL = _HF + "/year%3D{y}/month%3D{m:02d}/train-{i:05d}-of-{n:05d}.parquet"
ARCHIVE_2024_URL = _HF + "/year%3D2024/month%3D{m:02d}/train-{i:05d}-of-{n:05d}.parquet"

# -- External tools ----------------------------------------------------
# Used only for the complexity validation. The bare name is enough if it is
# on PATH.
STOCKFISH = os.environ.get("CHESS_STOCKFISH", "stockfish")

# -- Paths -------------------------------------------------------------
_ROOT   = Path(__file__).resolve().parent.parent
WORK    = str(Path(os.environ.get("CHESS_WORK", _ROOT / "out")))
DERIVED = str(_ROOT / "data" / "derived")

STAGE1     = f"{WORK}/players"        # player-level aggregates, untitled
STAGE1_FM  = f"{WORK}/titled"         # player-level aggregates, titled
STAGE2     = f"{WORK}/plies"          # ply-level intermediate, untitled
STAGE2_FM  = f"{WORK}/plies_fm"       # ply-level intermediate, titled
STAGE3     = f"{WORK}/epochs"         # event and control epochs
PRELIM     = f"{WORK}/prelim"         # preliminary analysis

# The player sample is **not** versioned.
#
# Lichess account names are quasi-identifiers, and the titled tier (136
# players) is small enough to be matched against public profiles. What
# reproduction needs is the sampling rule, not the list, and that rule is
# entirely in src/extract.py and this file. See docs/definitions.md.
#
# These therefore live under WORK, which .gitignore excludes.
SAMPLE     = f"{WORK}/sample.parquet"
SAMPLE_FM  = f"{WORK}/sample_fm.parquet"
PLAYERS_ALL = f"{WORK}/players_all.parquet"

# Analysis tables, written by run.py prep
PREPARED    = f"{WORK}/prepared.parquet"
PREPARED_FM = f"{WORK}/prepared_fm.parquet"

# Complexity validation
CX_FENS  = f"{WORK}/complexity_fens.parquet"
CX_MODEL = f"{WORK}/cx_model.json"


def prepared_path(is_fm: bool) -> str:
    return PREPARED_FM if is_fm else PREPARED


def ensure_dirs(*paths):
    """Create the directory for each path; for a file path, its parent."""
    for p in paths:
        d = p if (os.path.splitext(p)[1] == "") else os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
