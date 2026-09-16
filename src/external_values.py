"""
Reference values the pipeline does not produce.

Some values the manuscript reports do not come from this pipeline. They are
collected here so that `run.py verify-paper` can compare them against the
transcription in `src/paper_check.py`.

Each entry carries a status:
    "external"  produced outside this repository. The flanker comparison
                (OpenNeuro ds004883) is the only case; the code for that
                analysis is not here.
    "pending"   would be replaced by a CSV once the run.py stage named in the
                comment has been run. Every panel the manuscript reports now
                has a generating stage, and `data/derived/` takes precedence;
                these remain only as a fallback.

History of this file
    It was `figures/figure_data.py`. Despite the name it was not figure code
    but a table of values: at the time the plotting code had numbers written
    directly into it, so there was no way to check a figure against the
    output it supposedly came from. Fig 1c once carried error bars 1.6 to 1.8
    times larger than the standard errors in se_split.csv without anything
    catching it. The plotting code has since been removed from the
    repository; this table is still used for verification, so it moved to
    src/.
"""

TIER_NAMES = ["1300–1600", "1600–1900", "1900–2100", "2100+"]

# tier values in data/derived/lag_profiles.csv <-> figure legend names
TIER_ORDER = ["1300-1600", "1600-1900", "1900-2100", "2100+"]
TIER_DISPLAY = dict(zip(TIER_ORDER, TIER_NAMES))

# ★ t+1 disagrees with the manuscript text.
#   Manuscript: "Values at t+1 ranged from -0.641 to -0.548 across tiers"
#   Here:       -0.617 to -0.564
#   The t+2, t+3 and blunder rows match the manuscript exactly, so the t+1
#   column appears to come from a different run. run.py verify-paper
#   reports this.
MATERIAL_BY_TIER = {
    "status": "pending",
    "lags": [-3, -2, -1, 0, 1, 2, 3],
    # The pre-event range (t-3 to t-1) is -0.026 to +0.023. The
    # manuscript's "-0.057 to +0.048" matches the blunder_only t-1 and t-3
    # values in se_split.csv exactly, so figures from the three-group panel
    # (Fig 1c) appear to have been transcribed into the tier-wise paragraph.
    # Needs checking against the manuscript.
    "rows": [

        [-0.011, -0.001, 0.014, 0.260, -0.564, -0.082, -0.107],

        [0.004, -0.008, 0.000, 0.233, -0.599, -0.082, -0.079],

        [-0.026, -0.001, 0.020, 0.224, -0.617, -0.057, -0.083],

        [-0.008, -0.002, 0.023, 0.229, -0.598, -0.045, -0.038],
    ],
}

BLUNDER_BY_TIER = {
    "status": "pending",
    "lags": [-3, -2, -1, 0, 1, 2, 3],
    "rows": [

        [0.032, 0.005, -0.004, 0.318, -0.029, -0.001, -0.049],

        [0.056, 0.033, -0.040, 0.326, -0.035, 0.004, -0.011],

        [0.019, 0.039, -0.057, 0.335, -0.030, -0.019, -0.016],

        [0.014, 0.026, -0.028, 0.334, -0.007, -0.013, -0.047],
    ],
}

# ─────────────────────────────────────────────────────────────
# Fig 2 (b) - move-time change by bin of change in legal move count
# status: pending
# Superseded by: python run.py bins        -> legal_bins.csv
# ─────────────────────────────────────────────────────────────
LEGAL_MOVE_BINS = {
    "status": "pending",
    "x": [0, 1, 2, 3, 4, 5, 6, 7],
    "xticklabels": ["≤−12", "−8~−4", "−1~+1", "+4~+8"],
    "self_lost": [-1.109, -1.059, -0.911, -0.736, -0.663, -0.592, -0.513, 0.030],
    "self_se":   [0.048, 0.052, 0.039, 0.031, 0.034, 0.038, 0.049, 0.041],
    "opp_lost":  [-0.082, 0.368, 0.535, 0.662, 0.686, 0.728, 0.802, 0.875],
    "opp_se":    [0.061, 0.057, 0.043, 0.033, 0.033, 0.035, 0.039, 0.043],
}

# ─────────────────────────────────────────────────────────────
# Fig 2 (c) - effect by size of loss (own loss / opponent's loss)
# status: pending
# Superseded by: python run.py se dose     -> se_dose.csv (already read)
#       The labels are self_N / opp_N (formerly dose_N / gain_N).
# ─────────────────────────────────────────────────────────────
DOSE = {
    "status": "pending",
    "x": [1, 3, 5, 9],
    # Five-variable matching. se_dose.csv is the source; these are the
    # fallback for when it is absent.
    "self": [-0.3005, -0.6972, -0.8053, -0.9786],
    "self_se": [0.0147, 0.0147, 0.0156, 0.0161],
    "opp": [0.2190, 0.1591, 0.1628, 0.0269],
    "opp_se": [0.0141, 0.0144, 0.0163, 0.0158],
}

# ─────────────────────────────────────────────────────────────
# Fig 2 (d)(e) - reliability
# status: pending (chess) / external (flanker)
# Superseded by: python run.py reliability
#                -> reliability.csv, reliability_curve.csv
# ─────────────────────────────────────────────────────────────
RELIABILITY_CHESS = {
    "status": "pending",
    "n": [20, 40, 60, 100, 150, 237, 300],
    "sb": [0.378, 0.531, 0.603, 0.596, 0.685, 0.741, 0.754],
}
RELIABILITY_FLANKER = {
    "status": "external",
    "n": [10, 20, 30, 50, 80],
    "sb": [0.233, 0.095, 0.358, 0.544, 0.676],
}
RELIABILITY_BY_CONDITION = {
    "status": "pending",
    # Manuscript v8, Table 2 - two rows, tiers pooled.
    "labels": ["Blund.\nuntitled", "Mat.\nuntitled"],
    "sb": [-0.113, 0.781],
    "sigma_b": [0.284, 0.218],
}

# ─────────────────────────────────────────────────────────────
# Flanker comparison (OpenNeuro ds004883)
# status: external
# The flanker analysis code is not in this repository. These values come
# from a separate script, so they are held as status="external" and
# verify-paper does not compare them against anything produced here.
# Citing them in the paper requires naming the source (OpenNeuro ds004883).
# ─────────────────────────────────────────────────────────────
FLANKER = {
    "status": "external",
    "post_error_slowing": 0.131,
    "post_error_slowing_se": 0.020,
    "n_correct": 81108,
    "n_error": 10461,
    "err_rate_after_correct": 10.2,
    "err_rate_after_error": 21.4,
}
