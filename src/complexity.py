"""
Validation of the position-complexity proxy

Legal move count (n_legal) is used as a proxy for position complexity among
the matching variables. No established standard for complexity exists in the
literature, which made this unavoidable, but it does not capture tactical
sharpness.

This module measures complexity directly with Stockfish multi-PV — the
standard deviation of the evaluations of the top N moves — and correlates it
with legal move count. Wide spread means the choice of move matters a great
deal (a complex position); narrow spread means it does not.

Prediction model
    Stockfish cannot be run over millions of positions, so complexity measured
    on a sample of positions is regressed on features the extraction stage
    already records (n_legal, max_see_mine, mat_diff, n_checks) and extended
    to the whole corpus. That prediction is cx_pred, used as an additional
    matching variable by the `nlegal_cx` specification.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE
#   - Validation only; not part of the main analysis path.
#   - The engine path is set by CHESS_STOCKFISH (default: stockfish on PATH).
#   - ★ The prediction model was missing from the original repository. What is
#     implemented here is a reconstruction from the four features the
#     extraction stage happens to record, so confirm that it reproduces the
#     r = 0.47 reported in the manuscript — `run.py cx-model` prints it —
#     before citing that figure.
# ───────────────────────────────────────────────────────────

import json
import subprocess

import numpy as np
import pandas as pd

from .config import STOCKFISH

ENGINE = STOCKFISH
DEPTH = 18
MULTIPV = 5
MATE_CP = 10000

# Predictors for the model - columns already present in the stage 2 schema.
CX_FEATURES = ["n_legal", "max_see_mine", "mat_diff", "n_checks"]


class Engine:
    def __init__(self, path=None, threads=1, hash_mb=64):
        path = path or ENGINE
        try:
            self.p = subprocess.Popen(
                [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Stockfish not found ({path!r}). Install it and set "
                "CHESS_STOCKFISH to its path.") from exc
        self._cmd("uci")
        self._wait("uciok")
        self._cmd(f"setoption name Threads value {threads}")
        self._cmd(f"setoption name Hash value {hash_mb}")
        self._cmd(f"setoption name MultiPV value {MULTIPV}")
        self._cmd("isready")
        self._wait("readyok")

    def _cmd(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line or token in line:
                return line

    def analyse(self, fen, depth=DEPTH):
        """Evaluations of the top MULTIPV moves, in centipawns, from the
        mover's point of view."""
        self._cmd("ucinewgame")
        self._cmd(f"position fen {fen}")
        self._cmd(f"go depth {depth}")
        scores = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                break
            if line.startswith("bestmove"):
                break
            if " multipv " in line and " score " in line and f"depth {depth}" in line:
                t = line.split()
                try:
                    pv = int(t[t.index("multipv") + 1])
                    si = t.index("score")
                    if t[si + 1] == "cp":
                        v = int(t[si + 2])
                    else:
                        v = MATE_CP if int(t[si + 2]) > 0 else -MATE_CP
                    scores[pv] = v
                except (ValueError, IndexError):
                    pass
        return [scores[k] for k in sorted(scores)]

    def close(self):
        try:
            self._cmd("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def complexity(scores):
    """SD of the top-move evaluations. Larger means a more complex position."""
    if len(scores) < 2:
        return np.nan
    return float(np.std(scores))


def gap_top2(scores):
    """Gap between best and second-best move - an alternative complexity index."""
    if len(scores) < 2:
        return np.nan
    return float(scores[0] - scores[1])


# -- Complexity prediction model ---------------------------------

def fit_cx_model(df, target="sd", features=CX_FEATURES):
    """
    Regress measured complexity (target) on position features.

    df      : cx output joined to the corresponding stage 2 features
    Returns : {"features": [...], "coef": [...], "intercept": float,
               "r": in-sample correlation, "n": observations}

    The aim is a predicted value for matching, not explanation, so ordinary
    least squares is enough.
    """
    d = df.dropna(subset=[target] + list(features))
    if len(d) < 50:
        raise ValueError(f"sample too small (n={len(d)}).")
    X = d[list(features)].to_numpy(float)
    y = d[target].to_numpy(float)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ beta
    r = float(np.corrcoef(pred, y)[0, 1])
    return {"features": list(features), "intercept": float(beta[0]),
            "coef": [float(b) for b in beta[1:]], "r": r, "n": int(len(d))}


def predict_cx(d, model):
    """Apply the model to produce cx_pred. NaN where a feature is missing."""
    feats = model["features"]
    missing = [f for f in feats if f not in d.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    X = d[feats].to_numpy(float)
    out = model["intercept"] + X @ np.asarray(model["coef"], float)
    return pd.Series(out, index=d.index).where(d[feats].notna().all(axis=1))


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)


def load_model(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
