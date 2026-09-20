"""
The flanker comparison, computed here rather than quoted.

The manuscript used to say "the same analysis applied to a flanker dataset"
while holding only the resulting numbers, with no code that produced them.
Those numbers could not be reproduced from the archive under any selection of
its three task versions, so they are replaced by this.

The dataset (OpenNeuro ds004883; Clayson et al., 2024, Psychophysiology 61(9),
e14607) exists to test whether three commonly used flanker versions are
equivalent, and reports that they are not. They differ behaviourally too --
error rates run from 9% to 18% and one version is 2.7 times longer than the
others -- so the three are analysed separately and never pooled.

What "the same analysis" can mean here is limited by the task. Chess matches on
remaining time, legal move count, ply and preceding pace; a flanker trial has
no clock and two response options. The design carries over, the covariates are
the ones the task affords.
"""
import io
import os
import re

import numpy as np
import pandas as pd

# Matching, mirroring the chess specification.
CALIPER_SD = 0.2          # trial position
PRESPEED_CALIPER_SD = 0.6  # standardised time of the three preceding trials
PRESPEED_K = 3
RT_LO, RT_HI = 0.15, 1.5


def segments(d):
    """
    Recording segments. The archive marks discontinuities in the recording,
    not experimental blocks, so a "block" here is what lies between two
    interruptions. Nothing is looked up across one.
    """
    return d.groupby(["participant", "task"]).after_gap.cumsum()


def prepare(trials):
    """
    Add the columns the matching needs.

    z is standardised within participant AND version: the versions differ by
    60 ms or more in mean response time, so standardising within participant
    alone would leave the version difference inside z.
    """
    d = trials.copy()
    d = d[d.rt.between(RT_LO, RT_HI)].copy()
    d["segment"] = segments(d)
    d = d[~d.after_gap].copy()          # first trial of a segment

    key = ["participant", "task"]
    g = d.groupby(key).rt
    d["z"] = (np.log(d.rt) - g.transform("mean")) / g.transform("std")

    # Position inside the segment, and the neighbours, by trial number.
    idx = d.set_index(["participant", "task", "segment", "trial"])
    for k in range(1, PRESPEED_K + 1):
        d[f"_z{k}"] = _shift(d, idx, "z", -k)
    d["z_pre3"] = d[[f"_z{k}" for k in range(1, PRESPEED_K + 1)]].mean(
        axis=1, skipna=False)
    d = d.drop(columns=[f"_z{k}" for k in range(1, PRESPEED_K + 1)])

    d["prev_correct"] = _shift(d, idx, "correct", -1)
    d["next_z"] = _shift(d, idx, "z", +1)
    d["next_correct"] = _shift(d, idx, "correct", +1)
    d["next_congruency"] = _shift(d, idx, "congruency", +1)
    return d


def _shift(d, idx, col, step):
    """
    The value `step` trials away, within the same participant, version and
    recording segment. Absent rather than reached across a gap.
    """
    want = pd.MultiIndex.from_arrays(
        [d.participant, d.task, d.segment, d.trial + step])
    return idx[col].reindex(want).to_numpy()


def events_and_controls(d):
    """
    The window definition used throughout: an event is an error whose
    preceding trial was correct, and a control is a correct trial whose
    preceding trial was also correct. A run of errors is excluded on both
    sides, so the estimand is the effect of an isolated error.

    "Preceding trial was correct" requires that trial to exist and to have
    survived the filters. A trial that drew no response is closer to an error
    than to clean play, so treating its successor as clean would defeat the
    definition.
    """
    usable = d.prev_correct.notna() & d.z_pre3.notna() & d.next_z.notna()
    ev = d[usable & (d.correct == 0) & (d.prev_correct == 1)]
    ct = d[usable & (d.correct == 1) & (d.prev_correct == 1)]
    return ev, ct


def match(ev, ct, seed=0):
    """
    For each event, the controls from the same participant and version with
    the same congruency on the event trial and on the trial being measured,
    a trial position within 0.2 SD and a preceding pace within 0.6 SD.

    The congruency of the *next* trial is matched exactly. Unlike the chess
    analysis, where matching on a post-event quantity selected controls that
    had suffered something similar, the flanker stimulus order is fixed before
    the session: the event cannot have changed it, so matching on it removes
    variance without selecting on the outcome.
    """
    cal_trial = CALIPER_SD * float(ct.trial.std())
    cal_pre = PRESPEED_CALIPER_SD * float(ct.z_pre3.std())

    rows = []
    keys = ["participant", "task", "congruency", "next_congruency"]
    pool = {k: g for k, g in ct.groupby(keys, observed=True)}
    for r in ev.itertuples():
        g = pool.get((r.participant, r.task, r.congruency, r.next_congruency))
        if g is None or not len(g):
            continue
        m = ((g.trial - r.trial).abs() <= cal_trial) & \
            ((g.z_pre3 - r.z_pre3).abs() <= cal_pre)
        c = g[m]
        if not len(c):
            continue
        rows.append({
            "participant": r.participant, "task": r.task,
            "segment": r.segment, "trial": r.trial,
            "n_ctrl": len(c),
            "ev_rt": r.next_z, "ct_rt": float(c.next_z.mean()),
            "ev_err": float(r.next_correct == 0),
            "ct_err": float((c.next_correct == 0).mean()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["d_rt"] = out.ev_rt - out.ct_rt
        out["d_err"] = out.ev_err - out.ct_err
    return out


BLOCK = 40


def reliability(m, counts=(10, 20, 30, 40, 50, 60, 80), min_participants=25,
                seed=0, draws=50, block=BLOCK):
    """
    Split-half reliability of the per-person effect, against the number of
    error trials retained.

    The halves are blocks of `block` consecutive analysed trials, split on the
    parity of the block number -- the counterpart of splitting chess events by
    game. Two errors inside one block resemble each other more than two from
    different blocks, so splitting trial by trial would inflate the estimate.

    Blocks rather than recording segments: only ffa has more than one segment
    per participant (171 of 172; ffb and ffc have none), and with two segments
    a split is the first half against the second, which confounds the estimate
    with fatigue and practice. 40 trials was fixed before any of this was run:
    it leaves about eight blocks even in the shortest version, holding four to
    seven errors each, close to the 4.19 events of an average chess game.
    Sensitivity to that choice is reported at 20 and 80.

    Each point is the median of `draws` subsamples with its 2.5-97.5
    percentiles, for the reason given in analysis.reliability_curve: one draw
    is noisy enough to invent a dip.

    n_participants is the number whose correlation was computed, not the
    number left after truncation.
    """
    rng = np.random.default_rng(seed)
    m = m.assign(block=(m.trial // block).astype(int))
    out = []
    for n in counts:
        cnt = m.groupby("participant").size()
        sub = m[m.participant.isin(cnt[cnt >= n].index)]
        avail = int(sub.participant.nunique())
        sbs, rs, used = [], [], []
        for k in range(draws):
            if avail < min_participants:
                break
            g = sub.groupby("participant", group_keys=False)[sub.columns]
            cut = g.apply(lambda x: x.iloc[
                np.random.default_rng(seed + k).permutation(len(x))[:n]])
            piv = cut.assign(h=cut.block % 2).pivot_table(
                index="participant", columns="h", values="d_rt", aggfunc="mean")
            piv = piv.dropna()
            if len(piv) < min_participants or piv.shape[1] < 2:
                continue
            r = float(np.corrcoef(piv.iloc[:, 0], piv.iloc[:, 1])[0, 1])
            rs.append(r)
            sbs.append(2 * r / (1 + r) if r > -1 else np.nan)
            used.append(len(piv))
        if not sbs:
            out.append({"n_errors_per_participant": n, "r": np.nan,
                        "sb": np.nan, "sb_lo": np.nan, "sb_hi": np.nan,
                        "n_participants": avail, "n_participants_used": np.nan,
                        "draws": 0, "block": block,
                        "note": "too few participants with both halves"})
            continue
        out.append({"n_errors_per_participant": n,
                    "r": float(np.median(rs)), "sb": float(np.median(sbs)),
                    "sb_lo": float(np.percentile(sbs, 2.5)),
                    "sb_hi": float(np.percentile(sbs, 97.5)),
                    "n_participants": avail,
                    "n_participants_used": float(np.median(used)),
                    "draws": len(sbs), "block": block, "note": ""})
    return pd.DataFrame(out)


# ────────────────────────────────────────────────────────────────────
# Exclusion accounting
# ────────────────────────────────────────────────────────────────────

_PAT = re.compile(
    r"sub-(\w+)__ses-(\d+)__eeg__sub-\w+_ses-\d+_task-(\w+)_events\.tsv$")


def audit(root):
    """
    Per-version stimulus and exclusion counts, and the task's own error rate.

    The two exclusion rules overlap: ten trials are both outside the response
    time range and first in a recording segment. Subtracting the two counts
    separately removes those twice, which is how the analysed total came to be
    reported as 273,465 instead of 273,475, and later how the per-version
    column in the appendix came to be short by two and eight.

    So `analysed` adds the overlap back, and the caller asserts that the rows
    sum to the total rather than reading it off the table.

    `error_rate_pct` is the share of analysed trials that were errors. It is
    not `ev_err_rate` in flanker.csv, which is the error rate on the trial
    *after* a matched error -- a different quantity that happens to sit in a
    neighbouring column.
    """
    raw = os.path.join(root, "raw")
    rows = []
    for fn in sorted(os.listdir(raw)):
        m = _PAT.match(fn)
        if not m:
            continue
        sub, ses, task = m.groups()
        n_stim = n_resp = 0
        with io.open(os.path.join(raw, fn), encoding="utf-8",
                     errors="replace") as fh:
            for line in fh.read().splitlines()[1:]:
                p = line.split("\t")
                if len(p) < 4:
                    continue
                k = p[3].strip()
                n_stim += k in ("con", "inc")
                n_resp += k in ("cor", "err")
        rows.append((sub, ses, task, n_stim, n_resp))
    st = pd.DataFrame(rows, columns=["participant", "session", "task",
                                     "stimuli", "responses"])
    ex = pd.read_csv(os.path.join(root, "exclusions.csv"),
                     dtype={"participant": str, "session": str})
    d = pd.read_csv(os.path.join(root, "trials_v2.csv"),
                    dtype={"participant": str})

    g = ex.groupby("task").sum(numeric_only=True)
    g["stimuli"] = st.groupby("task").stimuli.sum()
    g["responses"] = st.groupby("task").responses.sum()
    bad = ~d.rt.between(RT_LO, RT_HI)
    g["rt_out"] = d.assign(b=bad).groupby("task").b.sum()
    g["both_rules"] = d.assign(b=bad).groupby("task").apply(
        lambda x: int((x.b & x.after_gap).sum()), include_groups=False)
    g["analysed"] = g.kept - (g.first_after_gap + g.rt_out - g.both_rules)

    keep = d.rt.between(RT_LO, RT_HI) & ~d.after_gap
    g["errors"] = d[keep].groupby("task").apply(
        lambda x: int((x.correct == 0).sum()), include_groups=False)
    g["error_rate_pct"] = (100 * g.errors / g.analysed).round(2)
    g.index.name = "task"
    return g[["stimuli", "responses", "no_response", "extra_response", "kept",
              "first_after_gap", "rt_out", "both_rules", "analysed",
              "errors", "error_rate_pct"]]
