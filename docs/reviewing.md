# Reviewing this code

A suggested order, and the specific things worth checking. Roughly two hours
for the whole pass; the first two sections are the ones that matter most.

---

## Start here

```bash
pip install -r requirements.txt
pytest                        # 39 tests, under a second
python run.py verify-paper    # 117 comparisons against data/derived/
```

`verify-paper` transcribes every value the manuscript reports, with the section
it came from, into `src/paper_check.py`, and compares each against the derived
data. A failure is tagged `DATA` (the repository output needs regenerating) or
`PAPER` (the manuscript's own arithmetic needs checking).

It should report `117 matched, 0 mismatched`. The remaining notices are
informational — the sample is not distributed by design, some values are
back-calculations, some record provenance.

**A passing check does not mean the analysis is correct.** It means the numbers
in the manuscript and the numbers this code produces are the same.

The transcription in `src/paper_check.py` is typed by hand, which was for a
long time the one step with nothing checking it -- and it failed exactly
there: a value was filled in from the repository rather than the paper, so the
comparison agreed with itself and a stale section went unnoticed for two
rounds. `tests/test_transcription.py` now checks every transcribed number
against `docs/paper_values.txt`, which is extracted mechanically from the
manuscript source, and checks that each came from the section it claims.

---

## 1. Are the events what the manuscript says they are?

`src/prepare.py`, then `docs/definitions.md`.

Two things to confirm:

- **Event B is net material change**, not the preregistered static-exchange
  criterion. The preregistration expected even exchanges to resolve to zero
  and be excluded; verification showed 55.7% of events selected that way
  involved no net change and 12.7% involved a net gain. The withdrawn path is
  still in the code as `events_B_see`, for the supplementary comparison only.
- **`net_mat` is computed before the measurement floor is applied.** Material
  difference is a property of the position, not of the clock, so discarding
  fast moves first would lose ply pairs unnecessarily. `z_pre3` is computed
  after standardisation, because it depends on it. The ordering is asserted in
  `tests/test_prep.py`.

## 2. Are the standard errors what the manuscript says they are?

`src/analysis.py`, function `cluster_stats`.

Every `se_*.csv` carries, for each lag, the event-weighted effect (`e`), the
player-clustered standard error (`se`), the player-weighted effect (`epw`), and
the number of clusters (`np`). The manuscript reports `e` with `se`.

Worth checking: that `np` is present and plausible (roughly 1,200 players per
group), and that `e` and `epw` do not diverge sharply. A large gap means a few
players dominate the event-weighted estimate.

## 3. Is the matching doing what it claims?

`src/analysis.py`, function `caliper_mask`.

Six variables — player, win probability, clock, legal move count, ply, and
pre-event speed — each within 0.6 SD. An event missing any of them is
discarded, not matched on the remainder. Confirm `strict=True` is the default.

## 4. Static exchange evaluation

`src/see.py`, with `tests/test_see.py`.

Self-contained and separately tested. Two properties to confirm: the king is
the last piece considered as a recapturer, never the first, and it cannot
recapture into a square that is still defended. The test file contains the
position that originally exposed the defect.

## 5. Spot-check a derived file against the code that wrote it

Pick any `data/derived/*.csv` and trace it back to its stage in `run.py`.
`python run.py <stage> --help` describes what the stage does and what to check
in it. Every source file opens with the same kind of note.

---

## Checklist

```
□ pytest passes
□ python run.py verify-paper reports 0 mismatched
□ verify-paper's notices contain no "read from transcribed values"
□ no absolute paths in the source
□ requirements.txt installs and the pipeline runs on it
□ the same command run twice gives the same result
```

The third line matters: if `verify-paper` falls back to `src/external_values.py`
for a value, that value is not being checked against a repository output. The
flanker comparison is the only one that should legitimately do so.

---

## If something does not match

The disagreement is the finding — record it before changing anything. Then:

- If `data/derived/` disagrees with the manuscript, the pipeline needs
  rerunning from a single commit, not patching file by file. Values from
  different runs are what produced the largest defect found in this review
  (see `corrections.md`).
- If the manuscript's own arithmetic does not hold, `verify-paper` tags it
  `PAPER` and the transcription in `src/paper_check.py` names the section it
  came from.
- **When a manuscript value changes, change the transcription too.** Otherwise
  the next run reports a mismatch. That is the point of the mechanism.
