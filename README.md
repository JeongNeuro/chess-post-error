# Post-error adjustment in online chess

Analysis code for *Perceptibility of Error Outcomes Determines the Direction of
Post-Error Adjustment: Evidence From 3.9 Million Moves in Online Chess.*

The study asks whether the change in decision time after an error depends on
whether the consequence of that error is visible. Two kinds of error are
identified independently in the same games: moves that an engine scores as large
evaluation drops, and moves after which material actually disappears from the
board.

- **Preregistration** — https://doi.org/10.17605/OSF.IO/VQ7XC
- **Repository** — https://github.com/JeongNeuro/chess-post-error
- **Manuscript** — under review at *Journal of Cognitive Psychology*

---

## What is here

```
run.py       single entry point — every stage is a subcommand
src/         analysis modules (five files)
tests/       unit tests (pytest)
data/
  derived/   outputs small enough to version (effects, SEs)
docs/        definitions, corrections, review guide, environment notes,
             and the extracted manuscript values (five files, all English)
check_external_docs.py   compares documents outside this repository
             (cover letter, README) against the manuscript
```

`python run.py --help` lists the stages; `python run.py <stage> --help` gives
that stage's arguments and a note on what to check in it.

Raw game archives are **not** included. They are ~30 GB per month and are
redistributed by Lichess under CC0; `run.py scan` downloads what it needs.

---

## Reproducing the analysis

Total runtime is roughly 1.5 hours for the untitled tiers, dominated by
stage 2. Including the titled tier adds 8 to 9 hours, almost all of it
download: `scan-titled` reads every shard of six months to find 136 accounts.
That stage skips existing outputs, so it can be split across sessions.

### 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Run from the repository root.

```bash
python run.py scan 0 421       # correct
cd src && python ../run.py     # will not find data/derived/
```

Intermediate files go to `out/` by default. Set `CHESS_WORK` to put them
elsewhere — useful if the archive shards do not fit on the same volume.

```bash
export CHESS_WORK=/mnt/scratch/chess
```

Stockfish 16 is required only for the complexity validation (step 7). On
Debian or Ubuntu: `apt-get install stockfish`. Set `CHESS_STOCKFISH` if it is not on
PATH. See `docs/environment.md`.

### 1. Select the sample

```bash
python run.py scan 0 421     # lower four tiers, Jan 2024
python run.py scan-titled    # titled accounts, all six months
python run.py sample         # → out/sample.parquet
```

`sample` checks the result against Table 1 of the manuscript and warns if the
tier counts do not match. `extract` refuses to run on a sample that fails that
check unless you pass `--force`.

The sample itself is **not distributed** — see *The player sample* below.

`scan` reads account names, ratings, and game counts from the monthly archive
without parsing game content, writing `out/players/`. `sample` then aggregates
those shards and draws the stratified sample.

The titled tier is drawn from six months (2024-01 to 2024-06) because one
month yields only 26 accounts with ≥30 games. Repeat `scan-titled` for each.

This stage exists so that board reconstruction runs only for selected players.
Processing the full corpus would take about 21 hours; this two-stage approach
reduces it to roughly 1.4 hours.

### 2. Reconstruct positions

```bash
python run.py extract 0 421     # lower four tiers
python run.py extract-titled    # titled players, Jan–Jun
```

Replays each game with `python-chess`, computing per-ply move time, win
probability, legal move count, static exchange evaluation, and the material
differential. Writes `out/plies/` and `out/plies_fm/`.

**This is the slow step (~45 min).** It is shard-parallel; the arguments are a
shard range, so the work can be split across sessions.

### 3. Build the analysis table

```bash
python run.py prep        # lower four tiers
python run.py prep fm     # titled
```

Applies the measurement floor, standardises log move time within player,
computes net material change (`net_mat`) and the pre-event speed control
(`z_pre3`), and writes `out/prepared.parquet` and `out/prepared_fm.parquet`.

This is where the manuscript's Event B is defined. See `docs/definitions.md`.

### 4. Main analyses

```bash
python run.py se split    # three-group partition, the headline result
python run.py se dose
python run.py se tier
python run.py lag-all     # lag profiles, all event × tier × spec
```

`se` writes `data/derived/se_*.csv` directly, and `split` also writes
`per_player_t1.csv`. `lag-all` writes per-run parquet files and then runs
`aggregate` to produce `data/derived/lag_profiles.csv`.

Standard errors are clustered by player throughout.

### 5. The remaining reported analyses

These produce the rest of what the manuscript reports. Until they are run,
`verify-paper` falls back to the transcribed values in `src/external_values.py`
and says so in its output.

```bash
python run.py lag B zpre --by-tier   # Fig 1a — one curve per tier
python run.py lag A zpre --by-tier   # Fig 1b
python run.py reliability            # Table 2 and Fig 2d/2e
python run.py bins                   # Fig 2b
python run.py describe               # event characteristics
python run.py robustness             # the six registered specifications
python run.py aggregate              # fold the lag runs into lag_profiles.csv
```

`lag-all` already includes the two `--by-tier` runs.

### 6. Supporting analyses

```bash
python run.py ps6 B                # pre-speed matched, with pre-trend
python run.py did B                # difference-in-differences comparison
python run.py mixed B              # preregistered mixed-effects model
python run.py lag B nlegal --see   # registered (withdrawn) SEE definition
```

### 7. Complexity validation

```bash
python run.py cx-fens 24 34   # sample positions (re-replays games)
python run.py cx              # Stockfish multi-PV
python run.py cx-model        # fit cx_pred, add it to prepared*.parquet
```

`cx-model` prints the two correlations the manuscript reports (legal
move count vs. measured complexity, and the prediction model). Check them
against the paper before citing.

### 8. Check the numbers

```bash
python run.py verify-paper
```

**This repository stops at the statistics.** It produces the CSVs in
`data/derived/`; it does not draw the figures. Every value plotted in the
manuscript's figures is a row in one of those files, so the figures can be
redrawn with whatever tool you prefer. The table under *Derived data* below
says which file backs which panel.

---

## The player sample

The player sample is not distributed. Lichess account names are
quasi-identifiers, and the titled tier is small enough that individual
players could be matched to public profiles. The sampling rule is fully
specified in `src/extract.py`; running stage 1 against the same monthly
archives reproduces the sample. Tier definitions, the minimum-games
criterion, and the random seed are all in `src/config.py`.

```bash
python run.py scan 0 421     # lower four tiers, 2024-01
python run.py scan-titled    # titled accounts, 2024-01 .. 2024-06
python run.py sample         # → out/sample.parquet, out/sample_fm.parquet
```

`scan-titled` reads every shard of all six months to find titled accounts, so
it is the slowest stage. Existing outputs are skipped, so it can be split
across sessions. One month yields only 26 accounts with ≥30 games, which is
why the titled window is six months.

`run.py sample` checks the result against Table 1 and reports any mismatch.

## Reviewing the code

Each source file carries a short review note at the top: what it does, what to
check, and which numbers in the paper depend on it. `docs/reviewing.md` gives a
suggested order and the specific values to verify against.
`docs/corrections.md` records the defects found in the pre-release review, which
reported values they affected, and which results did not reproduce.

```bash
pytest                           # 39 tests, under a second
python run.py verify-paper       # every number in the manuscript vs data/derived/
python check_external_docs.py --tex paper.tex    # cover letter, README, …
```

`verify-paper` transcribes every value reported in the manuscript
(`src/paper_check.py`, with the section each came from) and compares it against
the derived data. Failures are tagged `DATA` (regenerate the repository output)
or `PAPER` (check the manuscript's own arithmetic). All 117 comparisons
currently agree; the remaining fourteen notices are informational.

Transcription is checked separately. `docs/paper_values.txt` is extracted
mechanically from the manuscript source — every number, grouped by the section
it appears in — and the test suite confirms that each transcribed value occurs
in the section it claims to come from. A value that exists somewhere in the
paper but was filed under the wrong section is caught; so is a value that
appears in `paper_check.py` but not in the manuscript at all. Both failure
modes were introduced deliberately and confirmed to fail before this was
relied on.

This exists because the transcription was twice filled from the repository's
own outputs rather than from the manuscript, which makes the comparison
vacuous. `docs/corrections.md` records both occasions.

`verify-paper` sees only what is inside this repository. A cover letter or a
preprint title page quotes the same figures and is checked by nothing, so
`check_external_docs.py` reads the claims out of the manuscript source and
compares them against those documents. It matches on wording where the wording
is shared and on the values themselves where it is not: a letter that restates
a superseded move count in a sentence the manuscript never contains slips past
the first pass, and only the second catches it. Both passes were confirmed
against a document carrying the old figure.

This file is one of the documents checked. It is located from the script’s own
directory rather than the working directory, because the check is normally run
from the folder holding the manuscript — where a bare `README.md` resolves to
something else, or to nothing, and is skipped in silence. It was skipped that
way, and this README carried a superseded move count in its title line and
citation until that was fixed.

## Module reference

| File | Lines | Purpose |
|---|---:|---|
| `run.py` | ~1540 | Every stage as a subcommand; the whole pipeline in one place |
| `src/config.py` | ~180 | All thresholds and paths — opening cut, measurement floor, epoch window, calipers, archive URLs |
| `src/see.py` | ~185 | Static exchange evaluation (self-contained, separately tested) |
| `src/extract.py` | ~640 | PGN parsing and win probability → shard scan → stratified sampling → board replay and per-ply features |
| `src/prepare.py` | ~295 | Net material change, pre-event speed, z standardisation, event definitions, calipers |
| `src/analysis.py` | ~560 | Caliper matching, player-clustered SEs, lag decomposition, pre-trend, mixed-effects models |
| `src/complexity.py` | ~175 | Stockfish interface and the complexity prediction model |
| `src/paper_check.py` | ~845 | Every value reported in the manuscript, transcribed with its source; the comparison against `data/derived/`; and the cross-checks between derived files |
| `src/external_values.py` | ~140 | Values the manuscript reports that this pipeline does not produce — the flanker comparison (OpenNeuro ds004883), marked `external` |

Nine files, about 4,600 lines, plus tests. Each file is ordered to follow the
pipeline and opens with a review note saying what to check in it.

## Derived data

| File | Contents | Figure / table |
|---|---|---|
| `lag_profiles.csv` | Effect, SE, t, and n at each lag, by event type, tier, and control specification | Fig 1a, 1b |
| `se_split.csv` | Three-group partition with SEs | Fig 1c, Fig 2f |
| `per_player_t1.csv` | Per-player effects at t+1 | Fig 2a |
| `legal_bins.csv` | Effect within bins of equal change in legal move count | Fig 2b |
| `se_dose.csv` | Effects by material value and direction | Fig 2c |
| `reliability_curve.csv` | Reliability against events retained per player | Fig 2d |
| `reliability.csv` | Split-half reliability and σ_b by event type | Table 2, Fig 2e |
| `legal_bins_queen.csv` | The same legal-move bins restricted to nine-point losses | Results text |
| `mixed_B.csv` | Mixed-effects coefficients with and without the win-probability covariate, and the paired-difference estimate on each corresponding subset | Results text |
| `robustness.csv` | The six registered specifications at alternative levels | Results text |
| `se_tier.csv` | Effects by tier | Results text |
| `event_characteristics.csv` | Win-probability drop, material lost, and subsequent-blunder rate for each event group | Results text |
| `se_split_4tier.csv` | The three-group partition on the untitled tiers only, for comparison with the five-tier primary analysis | — |
| `per_player_t1_4tier.csv` | Per-player effects for the same four-tier comparison | — |

Fig 2g (the flanker comparison) is not produced here — see
`src/external_values.py`.

Standard errors in these files are clustered by player. Effects are in
within-player standard deviations of log move time. Each `se_*.csv` carries,
for every lag, the event-weighted effect (`e`), the clustered SE (`se`), the
player-weighted effect (`epw`), and the number of clusters (`np`).

`event_type` in `lag_profiles.csv` distinguishes `material` (net loss, the
manuscript definition) from `material_see` (the registered criterion, retained
for the supplementary comparison). They are not interchangeable — see below.

The `player` column in `per_player_t1.csv` holds salted hashes (`P` followed
by twelve hex digits), not account names. `run.py anonymize` produced them; the
salt and the mapping stay in `out/` and are excluded by `.gitignore`. The hashes
are stable within the file, so per-player effects can be linked across rows, but
they cannot be resolved back to accounts without the salt. See *The player
sample* above for why.

---

## Matching variables

Events are matched to control windows on five variables: the player
(identical), remaining time, legal move count, and ply (each within 0.2 SD),
together with the mean standardised time of the three preceding moves
(within 0.6 SD).

The preregistration included win probability as a sixth. It is not used in the
primary analysis. Material-loss events are defined without reference to the
engine, and requiring an evaluation restricts them to the games a player has
submitted for analysis — 12.7% to 29.1% of games in the untitled tiers and
93.9% among titled players. Matching on win probability would therefore make
each tier a differently self-selected subsample, which is the comparison the
manuscript's tier-level results rest on.

On the subsample where both specifications can be computed, they give −0.560
and −0.587, a difference smaller than either standard error, while the
five-variable set retains about half again as many events. `run.py robustness`
reports both as `caliper_vars`.

Blunder events are unaffected: they require an engine evaluation by
definition, so the caliper costs them nothing either way.

The variable sets are `MATCH_VARS` and `MATCH_VARS_WP` in `src/config.py`.

---

## Two definitions of material loss

The preregistration specified material loss as a move after which the
opponent's maximum static exchange evaluation exceeded zero, stating that even
exchanges would resolve to zero and be excluded. **They do not.** Verification
showed that 55.7% of the events identified that way involved no net change in
material and 12.7% involved a net gain.

The analysis reported in the manuscript therefore uses **net material change**:
the material differential immediately before a player's move compared with the
differential at that player's next move, counted as an event only when it
decreased.

Both paths are present in the code. `src/see.py` implements the original
criterion and is retained for the supplementary comparison, reachable through
`events_B_see` and `run.py lag ... --see`; the net-loss definition is computed by
`run.py prep` from the `mat_diff` column produced by the extract stage. See
`docs/definitions.md`.

---

## Citation

```
Jeong, Y., & Kim, Y. (2026). Perceptibility of error outcomes determines the
direction of post-error adjustment: Evidence from 3.9 million moves in online
chess. [Preprint]
```

Game data are from the Lichess open database (https://database.lichess.org),
released under CC0. The flanker comparison uses OpenNeuro ds004883; that
analysis is not part of this repository — its values are held in
`src/external_values.py` and marked `external`.

## Licence

Code: MIT. Derived data: CC0.
