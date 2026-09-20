# Post-error adjustment in online chess

Analysis code for *Perceptibility of Error Outcomes Determines the Direction of
Post-Error Adjustment: Evidence From 6.7 Million Moves in Online Chess.*

The study asks whether the change in decision time after an error depends on
whether the consequence of that error is visible. Two kinds of error are
identified independently in the same games: moves that an engine scores as large
evaluation drops, and moves after which material actually disappears from the
board.

- **Preregistration** — https://doi.org/10.17605/OSF.IO/VQ7XC
- **Archive** — https://doi.org/10.5281/zenodo.22787114
- **Repository** — https://github.com/JeongNeuro/chess-post-error
- **Manuscript** — under review at *Journal of Cognitive Psychology*

---

## What is here

```
run.py       single entry point — every stage is a subcommand
src/         analysis modules (seven files)
data/
  derived/   every value the manuscript reports (effects, SEs, counts)
docs/        definitions and environment notes
```

`python run.py --help` lists the stages; `python run.py <stage> --help` gives
that stage's arguments and a note on what to check in it.

Raw game archives are **not** included. They are ~30 GB per month and are
redistributed by Lichess under CC0; `run.py scan` downloads what it needs.

---

## Reproducing the analysis

Total runtime is roughly 1 hour 45 minutes for the untitled tiers,
dominated by stage 2. Measured over all 397 shards of January 2024: 18
minutes to scan, 85 to extract, under a minute to prepare. The scan figure
assumes a fast link; it is almost entirely download.

Including the titled tier adds 8 to 9 hours, again almost all of it download:
`scan-titled` reads every shard of six months to find 136 accounts.
That stage skips existing outputs, so it can be split across sessions.

### 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
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
Processing the full corpus would take about 21 hours; this two-stage
approach reduces it to under two.

### 2. Reconstruct positions

```bash
python run.py extract 0 421     # lower four tiers
python run.py extract-titled    # titled players, Jan–Jun
```

Replays each game with `python-chess`, computing per-ply move time, win
probability, legal move count, static exchange evaluation, and the material
differential. Writes `out/plies/` and `out/plies_fm/`.

**This is the slow step (~85 min).** It is shard-parallel; the arguments are a
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

These produce the rest of what the manuscript reports.

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
python run.py nextq --mode split   # quality of the following moves
python run.py nextq --mode split --qpre    # with the regression control
python tools/window_definition.py  # what the window definition costs
```

### 6b. The flanker comparison

Computed here rather than quoted. The dataset is OpenNeuro ds004883
(Clayson et al., 2024) and is **not** redistributed with this repository, for
the same reason the chess archives are not.

```bash
# 1. fetch the derivatives (about 40 MB of event tables)
#    https://openneuro.org/datasets/ds004883
# 2. the folder must hold trials_v2.csv, exclusions.csv and raw/
export CHESS_FLANKER=/path/to/ds004883-derivatives
python run.py flanker
```

Writes `flanker.csv`, `flanker_reliability.csv` and `flanker_by_version.csv`.
The three task versions are analysed separately and never pooled: the dataset
exists to show they are not equivalent, and their error rates run from 9% to
18%. The stage stops if the per-version analysed counts do not sum to the
273,475 trials the Method states.

### 7. Complexity validation

```bash
python run.py cx-fens 24 34   # sample positions (re-replays games)
python run.py cx              # Stockfish multi-PV
python run.py cx-model        # fit cx_pred, add it to prepared*.parquet
```

`cx-model` prints the two correlations the manuscript reports (legal
move count vs. measured complexity, and the prediction model). Check them
against the paper before citing.

### 8. What this produces

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
check, and which numbers in the paper depend on it. The files are ordered to
follow the pipeline, so reading them in the order of the *Module reference*
table below follows the data.

`docs/definitions.md` defines the events, the matching variables and every
column name used in `data/derived/`. It is needed to read those files at all.

## Module reference

| File | Lines | Purpose |
|---|---:|---|
| `run.py` | 1,984 | Every stage as a subcommand; the whole pipeline in one place |
| `src/config.py` | 216 | All thresholds and paths — opening cut, measurement floor, epoch window, calipers, archive URLs |
| `src/see.py` | 188 | Static exchange evaluation (self-contained) |
| `src/extract.py` | 762 | PGN parsing and win probability → shard scan → stratified sampling → board replay and per-ply features |
| `src/prepare.py` | 548 | Net material change, pre-event speed, z standardisation, event definitions, calipers |
| `src/analysis.py` | 660 | Caliper matching, player-clustered SEs, lag decomposition, pre-trend, mixed-effects models |
| `src/complexity.py` | 178 | Stockfish interface and the complexity prediction model |
| `src/flanker.py` | 281 | The flanker comparison (OpenNeuro ds004883), computed here rather than quoted |

Eight files, about 4,820 lines. Each is ordered to follow the pipeline and
opens with a review note saying what to check in it.

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
| `legal_change.csv` | Mean change in legal move count by size of loss, over all events | Results text |
| `mixed_B.csv` | Mixed-effects coefficients with and without the win-probability covariate, and the paired-difference estimate on each corresponding subset | Results text |
| `robustness.csv` | The six registered specifications at alternative levels | Results text |
| `se_tier.csv` | Effects by tier | Results text |
| `event_characteristics.csv` | Win-probability drop, material lost, and subsequent-blunder rate for each event group | Results text |
| `next_quality_split.csv` | Quality of the player's next moves after an event, three-group partition | Results text |
| `next_quality_split_qpre.csv` | The same, additionally matched on the mean of the outcome over the player's preceding three moves | Results text |
| `next_quality_dose.csv` | The same, by material value and direction | Results text |
| `next_quality_tier.csv` | The same, by tier | Results text |
| `flanker.csv` | The flanker comparison, per task version | Fig 2g |
| `flanker_reliability.csv` | Split-half reliability against errors retained per participant, flanker | Fig 2g |
| `see_reimplementation_columns.csv` | How far `see_loss` and `max_see_mine` move between the two implementations | Appendix |
| `see_reimplementation_transitions.csv` | Which values change into which | Appendix |
| `see_reimplementation_events.csv` | Event counts under each criterion, before and after | Appendix |
| `window_definition.csv` | t+1 under the three window definitions, four untitled tiers | Appendix |
| `flanker_by_version.csv` | Stimuli, exclusions, analysed trials and error rate per flanker version | Appendix |
| `se_split_4tier.csv` | The three-group partition on the untitled tiers only, for comparison with the five-tier primary analysis | — |
| `per_player_t1_4tier.csv` | Per-player effects for the same four-tier comparison | — |

Fig 2g (the flanker comparison) is produced by `src/flanker.py` from
OpenNeuro ds004883, which must be downloaded separately.

The `next_quality_*.csv` files carry two effects per row: `effect` weights
every event equally, `effect_pw` weights every player equally. They agree in
sign throughout; where they differ in size `effect_pw` is the larger, so the
reported `effect` is the conservative one.

They also run from lag -3 to +3. An event is a bad move by definition, so if
move quality drifts within a player the event is drawn from a temporary trough
and the next move returns towards that player's usual level with no adjustment
at all. The pre-event lags show whether there is such a trough. `--qpre` adds
the second check: it matches on the mean of the outcome over the player's own
preceding three moves, at the same caliper as pre-event speed, each outcome
against its own history. `run.py nextq --mode split --qpre` writes the file
above; `--qpre-sd` sets that caliper in SD.

Standard errors in these files are clustered by player. Effects are in
within-player standard deviations of log move time. Each `se_*.csv` carries,
for every lag, the event-weighted effect (`e`), the clustered SE (`se`), the
player-weighted effect (`epw`), and the number of clusters (`np`).

`event_type` in `lag_profiles.csv` distinguishes `material` (net loss, the
manuscript definition) from `material_see` (the registered criterion, retained
for the supplementary comparison). They are not interchangeable — see below.
Under the same specification the registered criterion gives -0.214 at t+1
against -0.593 for net loss, because more than half of what it counts as an
event involves no net change in material.

The three `see_reimplementation_*.csv` files compare the static exchange
evaluation as this code computes it against the values in the table released
with v1.0. They differ on 0.55% of moves, one-sidedly, and no move changes
its classification under the manuscript's `net_mat < 0` criterion. The code
that produced the earlier values is not preserved, so the direction of the
change is the only evidence about its cause.

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
submitted for analysis — 12.4% to 28.4% of games in the untitled tiers and
93.9% among titled players. Matching on win probability would therefore make
each tier a differently self-selected subsample, which is the comparison the
manuscript's tier-level results rest on.

On the subsample where both specifications can be computed, they give −0.639
and −0.644, a difference far smaller than either standard error, while the
five-variable set retains about half again as many events (8,807 against
5,573). `run.py robustness`
reports both as `caliper_vars`.

Blunder events are unaffected: they require an engine evaluation by
definition, so the caliper costs them nothing either way.

The variable sets are `MATCH_VARS` and `MATCH_VARS_WP` in `src/config.py`.

---

## Two definitions of material loss

The preregistration specified material loss as a move after which the
opponent's maximum static exchange evaluation exceeded zero, stating that even
exchanges would resolve to zero and be excluded. **They do not.** Verification
showed that 55.8% of the events identified that way involved no net change in
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
direction of post-error adjustment: Evidence from 6.7 million moves in online
chess. [Preprint]
```

Game data are from the Lichess open database (https://database.lichess.org),
released under CC0. The flanker comparison uses OpenNeuro ds004883 and is
computed by `src/flanker.py`; that dataset must be downloaded separately.

## Licence

Code is released under the MIT Licence (`LICENSE`); derived data in
`data/derived/` are released under CC BY 4.0 (`data/LICENSE`).

The game records behind those tables are the Lichess open database, released
by Lichess under CC0, and are not redistributed here. The flanker comparison
uses OpenNeuro ds004883 under its own licence, also not redistributed.
