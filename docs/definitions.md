# Definitions and deviations from the preregistration

This file records the operational definitions used in the reported analysis and
every point at which they differ from what was registered. It exists so that the
difference between the two remains visible to anyone reading the code.

---

## Event labels

The preregistration named three event types; the manuscript reports two. The
labels do not line up, so all three namings are given here. **The code follows
the manuscript.**

| Preregistration | Manuscript | Code | Content |
|---|---|---|---|
| A | A | `events_A` | Blunder — win probability drop |
| B | — | — | Errors under acute time pressure. Too few events survived matching, so it was dropped |
| C | B | `events_B` | Material net loss |
| C (as registered) | — | `events_B_see` | The withdrawn SEE criterion, kept for the supplementary comparison |

The dropped type is what `config.GRID_TIME_FLOOR` is left over from.

---

## Events

### Event A — Blunder

A move is classed as a blunder when the player's win probability drops by at
least 10 percentage points, provided the win probability before the move lay
between 10% and 90%.

The bounds exist because a 10-point drop means something different at 0.95 than
at 0.50. Engine evaluation is present only for games a player has submitted for
analysis: 12.4% to 28.4% of games in the lower tiers and 93.9% in the titled
tier.

Implemented in `src/prepare.py` (`events_A` / `mask_A`) from the `wp_delta`
and `wp_before` columns.

There is no Event B. A time-pressure event keyed to a remaining-time floor was
considered during Phase 1 and dropped; `config.GRID_TIME_FLOOR` is what remains
of that exploration. See the label table above; the code follows the
manuscript's A/B naming.

### Event B — Material net loss

The material differential immediately before a player's move is compared with
the differential at that player's next move. A decrease constitutes an event and
its magnitude is the size of the decrease.

An even exchange produces no net change and is therefore not an event. Event B
requires no engine evaluation and is computed on the full corpus.

Implemented by `run.py prep`, which calls
`src.prepare.net_material_change()` on the `mat_diff` column produced by
`src/extract.py`. The event mask is `src.prepare.events_B` / `mask_B`.

`mat_diff` is recorded immediately before a move, from the mover's point of
view, with the king excluded (pawn 1, knight 3, bishop 3, rook 5, queen 9).

---

## The revision to Event B

**What was registered.** The event now labelled B (registered as C) was specified as a move after which the
opponent's maximum static exchange evaluation exceeded zero, with the statement
that even exchanges resolve to SEE = 0 and are thereby excluded.

**What the implementation did.** The second half of that statement is false.
Consider white playing Nxf5, capturing a knight, and black replying gxf5,
capturing the knight back. The net change in material is zero, but the maximum
SEE available to black in the resulting position is 3. The move is counted as a
three-point loss.

**How large the problem was.** Of the events identified under the registered
criterion, across both analysis tables:

| Actual net change | Count | Share |
|---|---:|---:|
| Loss | 510,204 | 31.5% |
| Zero (even exchange) | 904,693 | **55.8%** |
| Gain | 206,437 | **12.7%** |

Events classified as nine-point losses had a mean net change of −1.00.

The criterion identifies 1,689,748 events; the 1,621,334 above are those with
a defined net change. The rest sit at the edge of a game, where the material
differential on the following move is not observed.

**What changed.** The definition was revised to net material change and all
analyses were rerun. Event count fell from 1,689,748 to 932,846 (7.61 to 4.20
per game). At the immediately following move, on the untitled tiers under the
legal-move-count specification, the effect went from −0.214 to −0.593.

**Where both live in the code.** `src/see.py` implements the registered
criterion and is retained so the supplementary comparison can be regenerated.
The net-loss definition is the one used for every number in the manuscript.

`events_B` is the net-loss definition. The registered criterion is
`events_B_see`, reachable only through `run.py lag ... --see`, and its output is
labelled `material_see` so the two never end up in the same column of
`lag_profiles.csv` again.

---

## Other deviations

### Pre-event speed was added as a control

Not in the registered specification. Inspection of pre-event trends showed that
the three moves before a blunder were already slow -- the mean event-minus-
control difference over lags -1 to -3, with no pre-speed control, is +0.064 in
the lower tiers and +0.079 in the titled tier -- and that the blunder move
itself sat at z = +0.47. The mean standardised time of the three preceding
moves was therefore added as a matching variable with a 0.6 SD caliper.

Three approaches are reported: no control (`run.py lag <which> nlegal`),
0.6 SD matching (`run.py lag <which> zpre`, `run.py ps6`), and
difference-in-differences (`run.py did`).

`z_pre3` is indexed by ply, not by row position: it is the mean of
z(t−2), z(t−4), z(t−6) for that player in that game, and is missing if any of
the three is absent. Events with a missing `z_pre3` are dropped rather than
matched on the remaining variables.

### The flanker comparison was exploratory

Not registered. Added to provide a reference point computed by identical
procedures rather than a comparison across differing analytic conventions.

### Standard errors

The manuscript reports player-clustered standard errors throughout. An earlier
version of the SE stage computed them at the event level, which understates them
by roughly 1.6× because events within a player are correlated. The current
version clusters by player, in one place (`src.analysis.cluster_stats`).

Two weightings are reported side by side because they are not the same
quantity. `e{k}` is the mean over events; `se{k}` is the standard error of the
per-player means, which weights players equally. `epw{k}` gives the
player-weighted mean so the two can be compared, and `np{k}` gives the number
of clusters. A large gap between `e{k}` and `epw{k}` means events are
distributed very unevenly across players.

---

## Fixed parameters

All are in `src/config.py`.

| Parameter | Value | Alternatives tested |
|---|---|---|
| Opening cut | 10 plies | 14, 20 |
| Measurement floor | 1 s | 2, 3 |
| Blunder threshold | 10 pp | 20, 30 |
| Epoch window | 3 moves | — |
| Matching caliper | 0.2 SD | 0.1, 0.4, 0.8 |
| Pre-speed caliper | 0.6 SD | — |
| Minimum games | 30 | — |
| Time control | 10+0 rated | — |

---

## Sample

Five tiers by rating. 500 players sampled at random from each of the four lower
tiers; all 136 eligible titled players (FM or above) included. Candidate Master
and National Master were excluded because they overlap in strength with the
untitled 2100+ tier.

The minimum-games criterion retains between 19.1% and 27.9% of players depending
on tier. This was checked before any outcome was examined.

The titled aggregation covers 2024-01 to 2024-06. An earlier version of the
titled scan covered February to June only, while stage 2 extracted January to
June, so titled accounts active only in January were missed.

### The sample is not distributed

The selected players are written to `out/sample.parquet` and, for the titled
tier, `out/sample_fm.parquet`. **Neither ships with this repository.**

Lichess account names are quasi-identifiers, and the titled tier — 136 players
holding a FIDE title of FM or above — is small enough that individuals could be
matched to public profiles. Redistributing the games is one thing; publishing a
list of named individuals together with per-player effect estimates is another.

What reproduction needs is the sampling rule, not the list. The rule is in
`src/extract.py` (`build_sample`, `build_titled_sample`) and every parameter it
uses — tier boundaries, the minimum-games criterion, the random seed — is in
`src/config.py`. Running stage 1 against the same monthly archives reproduces
the same sample; `run.py sample` then checks its tier counts against Table 1.

`run.py anonymize` replaces account names with salted hashes in any derived
file that still carries them, if you want to distribute one.
