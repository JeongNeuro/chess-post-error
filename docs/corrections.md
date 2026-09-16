# Corrections made before release

The analysis code was re-examined in full before this repository was made
public. This file records the defects that were found, whether each one
affected a number reported in the manuscript, and which reported results did
not reproduce.

It is not a development log. It lists the substantive findings only.

---

## The matching variables changed

The registered specification matched on win probability. Material loss is
defined without reference to the engine, so requiring an evaluation confines
those events to the games a user submitted for analysis — 12.7% of games in the
lowest tier against 93.9% in the titled tier, a subset that is neither random
nor comparable across tiers.

On the engine-evaluated subsample, where both specifications can be computed,
the estimate moves from −0.560 to −0.587: less than either standard error. The
five-variable set is therefore the main specification and the registered one is
reported under robustness (`robustness.csv`, rows `caliper_vars`).

Every material-based result changed as a consequence. The blunder-based ones
did not: those events require an evaluation anyway, so the caliper cost them
nothing.

---

## One estimand throughout

Every stage reports the difference at t+1. Three did not, and were corrected:
`bins` and `reliability` were averaging t+1..t+3, and so was the mixed model's
dependent variable. Because t+2 and t+3 sit near zero, the average ran about a
third of the t+1 value — two numbers for one quantity, side by side in the same
Results section.

`ps6` and `did` keep the window. A difference-in-differences needs its pre- and
post-windows symmetric, so there the window is the design rather than a choice.
Their values are not comparable with the rest and the manuscript should say so.

---

## Defects that affected reported values

Every item here required the pipeline to be rerun; the values now in
`data/derived/` are from a single run of the code as released.

| Defect | Consequence |
|---|---|
| **Event B used the withdrawn criterion.** The analysis path selected events by static exchange evaluation (`see_loss >= 1`), the preregistered rule, not by net material change, the rule the manuscript describes. | All Event B results. The two criteria differ substantially — see `definitions.md`. |
| **Static exchange evaluation treated the king as the cheapest attacker.** The king was valued at 0 in the routine that picks the least valuable piece, so it was always selected to recapture, and was then itself captured for 0. The exchange sequence unwound incorrectly from that point. | `see_loss`, `max_see_mine`, and every result computed under the withdrawn criterion. A regression test is in `tests/test_see.py`. |
| **Matching calipers were skipped silently.** When a matching variable was missing for an event, that variable was dropped and the event was matched on the rest. The control pool was filtered with `dropna`, so only the event side was loosened. The manuscript states that all six variables fall inside the caliper. | All matched estimates. Events with a missing caliper variable are now discarded (`strict=True`). |
| **The pre-event speed control was indexed by row, not by ply.** `z_pre3` was computed with a positional shift. Because the measurement floor removes moves below one second, "the previous three moves" could reach further back than intended, while the surrounding lag windows were ply-indexed. Two indexing conventions were in use within one analysis. | `z_pre3`, and every specification that controls for it. Now ply-indexed throughout; missing if any of the three preceding own-moves is absent. |
| **The titled-tier sampling window was missing January.** The month list read `{2,3,4,5,6}` while the extraction stage, the file docstring, and the README all specified 2024-01 to 2024-06. Titled accounts active only in January were absent from the sample. | Titled-tier composition. The window is now taken from one constant, `config.TITLED_MONTHS`. |
| **The event-sampling seed differed between analyses.** The standard-error stage used `random_state=1` and the lag stage `0`. Both subsample events above a cap, so the two analyses were not using the same set of events. | Any comparison across the two stages. Unified to `config.EVENT_SAMPLE_SEED`. |
| **The three-group decomposition existed in two versions.** The path producing the main result matched on six variables with no player cap; a second path matched on five, capped players at 110 per tier, and applied no event cap. | This is why the reported effects and their standard errors did not correspond: they came from different runs. The effects matched one version, the standard errors the other — the discrepancy was a uniform factor of about 1.6. Both now use the main specification. |

### Derived data and code were out of step

The `data/derived/*.csv` files originally distributed lacked the cluster-count
columns that the code writes, which showed they were produced by a different
version of it. Effects agreed with the manuscript; standard errors were 1.5 to
1.8 times larger throughout. The same quantity appeared in two figures with two
different values.

`run.py verify-paper` now compares every reported value against
`data/derived/`, and additionally cross-checks the derived files against each
other — the per-player file against the grouped file, for both the effect and
its standard error. That second check is what catches this class of problem;
comparison against the manuscript alone does not.

---

## Analyses the manuscript reported with no generating code

Six reported analyses had no implementation in the original repository: the
tier-wise lag profiles, per-player reliability, the legal-move bins, the event
descriptives, the registered robustness specifications, and the complexity
prediction model. Their values existed only as constants inside the figure
code.

All six are now implemented as stages of `run.py` and write to
`data/derived/`. Because they are reconstructions rather than the original
code, they should be read as such.

---

## The titled tier reported a different definition

Table 2 of the earlier manuscript carried four rows: material loss and blunder,
each for the untitled tiers and for the titled tier. The two halves were not
computed the same way.

| Row | Definition |
|---|---|
| Material loss, untitled | net material change (the manuscript's Event B) |
| Material loss, titled | **static exchange evaluation** (the withdrawn criterion) |

When Event B was revised, the untitled tiers were re-extracted with the new
definition. The titled tier was not: the files behind its numbers carry
`see_loss` but no `mat_diff`, so the revision could not have been applied to
them. The reliability of .446 and the between-player SD of 0.071 came from the
withdrawn criterion.

Reducing Table 2 to two rows was therefore the right call, but not for the
reason first recorded here. This file previously said the titled tier could not
be compared because its sample is not distributed. That was incomplete: the
sample not being distributed is a publication decision, whereas the reason the
titled rows could not sit beside the untitled ones is that they answered a
different question.

The titled tier is being re-extracted under the net-loss definition. Until that
finishes, the manuscript reports four untitled tiers, and the titled tier
contributes only to Table 1 and the event counts — quantities that do not
depend on which definition of material loss is used.

Because the effect grew by a factor of about 3.5 in the untitled tiers when the
definition changed (−0.167 to −0.587), the titled-tier values are expected to
change substantially too. In particular, the observation that between-player
variance in the titled tier was about half that of the untitled tiers rests on
the old definition and may not survive.

---

## Event counts describe a larger scan than the effects

The analysis tables hold

```
                players     games       moves   material    blunders
untitled          1,878   112,153   3,390,227    473,603      59,605
titled              136    17,686     553,312     70,384      28,943
                  -----   -------   ---------    -------      ------
                  2,014   129,791   3,943,539    543,987      88,548
```

An earlier draft reported 887,287 material-loss events and 137,490 blunders
from 2,136 players. Events per game agreed — 4.21 against 4.19 — so the
shortfall was not a difference in how events are counted. Dividing the
reported count by that rate implies about 211,000 games, roughly 397 shards;
the tables were built from 208. The counts described the full January scan
while every effect came from the subset.

2,136 is also how many players were *sampled*. 122 of them have no game in the
extracted shards, so the analysis tables hold 2,014.

Because events per game agrees, the subset is representative and the effects
would not move appreciably if the remaining shards were added. What had to
change was the description: the paper now reports what was analysed.

### The check had the same hole, four ways

Filling these constants from the repository rather than the manuscript makes
`verify-paper` compare the repository with itself, and it duly reported zero
mismatches. The same shape appeared four times:

| | how the check failed |
|---|---|
| Title count | read one prepared table where the claim spans two |
| Transcription (twice) | constants filled from derived data, not the paper |
| Event counts | exempted in the transcription test |
| Revision figures | constants present, no comparison attached |

The last is the quietest. `N_EVENTS_MATERIAL_REGISTERED`, the two shares and
the verification subsample sat in `paper_check.py` for the whole project with
nothing reading them, so a stale value survived even though the data needed to
check it was on disk the entire time. All four are now compared.

For the record, the revision figures on the current tables: 975,871 events
under the registered criterion against 543,987 under net loss, 44% fewer; 54.1%
of the registered events involved no net change and 12.3% a net gain.


A check that draws both its sides from the same place cannot fail. Neither
can one that is never run, nor one whose reach stops short of where the claim
is repeated. And one that compares a part against a whole fails for no reason
at all: with `prepared_fm.parquet` absent, the sample-wide counts were read
from the untitled table alone and six comparisons came back short by exactly
the titled tier. Those checks now run only when both tables are present, and
say why when they do not.

That last case is what `check_external_docs.py` covers. `verify-paper` sees
only this repository; the cover letter and the preprint title page quote the
same figures and were checked by nothing. Its first pass matches the
manuscript's own wording, which a letter phrasing the fact differently slips
past — confirmed by feeding it a document saying "our study of 3.4 million
moves", which it passed. Its second pass therefore looks for the superseded
values themselves, whatever sentence they sit in.

-----   -------   ---------    -------      ------
                  2,014   129,791   3,943,539    543,987      88,548
```

Events per game agrees — 4.21 in the paper against 4.19 here — so the shortfall
is not a difference in how events are counted. Dividing the reported count by
that rate implies about 211,000 games, which is roughly 397 shards. The tables
were built from 208. The counts therefore describe the full January scan while
every effect comes from the subset.

2,136 is also how many players were *sampled*; 122 of them have no game in the
extracted shards.

`verify-paper` reads both prepared tables and checks the counts, the analysed
player total, and events per game, and says so when the sampled and analysed
totals differ.

---

## The title count no longer matches

The title says 3.4 million moves. That was the untitled tiers alone. With the
titled tier restored the analysis spans

```
untitled   3,390,227
titled       553,312
           ---------
total      3,943,539
```

`verify-paper` reads both prepared tables and reports the mismatch. It used to
read only the untitled one, so it agreed with the old title while the paper
had grown past it — the same shape of error as the transcription filled from
the repository: a check comparing something with itself.

The manuscript was corrected. This README was not, and kept the old count in
its title line and its citation block. `check_external_docs.py` exists to find
exactly that, and it had the old value listed as superseded, so it should have
found it. It did not, because it looked for `README.md` in the working
directory: the check is run from the folder holding the manuscript, that name
resolves to something else there, and a document that cannot be read is skipped
without comment. The one file the check was certain to be run alongside was the
one file it never opened.

The repository README is now located from the script’s own directory, and
failing to read it is reported rather than skipped. Both the stale and the
corrected README were run through the check to confirm it separates them.

The pattern is the one running through this file. A check that draws both sides
from the same place cannot fail; neither can a check that silently drops its
subject. The first is vacuous, the second is empty, and both report success.

---

## Results that did not reproduce

**The blunder effect changed sign.** Under the five-variable specification
blunders are followed by *slower* moves, not faster ones: +0.055 to +0.078
across the five tiers, significant in every one (t = 2.4 to 4.0), and +0.066,
+0.063, +0.078 across thresholds of 10, 20 and 30 percentage points. Earlier
specifications put these near zero and mostly not significant.

This resolves a tension the earlier numbers carried. The three-group split has
always shown blunder-only events at a positive value while the tier-wise
blunder effect read negative; the two now agree in sign.

The compositional argument survives on its level. Weighting +0.209 and −0.407
by their observed frequencies gives +0.055 against an observed +0.066.

**Its trend does not.** Raising the threshold should raise the share of
blunders that also cost material and pull the average down — +0.055 at a 25%
share, +0.018 at 31% — but the observed values are flat or rising. Rest the
argument on the level agreement alone.

**Between-player variance does not differ across tiers.** An earlier draft
argued that titled players vary about half as much as untitled ones, which
would say that skill makes people respond more alike. Measured on one ruler
the two are the same: 0.115 against 0.122 for material loss, 0.048 against
0.051 for blunders — ratios of 1.06 and 1.07.

The earlier figures (0.212 against 0.051) came from a specification that
confined the untitled tiers to their engine-evaluated subsample, which is
12.7% of games in the lowest tier and 93.9% in the titled tier. The claim was
withdrawn.

These cases are noted in the Method section of the manuscript.

---

## Data not distributed

The player sample is not included. Lichess account names are quasi-identifiers,
and the titled tier is small enough that individual players could be matched to
public profiles. The sampling rule is fully specified in `src/extract.py`;
running stage 1 against the same monthly archives reproduces the sample.

The `player` column of `per_player_t1.csv` holds salted hashes for the same
reason. The salt is not distributed, so the hashes cannot be resolved back to
accounts.

---

## Scope of this repository

This repository produces the statistics. It does not draw the figures. Every
value plotted in the manuscript is a row in one of the files in
`data/derived/`; the table in the README says which file backs which panel.

The flanker comparison (OpenNeuro ds004883) was not run here. Its values are in
`src/external_values.py`, marked `external`.
