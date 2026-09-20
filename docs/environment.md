# Environment notes

## Python

The pipeline is plain CPython with the packages in `requirements.txt`.
No compiled extensions of our own.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run everything from the repository root through `run.py`. There is no
install step.

### Pinning

`requirements.txt` gives ranges that the pipeline has been run against, not
the exact versions that produced the published numbers. Before archiving,
capture the real environment:

```bash
pip freeze > requirements.lock
```

and commit that alongside. Parquet files record the writer version in their
metadata; the study's own artefacts were written by pyarrow 25.0.1 under
pandas 3.0.2.

`chess` matters more than the others. `chess.between()` and the `SquareSet`
return types have changed across 1.x releases and `src/see.py` depends on
them. The old pin `python-chess==1.999` is a shim package that installs
`chess` without constraining its version.

## Disk and memory

| Stage | Disk | Notes |
|---|---|---|
| Shard download | ~1 GB at a time | each shard is deleted right after it is scanned |
| `plies/` | ~2–4 GB | kept; `run.py prep` reads all of it |
| `prepared.parquet` | ~300 MB | single file, loaded fully into memory |

`run.py prep` and `run.py se` hold the analysis table in memory. Budget
about 8 GB. If that is tight, `run.py lag-split` takes a `PLT` environment
variable that caps players per tier.

Set `CHESS_WORK` to move intermediate files off the repository volume:

```bash
export CHESS_WORK=/mnt/scratch/chess
```

Everything except `data/derived/` goes there.

## Runtime

Roughly 1 hour 45 minutes end to end on one machine, dominated by stage 2
(~85 min over all 397 shards of January 2024).
The scan and extract stages are shard-parallel — the arguments are a shard
range, so the work splits across sessions and resumes: existing outputs are
skipped.

## Network

The scan and extract stages download from the Lichess dataset mirror on
Hugging Face via `curl`. `curl` must be on PATH. Shards that fail to download are reported and
skipped, not retried — rerun the same range to fill gaps.

## Stockfish

Only the complexity validation needs it.

```bash
apt-get install stockfish          # Debian / Ubuntu
brew install stockfish             # macOS
```

The engine is found on PATH as `stockfish`. Override with:

```bash
export CHESS_STOCKFISH=/path/to/stockfish
```

Depth is 18 in `src/complexity.py`; `run.py cx` uses 12 by default and takes
`--depth` (or `CX_DEPTH`). Report whichever you actually ran.

## Platform

Developed on Linux. The tests and the full pipeline have also been run on
Windows. Two things to watch there:

- Console encoding. If you see `UnicodeEncodeError` on the Korean output, set
  `PYTHONUTF8=1`.
- `curl` ships with Windows 10+ but shells may alias it; check `curl --version`
  returns real curl, not a PowerShell alias.
