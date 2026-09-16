#!/usr/bin/env python
"""
Single entry point for the pipeline.

    python run.py <stage> [arguments]
    python run.py --help              list every stage
    python run.py <stage> --help      arguments for that stage, and a note on
                                      what to check in it

Full order (approximate running times in brackets)

    scan            stage 1, account metadata over shards 0-421   (~15 min)
    scan-titled     titled accounts, 2024-01 .. 2024-06      <- slowest stage
    sample          fix the player sample -> out/ (not distributed)  (~1 min)
    extract         stage 2, board replay and ply features        (~45 min)
    extract-titled  the same for the titled tier                  (~20 min)
    prep            analysis table (net_mat, z, z_pre3)            (~5 min)
    se              clustered SEs -> data/derived/se_*.csv         (~10 min)
    lag-all         every lag profile -> lag_profiles.csv          (~20 min)
    reliability     per-player reliability (Table 2)               (~15 min)
    bins            effect within legal-move-count bins            (~10 min)
    describe        event descriptives                             (~2 min)
    robustness      the six registered specifications              (~30 min)
    verify-paper    manuscript values vs data/derived/    (instant) <- run
                                                          this before submitting

Supporting stages: prelim, lag, lag-split, ps6, did, mixed,
                   cx-fens, cx, cx-model, aggregate, anonymize
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE
#   - The whole pipeline is in this one file; stages correspond 1:1 to the
#     commands in the README.
#   - Each stage skips output that already exists, so an interrupted run can
#     be resumed.
#   - scan and extract take a shard range and can be split across sessions.
#   - Every path is in src/config.py. Do not write path strings here.
# ───────────────────────────────────────────────────────────

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src import config as C
from src.analysis import lagwise, summarize, per_player_at, cluster_stats
from src.analysis import build_with_pretrend, build_pairs, fit_mixed
from src.extract import (download, shard_url, scan_shard, scan_titled_shard,
                         extract_shard, sample_fens, build_all)
from src.prepare import (add_pre_speed, net_material_change, events_A, events_B,
                         events_B_see, mask_A, mask_B, calipers)

SORT_KEY = ["player", "game_id", "ply"]


# ══════════════════════════════════════════════════════════════
# 공통
# ══════════════════════════════════════════════════════════════

def _parallel(fn, jobs, workers, label):
    t = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        r = list(ex.map(fn, jobs))
    counts = {k: r.count(k) for k in ("ok", "skip", "fail", "excl") if r.count(k)}
    print(f"{label}: {len(jobs)}개 / {time.time()-t:.0f}s  "
          + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return r


def _need(path, hint):
    if not os.path.exists(path):
        sys.exit(f"필요한 파일이 없다: {path}\n{hint}")


def _calipers_from_main(with_prespeed=True, with_wp=False):
    """
    캘리퍼 기준 SD 는 전 층 공통(하위 4개 층)으로 잡는다.

    ★ SD 는 **z_pre3 결측을 버린 뒤** 계산한다. 분석은 전부
      `.dropna(subset=[PRESPEED_VAR])` 한 프레임에서 도는데, 예전에는 이
      함수만 전체 프레임에서 SD 를 내고 있었다. z_pre3 결측이 9.9% 이고
      그 행들은 대국 시작부(ply 가 작고 시계가 많은 구간)에 몰려 있어,
      두 기준의 SD 가 최대 4% 어긋났다.

          변수          결측 제거(논문)   전체 프레임(옛 코드)
          wp_before        0.055514          0.053049
          clk_before      30.935437         31.491718
          n_legal          2.545109          2.483746
          ply              5.958813          6.157324

      공개된 CSV 는 결측 제거 기준으로 나왔다. tests/test_prep.py 가
      이 순서를 고정한다.
    """
    base_vars = C.MATCH_VARS_WP if with_wp else C.MATCH_VARS
    cols = list(base_vars) + ([C.PRESPEED_VAR] if with_prespeed else [])
    base = pd.read_parquet(C.PREPARED, columns=cols)
    if with_prespeed:
        base = base.dropna(subset=[C.PRESPEED_VAR])
    sd = {v: base[v].std() for v in base.columns}
    del base
    cal = {v: C.CALIPER_SD * sd[v] for v in base_vars}
    if with_prespeed:
        cal[C.PRESPEED_VAR] = C.PRESPEED_CALIPER_SD * sd[C.PRESPEED_VAR]
    return cal, sd


def _events(d, which, use_see=False):
    if which.endswith("A"):
        return events_A(d), "blunder"
    if use_see:
        return events_B_see(d), "material_see"
    return events_B(d), "material"


def _slice(ev, a, b):
    ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
    return ev.iloc[a:b] if b is not None else ev.iloc[a:]


# ══════════════════════════════════════════════════════════════
# Stage 1 — 계정 메타데이터
# ══════════════════════════════════════════════════════════════

def cmd_scan(args):
    """0~421 샤드의 계정 메타데이터를 집계한다. 대국 내용은 읽지 않는다."""
    C.ensure_dirs(C.STAGE1, C.WORK)

    def work(i):
        p = f"{C.STAGE1}/s{i:05d}.parquet"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return "skip"
        tmp = os.path.join(C.WORK, f"_a{i}.parquet")
        if not download(shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i,
                                  C.MAIN_N_SHARDS), tmp):
            return "fail"
        try:
            scan_shard(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return "ok"

    todo = [i for i in range(args.start, args.end) if i not in C.PRELIM_SHARDS]
    _parallel(work, todo, 6, "scan")
    print("다음: python run.py scan-titled <월> / python run.py sample")


def cmd_scan_titled(args):
    """
    타이틀 계정만 집계한다. 월을 생략하면 1~6월을 모두 돈다.

    타이틀 층이 1개월이면 ≥30판이 26명뿐이라 층이 성립하지 않는다.
    논문 Table 1 의 136명은 6개월 집계 결과다. 이 단계는 타이틀 계정을
    찾기 위해 각 달의 전 샤드를 훑어야 하므로 전체에서 가장 오래 걸린다.
    이미 받은 샤드는 건너뛰므로 나눠 돌려도 된다.
    """
    C.ensure_dirs(C.STAGE1_FM, C.WORK)
    months = sorted(C.TITLED_MONTHS) if args.month is None else [args.month]
    for m in months:
        n = C.TITLED_MONTHS[m]
        end = args.end if args.end is not None else n
        jobs = [(m, i, n) for i in range(args.start, min(end, n))
                if not (m == 1 and i in C.PRELIM_SHARDS)]
        _parallel(scan_titled_shard, jobs, 5, f"scan-titled m{m:02d}")
    print("다음: python run.py sample")


def cmd_sample(args):
    """1단계 산출물을 합쳐 최종 표본 명단을 만든다."""
    build_all()
    if os.path.exists(C.SAMPLE):
        print()
        _check_sample(pd.read_parquet(C.SAMPLE))


# ══════════════════════════════════════════════════════════════
# Stage 2 — 대국 재현
# ══════════════════════════════════════════════════════════════

def _check_sample(samp):
    """
    표본 명단이 논문 표본인지 확인한다.

    타이틀 층을 1개월로만 집계하면 26명, 6개월이면 136명이 된다.
    26명짜리 파일로 stage 2 를 돌리면 논문과 다른 표본이 나오는데,
    실제로 저장소에 그 파일이 들어 있었다. 조용히 지나가지 않게 막는다.
    """
    from src.paper_check import TABLE1, N_PLAYERS
    n_fm = int((samp.tier == C.TITLE_TIER).sum())
    want_fm = TABLE1[C.TITLE_TIER][3]
    if len(samp) != N_PLAYERS or n_fm != want_fm:
        print(f"  경고: 표본이 논문과 다르다 — 총 {len(samp):,}명 "
              f"(논문 {N_PLAYERS:,}), 타이틀 {n_fm}명 (논문 {want_fm}).", flush=True)
        if n_fm and n_fm < want_fm / 2:
            print(f"         타이틀 층이 1개월 집계로 보인다. "
                  f"run.py scan-titled 를 1~6월 모두 돌린 뒤 "
                  f"run.py sample 을 다시 돌릴 것.", flush=True)
        print("         그대로 진행하려면 --force 를 붙인다.", flush=True)
        return False
    return True


def _extract_common(sample_path, out_dir, tier_filter, url_fn, jobs, label,
                    force=False):
    _need(sample_path, "먼저 python run.py scan / sample 을 돌릴 것.")
    C.ensure_dirs(out_dir, C.WORK)
    samp = pd.read_parquet(sample_path)
    if not _check_sample(samp) and not force:
        sys.exit(1)
    keep = set(tier_filter(samp).player)
    tier_map = dict(zip(samp.player, samp.tier))
    print(f"대상 플레이어 {len(keep):,}명", flush=True)

    def work(job):
        out, url, tag = url_fn(job)
        if tag == "excl":
            return "excl"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return "skip"
        tmp = os.path.join(C.WORK, f"_x{tag}.parquet")
        if not download(url, tmp):
            return "fail"
        try:
            extract_shard(tmp, out, keep_players=keep, eval_only=False,
                          do_see=True, tier_map=tier_map)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return "ok"

    _parallel(work, jobs, 5, label)


def cmd_extract(args):
    """하위 4개 층. do_see=True 라야 mat_diff 가 채워진다."""
    def url_fn(i):
        return (f"{C.STAGE2}/s{i:05d}.parquet",
                shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i, C.MAIN_N_SHARDS),
                f"b{i}")
    jobs = [i for i in range(args.start, args.end) if i not in C.PRELIM_SHARDS]
    _extract_common(C.SAMPLE, C.STAGE2, lambda s: s[s.tier != C.TITLE_TIER],
                    url_fn, jobs, "extract", force=args.force)


def cmd_extract_titled(args):
    """타이틀 층. 1~6월 전체가 기본이다."""
    def url_fn(job):
        m, i = job
        if m == 1 and i in C.PRELIM_SHARDS:
            return None, None, "excl"
        return (f"{C.STAGE2_FM}/m{m:02d}_s{i:05d}.parquet",
                C.ARCHIVE_2024_URL.format(m=m, i=i, n=C.TITLED_MONTHS[m]),
                f"f{m}_{i}")

    if args.month is None:
        jobs = [(m, i) for m in sorted(C.TITLED_MONTHS)
                for i in range(C.TITLED_MONTHS[m])]
    else:
        end = args.end if args.end is not None else C.TITLED_MONTHS[args.month]
        jobs = [(args.month, i)
                for i in range(args.start, min(end, C.TITLED_MONTHS[args.month]))]
    _extract_common(C.SAMPLE_FM, C.STAGE2_FM, lambda s: s, url_fn, jobs,
                    "extract-titled", force=True)   # FM 파일은 층이 하나뿐


def cmd_prelim(args):
    """예비 분석용 추출. 본분석 표본과 겹치지 않게 0~23 샤드만 쓴다."""
    C.ensure_dirs(C.PRELIM, C.WORK)

    def work(i):
        p = f"{C.PRELIM}/s{i:05d}.parquet"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return "skip"
        tmp = os.path.join(C.WORK, f"_p{i}.parquet")
        if not download(shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i,
                                  C.MAIN_N_SHARDS), tmp):
            return "fail"
        try:
            extract_shard(tmp, p, keep_players=None, eval_only=False,
                          do_see=False)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return "ok"

    _parallel(work, list(range(args.start, args.end)), 4, "prelim")


# ══════════════════════════════════════════════════════════════
# Stage 3 — 분석 테이블
# ══════════════════════════════════════════════════════════════

KEEP_COLS = ["game_id", "player", "color", "tier", "ply", "move_time",
             "clk_before", "clk_after", "wp_before", "wp_after", "wp_delta",
             "see_loss", "n_legal", "mat_diff", "max_see_mine", "n_checks",
             "nag", "is_capture", "ply_total"]
KEY = ["game_id", "player", "ply"]


def cmd_prep(args):
    """
    plies → 분석 테이블.

    순서가 중요하다:
      1. net_mat   ← 측정 하한 **전**. mat_diff 는 시계와 무관한 국면
                     정보라, 하한으로 버려질 수를 먼저 지우면 ply+2 짝을
                     불필요하게 잃는다.
      2. 측정 하한 (move_time >= TAU)
      3. z 표준화 (플레이어 내)
      4. z_pre3    ← z 가 정의된 뒤라야 계산할 수 있다
    """
    src_dir = C.STAGE2_FM if args.fm else C.STAGE2
    out = C.PREPARED_FM if args.fm else C.PREPARED
    C.ensure_dirs(out)

    files = sorted(glob.glob(f"{src_dir}/*.parquet"))
    if not files:
        sys.exit(f"plies 가 없다: {src_dir}\n"
                 f"먼저 python run.py extract 를 돌릴 것.")
    print(f"{len(files)}개 샤드 읽는 중...", flush=True)
    parts, blank = [], 0
    for f in files:
        dd = pd.read_parquet(f)
        if len(dd) == 0:
            # A shard with no wanted game writes an empty frame, and an empty
            # frame's columns are all dtype object. Concatenating one of those
            # turns move_time into object for the whole table, and np.log then
            # fails with "loop of ufunc does not support argument 0 of type
            # float". Dropping them changes nothing and avoids the cast.
            #
            # The four untitled tiers never hit this -- every shard contained
            # one of the 1,878 players. The titled tier is sparse enough that
            # 119 of its 2,355 shards are empty.
            blank += 1
            continue
        parts.append(dd[[c for c in KEEP_COLS if c in dd.columns]])
    if not parts:
        sys.exit(f"샤드 {len(files)}개가 전부 비어 있다: {src_dir}")
    if blank:
        print(f"  빈 샤드 {blank}개 건너뜀", flush=True)
    d = pd.concat(parts, ignore_index=True)
    print(f"  {len(d):,}행 / 플레이어 {d.player.nunique():,}명 / "
          f"대국 {d.game_id.nunique():,}판", flush=True)

    t = time.time()
    dup = d.duplicated(subset=KEY).sum()
    if dup:
        print(f"  경고: 중복 {dup:,}행 제거", flush=True)
        d = d.drop_duplicates(subset=KEY)
    d = d.sort_values(KEY).reset_index(drop=True)

    if "mat_diff" not in d.columns or d.mat_diff.isna().all():
        sys.exit("mat_diff 가 비어 있다. extract 를 do_see=True 로 다시 돌릴 것.")
    d = net_material_change(d)
    n_ev = int((d.net_mat < 0).sum())
    print(f"  net_mat — 손실 사건 {n_ev:,}건 "
          f"({n_ev / max(d.game_id.nunique(), 1):.2f}/게임)", flush=True)

    before = len(d)
    d = d[d.move_time >= C.TAU].copy()
    print(f"  측정 하한 {C.TAU}초: {before:,} → {len(d):,}행 "
          f"({100 * (before - len(d)) / before:.1f}% 제거)", flush=True)

    d["y"] = np.log(d.move_time + C.LOG_OFFSET)
    g = d.groupby("player")["y"]
    d["z"] = (d.y - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    d = d.dropna(subset=["z"]).reset_index(drop=True)

    d = add_pre_speed(d, k=C.PRESPEED_K)
    print(f"  z_pre3 결측 {d.z_pre3.isna().sum():,}행 "
          f"({100 * d.z_pre3.isna().mean():.1f}%) — 대국 초반은 정의 불가",
          flush=True)

    d.to_parquet(out, index=False)
    print(f"\n{'FM+' if args.fm else '하위 4개 층'}: {len(d):,}행 → {out}  "
          f"[{time.time()-t:.0f}s]")
    print("다음: python run.py se split")


# ══════════════════════════════════════════════════════════════
# 주 분석 — 표준오차
# ══════════════════════════════════════════════════════════════

SE_LAGS = range(-3, 4)


def cmd_se(args):
    """
    논문에 보고되는 모든 SE 가 여기서 나온다.

    출력 열:
      e{k}    사건 가중 평균 (논문 보고값)
      se{k}   플레이어 클러스터 SE
      epw{k}  플레이어 동일가중 평균
      np{k}   클러스터(플레이어) 수

    e 와 epw 가 크게 다르면 플레이어별 사건 수 불균형이 크다는 뜻이다.
    """
    _need(C.PREPARED, "먼저 python run.py prep 을 돌릴 것.")
    d = pd.read_parquet(C.PREPARED).dropna(subset=[C.PRESPEED_VAR])

    # The titled tier joins every mode, not just the tier breakdown.
    #
    # Adding it cannot disturb the four untitled tiers: lagwise draws controls
    # only from the same player (by_player[r.player]) and discards players
    # with no events. Caliper widths still come from the untitled sample, so
    # every tier is matched on one ruler.
    #
    # --tiers four reproduces the untitled-only version for comparison.
    if args.tiers == "five" and os.path.exists(C.PREPARED_FM):
        fm = pd.read_parquet(C.PREPARED_FM).dropna(subset=[C.PRESPEED_VAR])
        keep = [c for c in d.columns if c in fm.columns]
        d = pd.concat([d[keep], fm[keep]], ignore_index=True)
        print(f"  타이틀 층 포함: +{len(fm):,}행 "
              f"({fm.player.nunique()}명)", flush=True)
    elif args.tiers == "four":
        print("  하위 네 층만 (--tiers four)", flush=True)

    cal, _ = _calipers_from_main()
    hasA, hasB = mask_A(d), mask_B(d)

    def run(lab, ev, want_per_player=False):
        ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > C.MAX_EVENTS_SE:
            ev = (ev.sample(C.MAX_EVENTS_SE, random_state=C.EVENT_SAMPLE_SEED)
                    .sort_values(SORT_KEY).reset_index(drop=True))
        if len(ev) < C.MIN_EVENTS:
            return None, None
        p = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        out = {"label": lab, "n": len(ev)}
        for k in SE_LAGS:
            c = f"lag{k:+d}"
            st = cluster_stats(p[["player", c]].dropna(), c)
            if st is None:
                continue
            out[f"e{k}"], out[f"se{k}"] = st["effect"], st["se"]
            out[f"epw{k}"], out[f"np{k}"] = st["effect_pw"], st["n_players"]
        per = per_player_at(p, lag=1, label=lab) if want_per_player else None
        return out, per

    rows, pers = [], []
    if args.mode == "tier":
        for t in sorted(d.tier.unique()):
            dd = d[d.tier == t]
            for tag, m in [(f"B_{t}", mask_B(dd)), (f"A_{t}", mask_A(dd))]:
                r, _ = run(tag, dd[m])
                if r:
                    rows.append(r)
    elif args.mode == "split":
        for nm, m in [("blunder_only", hasA & ~hasB),
                      ("blunder_loss", hasA & hasB),
                      ("loss_only", hasB & ~hasA)]:
            r, per = run(nm, d[m], want_per_player=True)
            if r:
                rows.append(r)
            if per is not None:
                pers.append(per)
    else:  # dose — 방향 × 크기
        # 라벨은 self_N(플레이어가 잃음) / opp_N(상대가 잃음).
        # direction·size 열을 함께 내 그림과 대조가 쉽도록 한다.
        for v in [1, 3, 5, 9]:
            for tag, direction, m in [(f"self_{v}", "self", d.net_mat == -v),
                                      (f"opp_{v}", "opp", d.net_mat == v)]:
                r, _ = run(tag, d[m])
                if r:
                    r["direction"], r["size"] = direction, v
                    rows.append(r)

    C.ensure_dirs(C.DERIVED)
    x = pd.DataFrame(rows)
    lead = [c for c in ["label", "direction", "size", "n"] if c in x.columns]
    x = x[lead + [c for c in x.columns if c not in lead]]
    path = os.path.join(C.DERIVED, f"se_{args.mode}.csv")
    x.to_csv(path, index=False)
    print(f"→ {path}")
    if pers:
        pp = os.path.join(C.DERIVED, "per_player_t1.csv")
        pd.concat(pers, ignore_index=True).to_csv(pp, index=False)
        print(f"→ {pp}")
    for r in x.itertuples():
        e1, se1 = getattr(r, "e1"), getattr(r, "se1")
        print("{:>18} n={:>6,}  t+1 {:+.4f} +/- {:.4f}  t={:>7.1f}  "
              "(플레이어가중 {:+.4f}, 클러스터 {:,})".format(
                  r.label, r.n, e1, se1, e1 / se1,
                  getattr(r, "epw1"), int(getattr(r, "np1"))))


# ══════════════════════════════════════════════════════════════
# lag 분해
# ══════════════════════════════════════════════════════════════

SPECS = ("none", "nlegal", "nlegal_cx", "zpre")
LAG_COLS = ["game_id", "player", "tier", "ply", "z", "wp_before", "clk_before",
            "n_legal", "see_loss", "wp_delta", "net_mat", "z_pre3", "cx_pred"]


def _load_for_lag(is_fm, spec, need_cx=False):
    import pyarrow.parquet as pq
    src = C.prepared_path(is_fm)
    _need(src, "먼저 python run.py prep 을 돌릴 것.")
    avail = set(pq.ParquetFile(src).schema.names)
    d = pd.read_parquet(src, columns=[c for c in LAG_COLS if c in avail])
    if spec == "zpre":
        d = d.dropna(subset=[C.PRESPEED_VAR])
    if need_cx and "cx_pred" not in d.columns:
        sys.exit("cx_pred 열이 없다. python run.py cx-model 을 먼저 돌릴 것.\n"
                 "(없이 돌리면 사양이 조용히 nlegal 로 바뀐다)")
    return d


def cmd_lag(args):
    """층 묶음별 lag 프로파일. 그림 1의 (a)(b) 패널."""
    if args.spec not in SPECS:
        sys.exit(f"spec 은 {SPECS} 중 하나여야 한다 (받은 값: {args.spec!r})")
    is_fm = args.which.startswith("fm")
    d = _load_for_lag(is_fm, args.spec, need_cx=(args.spec == "nlegal_cx"))

    _, sd = _calipers_from_main()
    if "cx_pred" in d.columns:
        sd["cx_pred"] = d.cx_pred.std()
    # The core set follows config.MATCH_VARS minus n_legal, which each spec
    # adds for itself. It used to name wp_before, clk_before and ply directly,
    # which broke the moment the main specification dropped win probability.
    core = {v: C.CALIPER_SD * sd[v]
            for v in C.MATCH_VARS if v != "n_legal"}
    if args.spec == "none":
        cal = {}
    elif args.spec == "nlegal":
        cal = {**core, "n_legal": C.CALIPER_SD * sd["n_legal"]}
    elif args.spec == "nlegal_cx":
        cal = {**core, "n_legal": C.CALIPER_SD * sd["n_legal"],
               "cx_pred": C.CALIPER_SD * sd["cx_pred"]}
    else:
        cal = {**core, "n_legal": C.CALIPER_SD * sd["n_legal"],
               C.PRESPEED_VAR: C.PRESPEED_CALIPER_SD * sd[C.PRESPEED_VAR]}

    ev_all, et = _events(d, args.which, args.see)

    # --by-tier 면 층별로 따로 돈다. 논문 Fig 1a/1b 는 다섯 층을 각각
    # 그리는데, 예전에는 하위 네 층을 "lower" 하나로 묶어서만 돌렸다.
    if args.by_tier:
        tiers = [t for t in sorted(ev_all.tier.dropna().unique())]
    else:
        tiers = [None]

    C.ensure_dirs(C.WORK)
    parts = []
    for tier in tiers:
        ev = ev_all if tier is None else ev_all[ev_all.tier == tier]
        ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > args.nmax:
            ev = (ev.sample(args.nmax, random_state=C.EVENT_SAMPLE_SEED)
                    .sort_values(SORT_KEY).reset_index(drop=True))
        if len(ev) < C.MIN_EVENTS:
            print(f"  건너뜀 {tier}: 사건 {len(ev)}건")
            continue
        dd = d[d.player.isin(set(ev.player))]
        tier_label = tier if tier is not None else (
            "fm_plus" if is_fm else "lower")
        print(f"  {tier_label}: 데이터 {len(dd):,}행 / 사건 {len(ev):,}",
              flush=True)
        t = time.time()
        part = summarize(lagwise(dd, ev, cal, seed=C.MATCH_SEED),
                         et, tier_label, args.spec)
        if part.empty:
            # summarize 는 lag 마다 관측이 min_n 미만이면 그 행을 버린다.
            # 전부 버려지면 빈 프레임이 되므로 concat 대상에서 뺀다.
            print(f"    관측 부족 — 건너뜀", flush=True)
            continue
        parts.append(part)
        print(f"    [{time.time()-t:.0f}s]", flush=True)

    if not parts:
        print("산출 없음 (모든 층에서 관측 부족)")
        return
    res = pd.concat(parts, ignore_index=True)
    out = (f"{C.WORK}/lag_{args.which}_{args.spec}"
           f"{'_bytier' if args.by_tier else ''}"
           f"{'_see' if args.see else ''}.parquet")
    res.to_parquet(out, index=False)
    print(f"{et}/{args.spec} → {out}")
    cols = ["tier", "lag", "effect", "se", "t", "n_events", "n_players"]
    print(res[cols].to_string(index=False))


def cmd_lag_all(args):
    """사건 2 × 층 2 × 사양 4 를 전부 돌리고 CSV 로 합친다."""
    import pyarrow.parquet as pq
    failed = []
    for which in ["A", "B", "fmA", "fmB"]:
        src = C.prepared_path(which.startswith("fm"))
        if not os.path.exists(src):
            print(f"건너뜀 {which}: {src} 없음")
            continue
        has_cx = "cx_pred" in set(pq.ParquetFile(src).schema.names)
        for spec in SPECS:
            if spec == "nlegal_cx" and not has_cx:
                print(f"건너뜀 {which}/{spec}: cx_pred 없음 (run.py cx-model 필요)")
                continue
            print(f"\n--- {which} / {spec}", flush=True)
            try:
                cmd_lag(argparse.Namespace(which=which, spec=spec,
                                           nmax=C.MAX_EVENTS_LAG, see=False,
                                           by_tier=False))
            except SystemExit as e:
                if e.code:
                    failed.append(f"{which}/{spec}")
    # 층별 프로파일 (Fig 1a/1b) — 주 사양으로 한 번씩
    for which in ["A", "B"]:
        if not os.path.exists(C.prepared_path(False)):
            break
        print(f"\n--- {which} / zpre / 층별", flush=True)
        try:
            cmd_lag(argparse.Namespace(which=which, spec="zpre",
                                       nmax=C.MAX_EVENTS_LAG, see=False,
                                       by_tier=True))
        except SystemExit as e:
            if e.code:
                failed.append(f"{which}/zpre/by-tier")

    cmd_aggregate(args)
    if failed:
        print(f"\n실패한 조합: {', '.join(failed)}")


def cmd_lag_split(args):
    """
    A만 / B만 / A∩B 세 집단으로 나눠 lag 분해. 그림 1의 (c) 패널.

    ★ 사양 주의 — 예전 판은 주 결과(run.py se split)와 **다른 사양**이었다.
      캘리퍼가 4변수(z_pre3 없음)였고 층당 110명으로 플레이어를 잘랐다.
      같은 세 집단인데 두 산출물이 서로 비교 불가였고, 논문의 효과(+0.123
      등)와 SE(0.026 등)가 다른 실행에서 온 원인이 이것이다.

      기본값을 주 결과와 동일한 6변수(플레이어 + 4 + z_pre3)로 맞췄다.
      옛 사양을 재현하려면 --no-prespeed 를 쓴다.
    """
    is_fm = args.which == "fm"
    spec = "nlegal" if args.no_prespeed else "zpre"
    d = _load_for_lag(is_fm, spec)
    cal, _ = _calipers_from_main(with_prespeed=not args.no_prespeed)

    hasA, hasB = mask_A(d), mask_B(d)
    grp = {"Aonly": (hasA & ~hasB, "blunder_only"),
           "Bonly": (~hasA & hasB, "material_only"),
           "AB": (hasA & hasB, "both")}
    if args.grp not in grp:
        sys.exit("grp 은 Aonly / Bonly / AB 중 하나여야 한다")
    m, et = grp[args.grp]
    ev = d[m].sort_values(SORT_KEY).reset_index(drop=True)

    # 플레이어 표집으로 메모리 제한
    rng = np.random.default_rng(C.EVENT_SAMPLE_SEED)
    plt_n = int(os.environ.get(
        "PLT", 136 if is_fm else C.PLAYERS_PER_TIER_LAGSPLIT))
    keep = set()
    for _, g in ev.groupby("tier"):
        u = g.player.unique()
        keep |= set(rng.choice(u, min(plt_n, len(u)), replace=False))
    ev = ev[ev.player.isin(keep)].reset_index(drop=True)
    d = d[d.player.isin(keep)]

    C.ensure_dirs(C.WORK)
    out = (f"{C.WORK}/lagsplit_{args.which}_{args.grp}"
           f"{'_nopre' if args.no_prespeed else ''}.parquet")
    if os.path.exists(out):
        print("skip")
        return
    t = time.time()
    s = summarize(lagwise(d, ev, cal, seed=C.MATCH_SEED), et,
                  "fm_plus" if is_fm else "lower", spec)
    s.to_parquet(out, index=False)
    print(f"{et}/{args.which} [{time.time()-t:.0f}s] n={len(ev):,} → {out}")
    print(" ".join(f"{r.lag:+d}:{r.effect:+.3f}({r.t:.1f})"
                   for r in s.itertuples()))


# ══════════════════════════════════════════════════════════════
# 보조 분석
# ══════════════════════════════════════════════════════════════

def cmd_ps6(args):
    """주 사양(0.6 SD 사전속도 매칭)으로 사건별 pre/at/post 를 낸다."""
    is_fm = args.which.startswith("fm")
    src = C.prepared_path(is_fm)
    _need(src, "먼저 python run.py prep 을 돌릴 것.")
    d = pd.read_parquet(src).dropna(subset=[C.PRESPEED_VAR])
    cal, _ = _calipers_from_main()
    ev, _et = _events(d, args.which)
    ev = _slice(ev, args.start, args.end)
    if len(ev) == 0:
        print("empty")
        return

    C.ensure_dirs(C.WORK)
    out = f"{C.WORK}/ps6_{args.which}_{args.start:07d}.parquet"
    if os.path.exists(out):
        print("skip")
        return
    t = time.time()
    e = build_with_pretrend(d, ev, cal, seed=C.MATCH_SEED).reset_index(drop=True)
    e["size"] = ev["net_mat"].values
    e["wp_delta"] = ev["wp_delta"].values
    e.to_parquet(out, index=False)
    ok = e.dropna(subset=["effect"])
    print("{} {:,}건 {:.0f}s  매칭 {:.0f}%  pre{:+.4f}  eff{:+.4f}".format(
        args.which, len(ev), time.time() - t,
        100 * len(ok) / len(e), ok.pretrend.mean(), ok.effect.mean()))


def cmd_did(args):
    """이중차분 보정. 캘리퍼에 z_pre3 를 넣지 않는 것이 이 사양의 요점이다."""
    is_fm = args.which.startswith("fm")
    src = C.prepared_path(is_fm)
    _need(src, "먼저 python run.py prep 을 돌릴 것.")
    cols = [c for c in LAG_COLS if c != "cx_pred"]
    d = pd.read_parquet(src, columns=cols)
    if is_fm:
        main = pd.read_parquet(C.PREPARED, columns=list(C.MATCH_VARS))
        cal = calipers(main)
        del main
    else:
        cal = calipers(d)

    ev, _et = _events(d, args.which)
    ev = _slice(ev, args.start, args.end)
    if len(ev) == 0:
        print("empty")
        return

    C.ensure_dirs(C.WORK)
    out = f"{C.WORK}/did_{args.which}_{args.start:07d}.parquet"
    if os.path.exists(out):
        print("skip")
        return
    t = time.time()
    e = build_with_pretrend(d, ev, cal, seed=C.MATCH_SEED)
    e.to_parquet(out, index=False)
    ok = e.dropna(subset=["effect", "pretrend"])
    print("{} {:,}건 {:.0f}s  pre{:+.4f} post{:+.4f} did{:+.4f}".format(
        args.which, len(ev), time.time() - t,
        ok.pretrend.mean(), ok.effect.mean(), ok.did.mean()))


def cmd_mixed(args):
    """
    Preregistered mixed-effects model.

    Its job is to show that the effect does not depend on the matching: the
    same quantity, estimated with player random effects instead. So it uses
    the same dependent variable as everything else -- the move at t+1 -- and
    the same five matching variables as controls.

    Three rows:
      mixed model          random intercept and slope by player
      paired difference    the same sample, event minus its matched controls
      with wp (robustness)  the registered specification, on the engine-
                            evaluated subsample where it can be computed

    The earlier version reported "with wp" and "without wp" as two headline
    numbers. That split existed because win probability sat in the caliper;
    once it moved to robustness the split had nothing left to mean.

    ★ Always report which path the fit took. A model that falls back has not
      converged, and the number is a paired difference wearing its name.
    """
    is_fm = args.which.startswith("fm")
    src = C.prepared_path(is_fm)
    _need(src, "먼저 python run.py prep 을 돌릴 것.")
    d = pd.read_parquet(src).dropna(subset=[C.PRESPEED_VAR])
    ev_all, _et = _events(d, args.which)
    ev_all = ev_all.sort_values(SORT_KEY).reset_index(drop=True)
    print(f"사건 {len(ev_all):,}건", flush=True)

    activity = (pd.read_parquet(C.SAMPLE)[["player", "n_games"]]
                if os.path.exists(C.SAMPLE) else None)

    rows = []

    def one(tag, frame, events, cal, fit_model=True, with_wp=False):
        t = time.time()
        long_df = build_pairs(frame, events, cal, seed=C.EVENT_SAMPLE_SEED,
                              max_events=args.max_events)
        n_pairs = len(long_df) // 2
        if n_pairs == 0:
            print(f"  건너뜀 {tag}: 쌍이 없다", flush=True)
            return
        w = long_df.pivot_table(index="eid", columns="event_condition",
                                values="z_dv").dropna()
        diff = w[1] - w[0]
        pl = long_df.drop_duplicates("eid").set_index("eid").player
        per = diff.groupby(pl).mean()
        rows.append({"spec": f"paired difference ({tag})",
                     "coefficient": float(diff.mean()),
                     "se": float(per.sem()), "convergence": "n/a",
                     "n_obs": len(diff), "n_players": int(per.size)})
        print(f"  짝차이 ({tag}) {diff.mean():+.4f} +/- {per.sem():.4f}  "
              f"쌍 {len(diff):,} / 플레이어 {per.size:,}  "
              f"[{time.time()-t:.0f}s]", flush=True)

        if not fit_model:
            return
        out, _fit = fit_mixed(long_df, activity=activity,
                              with_wp=with_wp)
        rows.append({"spec": f"mixed model ({tag})",
                     "coefficient": float(out.get("coef", float("nan"))),
                     "se": float(out.get("se", float("nan"))),
                     "convergence": out.get("path", "random slope")
                     if out.get("converged") else
                     f"NOT CONVERGED -- {out.get('path')}",
                     "n_obs": int(out.get("n_obs", 0)),
                     "n_players": int(out.get("n_players", 0))})
        print(f"  혼합모형 ({tag}) {out.get('coef', float('nan')):+.4f} "
              f"+/- {out.get('se', float('nan')):.4f}  "
              f"경로 {out.get('path')}  수렴 {out.get('converged')}",
              flush=True)

    cal5, _ = _calipers_from_main()
    one("main", d, ev_all, cal5)

    # Registered specification, where it exists.
    cal6, _ = _calipers_from_main(with_wp=True)
    sub = d.dropna(subset=["wp_before"])
    ev_sub = ev_all[ev_all.player.isin(sub.player.unique())]
    ev_sub = ev_sub.dropna(subset=["wp_before"]).reset_index(drop=True)
    one("with wp, engine-eval subset", sub, ev_sub, cal6, with_wp=True)

    C.ensure_dirs(C.DERIVED)
    res = pd.DataFrame(rows)
    path = os.path.join(C.DERIVED, f"mixed_{args.which}.csv")
    res.to_csv(path, index=False)
    print(f"\n→ {path}")
    print(res.to_string(index=False))


def cmd_cx_fens(args):
    """plies 에는 FEN 을 저장하지 않으므로 대국을 다시 재현해 국면을 뽑는다."""
    C.ensure_dirs(C.WORK, C.CX_FENS)
    keep = tier_map = None
    if os.path.exists(C.SAMPLE):
        samp = pd.read_parquet(C.SAMPLE)
        keep = set(samp[samp.tier != C.TITLE_TIER].player)
        tier_map = dict(zip(samp.player, samp.tier))

    parts, t = [], time.time()
    for i in range(args.start, args.end):
        tmp = os.path.join(C.WORK, f"_cx{i}.parquet")
        if not download(shard_url(C.MAIN_YEAR, C.MAIN_MONTH, i,
                                  C.MAIN_N_SHARDS), tmp):
            print(f"  [{i}] download FAILED", flush=True)
            continue
        try:
            part = sample_fens(tmp, n=2000, seed=i,
                               keep_players=keep, tier_map=tier_map)
            parts.append(part)
            print(f"  [{i}] {len(part):,}국면", flush=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if not parts:
        sys.exit("표집된 국면이 없다.")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(C.CX_FENS, index=False)
    print(f"{len(out):,}국면 / {time.time()-t:.0f}s → {C.CX_FENS}")


def cmd_cx(args):
    """Stockfish multi-PV 로 국면 복잡도를 잰다. 엔진이 필요하다."""
    from src.complexity import Engine, complexity, gap_top2
    _need(C.CX_FENS, "먼저 python run.py cx-fens 를 돌릴 것.")
    s = pd.read_parquet(C.CX_FENS)
    end = args.end if args.end is not None else len(s)
    s = s.iloc[args.start:end]

    C.ensure_dirs(C.WORK)
    out = f"{C.WORK}/cx_{args.start:05d}.parquet"
    if os.path.exists(out):
        print("skip")
        return
    recs, t = [], time.time()
    with Engine() as e:
        for r in s.itertuples():
            sc = e.analyse(r.fen, depth=args.depth)
            recs.append({"tier": r.tier, "ply": r.ply, "n_legal": r.n_legal,
                         "max_see_mine": getattr(r, "max_see_mine", None),
                         "mat_diff": getattr(r, "mat_diff", None),
                         "n_checks": getattr(r, "n_checks", None),
                         "sd": complexity(sc), "gap": gap_top2(sc),
                         "n_pv": len(sc)})
    pd.DataFrame(recs).to_parquet(out, index=False)
    print(f"[{args.start}:{end}] {len(recs)}국면 {time.time()-t:.0f}s → {out}")


def cmd_cx_model(args):
    """
    복잡도 예측 모형을 적합하고 cx_pred 를 prepared*.parquet 에 붙인다.

    ★ 이 모형은 원래 저장소에서 누락되어 있었다. stage 2 가 남기던 특징
      네 개로 재구성한 것이므로, 논문의 r = 0.47 을 재현하는지 아래 출력으로
      반드시 확인할 것.
    """
    from src.complexity import (CX_FEATURES, fit_cx_model, predict_cx,
                                save_model)
    files = sorted(glob.glob(f"{C.WORK}/cx_*.parquet"))
    if not files:
        sys.exit(f"복잡도 측정 결과가 없다: {C.WORK}/cx_*.parquet\n"
                 f"먼저 run.py cx-fens → run.py cx 를 돌릴 것.")
    cx = pd.concat([pd.read_parquet(f) for f in files],
                   ignore_index=True).dropna(subset=["sd"])
    print(f"측정 국면 {len(cx):,}개")
    ok = cx.dropna(subset=["n_legal", "sd"])
    print(f"  n_legal ~ 복잡도      r = {np.corrcoef(ok.n_legal, ok.sd)[0,1]:+.3f}"
          f"   (논문 보고: -0.16)")
    model = fit_cx_model(cx)
    print(f"  예측 모형             r = {model['r']:+.3f}   (논문 보고: +0.47)")
    print(f"  특징: {model['features']}")
    print(f"  계수: {[round(c, 5) for c in model['coef']]}  "
          f"절편 {model['intercept']:.4f}  n={model['n']:,}")

    C.ensure_dirs(C.CX_MODEL)
    save_model(model, C.CX_MODEL)
    print(f"→ {C.CX_MODEL}")
    for path in (C.PREPARED, C.PREPARED_FM):
        if not os.path.exists(path):
            continue
        d = pd.read_parquet(path)
        missing = [f for f in CX_FEATURES if f not in d.columns]
        if missing:
            print(f"  건너뜀 {os.path.basename(path)}: 특징 없음 {missing}")
            continue
        d["cx_pred"] = predict_cx(d, model)
        d.to_parquet(path, index=False)
        print(f"  cx_pred 추가 → {path} "
              f"(결측 {100 * d.cx_pred.isna().mean():.1f}%)")


# ══════════════════════════════════════════════════════════════
# 산출물 정리
# ══════════════════════════════════════════════════════════════

def cmd_aggregate(args):
    """
    {WORK} 의 실행별 parquet 를 data/derived/ 의 CSV 한 장으로 합친다.

    lag_profiles.csv 의 event_type 이 material / material_see 로 구분된다.
    예전 판은 구분 없이 material 로만 적혀 있어서, 철회된 SEE 정의의 결과가
    논문 정의인 것처럼 보였다 (t+1 이 −0.205, 논문 값은 −0.587).
    """
    C.ensure_dirs(C.DERIVED)
    jobs = [(f"{C.WORK}/lag_*.parquet", "lag_profiles.csv"),
            (f"{C.WORK}/lagsplit_*.parquet", "lagsplit_profiles.csv")]
    for pattern, name in jobs:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"건너뜀 {name}: {pattern} 에 파일이 없다")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        cols = [c for c in ["event_type", "tier", "spec", "lag"]
                if c in df.columns]
        if cols:
            df = df.sort_values(cols).reset_index(drop=True)
        path = os.path.join(C.DERIVED, name)
        df.to_csv(path, index=False)
        print(f"→ {path}  ({len(files)}개 파일, {len(df):,}행)")
        if cols:
            for k, v in df.groupby(cols[:-1]).size().items():
                print(f"     {k}  {v}행")


def cmd_anonymize(args):
    """
    남아 있는 리체스 계정명을 소금친 해시로 바꾼다.

    계정명은 CC0 공개 정보라 재배포 자체는 합법이지만, 식별 가능한 계정에
    개인별 행동 추정치를 붙여 공개하는 것은 별개 문제다. 실행 여부는
    연구자 판단이므로 자동으로 돌지 않는다.
    """
    C.ensure_dirs(C.WORK)
    salt_file = os.path.join(C.WORK, "anon_salt.txt")
    if os.path.exists(salt_file):
        salt = open(salt_file, encoding="utf-8").read().strip()
    else:
        salt = os.urandom(16).hex()
        open(salt_file, "w", encoding="utf-8").write(salt)
        print(f"새 솔트 생성 → {salt_file}  (잃으면 매핑을 못 되돌린다)")

    map_file = os.path.join(C.WORK, "player_mapping.json")
    mapping = (json.load(open(map_file, encoding="utf-8"))
               if os.path.exists(map_file) else {})

    targets = [(C.WORK, "sample.parquet"), (C.WORK, "sample_fm.parquet"),
               (C.DERIVED, "per_player_t1.csv")]
    for base, name in targets:
        path = os.path.join(base, name)
        if not os.path.exists(path):
            print(f"건너뜀 {name}: 파일 없음")
            continue
        read = pd.read_parquet if name.endswith(".parquet") else pd.read_csv
        d = read(path)
        if "player" not in d.columns:
            print(f"건너뜀 {name}: player 열 없음")
            continue
        names = d["player"].astype(str)
        if names.str.fullmatch(r"P[0-9a-f]{12}").all():
            print(f"건너뜀 {name}: 이미 익명화됨")
            continue
        for n in names.unique():
            mapping.setdefault(n, "P" + hashlib.sha256(
                (salt + n).encode("utf-8")).hexdigest()[:12])
        print(f"{name}: {len(d):,}행 / 계정 {names.nunique():,}개 "
              f"(예: {names.iloc[0]} → {mapping[names.iloc[0]]})")
        if args.dry_run:
            continue
        d["player"] = names.map(mapping)
        (d.to_parquet if name.endswith(".parquet") else d.to_csv)(
            path, index=False)

    if args.dry_run:
        print("\n--dry-run 이므로 아무것도 바꾸지 않았다.")
        return
    json.dump(mapping, open(map_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n매핑 {len(mapping):,}건 → {map_file}")
    print("이 파일과 솔트는 .gitignore 에 있다. 공개 저장소에 올리지 말 것.")


# ══════════════════════════════════════════════════════════════
# 논문이 보고하는 나머지 분석
# ══════════════════════════════════════════════════════════════
# 아래 다섯 단계는 원래 저장소에 없었다. 논문은 결과를 보고하는데
# 만들어내는 코드가 없어, 해당 수치가 참조표에 전사값으로만 박혀 있었다.
# 그 참조표는 지금 src/external_values.py 다. docs/corrections.md 참조.


def _events_with_effect(d, cal, ev, seed=None):
    """사건별 효과를 낸다 (game_id·ply 포함 — 신뢰도 반분에 필요)."""
    from src.analysis import build_with_pretrend
    return build_with_pretrend(d, ev, cal, seed=seed or C.MATCH_SEED)


def _select_and_truncate(ev, threshold, seed=0):
    """
    Keep players with at least `threshold` events, then cut every one of them
    to exactly that many.

    A global cap cannot be used here. It divides a fixed budget among however
    many players qualify, so the observations per person -- the thing
    reliability is a function of -- depend on how many players there are. The
    published table came from a run that had quietly subsampled the player
    list (every third player, first 230), which left 443 events each; running
    the same command on all 607 players left 148 each and a different answer.

    Truncating per player fixes the observations per person regardless of how
    many qualify, so the number is reproducible from the rule alone.
    """
    rng = np.random.default_rng(seed)
    cnt = ev.groupby("player").size()
    sel = sorted(cnt[cnt >= threshold].index)
    if not sel:
        return ev.iloc[:0], 0, len(cnt)
    sub = ev[ev.player.isin(sel)].reset_index(drop=True)
    idx = []
    for _, g in sub.groupby("player", sort=True):
        idx.extend(rng.choice(g.index.values, threshold, replace=False))
    out = sub.loc[sorted(idx)].sort_values(SORT_KEY).reset_index(drop=True)
    return out, len(sel), len(cnt)


def cmd_reliability(args):
    """
    개인 수준 신뢰도 — 논문 Table 2 와 Fig 2d/2e.

    대국 단위로 반분하고 Spearman-Brown 으로 보정한다. 사건 단위로 나누면
    같은 대국의 사건끼리 상관돼 신뢰도가 부풀려진다.

    관측수 곡선은 플레이어마다 사건을 N개로 잘라가며 다시 잰다. 논문의
    주장("블런더는 관측을 더 모아도 오르지 않는다")은 관측수를 맞춘 비교라야
    성립하기 때문이다.

    출력: data/derived/reliability.csv        (Table 2)
          data/derived/reliability_curve.csv  (Fig 2d)
    """
    from src.analysis import reliability_curve, lagwise
    from src.prepare import reliability

    rows, curves = [], []
    for tier_label, is_fm in [("lower", False), ("titled", True)]:
        src = C.prepared_path(is_fm)
        if not os.path.exists(src):
            print(f"건너뜀 {tier_label}: {src} 없음")
            continue
        d = pd.read_parquet(src).dropna(subset=[C.PRESPEED_VAR])
        cal, _ = _calipers_from_main()
        for ev_label, which in [("Material loss", "B"), ("Blunder", "A")]:
            ev, _et = _events(d, which)
            thr = (args.select if args.select is not None
                   else C.RELIABILITY_SELECT[which])
            ev, n_sel, n_all = _select_and_truncate(
                ev, thr, seed=C.EVENT_SAMPLE_SEED)
            print(f"  {ev_label}, {tier_label}: {thr}건 이상 {n_sel}명 "
                  f"(전체 {n_all}명) → 인당 {thr}건 절단, "
                  f"사건 {len(ev):,}건", flush=True)
            if len(ev) < C.MIN_EVENTS:
                print(f"건너뜀 {ev_label}/{tier_label}: 사건 {len(ev)}건")
                continue
            print(f"  {ev_label}, {tier_label}: 사건 {len(ev):,}건 처리 중...",
                  flush=True)
            # The per-event effect is the t+1 difference, as everywhere else.
            #
            # This used to take build_with_pretrend's z_post, the mean of
            # t+1..t+3. Averaging cuts measurement error but dilutes the
            # signal too, and reliability is their ratio: material loss read
            # .393 that way against .685 on t+1, and titled blunders +0.283
            # against -0.032. Table 2's sigma_b is discussed beside the
            # tier-wise t+1 effects, so it has to be the same quantity.
            #
            # (Either window gives the same answer on the tier comparison:
            # between-player SD is within 7% across tiers in both.)
            prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
            e = prof[["player", "lag+1"]].copy()
            e.columns = ["player", "effect"]
            e["game_id"] = ev["game_id"].values
            e["ply"] = ev["ply"].values
            e = e.dropna(subset=["effect"])
            rel = reliability(e, min_events=args.min_events) or {}
            per = e.groupby("player")["effect"].size()
            rows.append({
                "label": f"{ev_label}, {tier_label}",
                "reliability": rel.get("sb", np.nan),
                "r_halves": rel.get("r", np.nan),
                "sigma_b": rel.get("sigma_b", np.nan),
                "sigma_w": rel.get("sigma_w", np.nan),
                "events_per_player": float(per.mean()) if len(per) else np.nan,
                "n_players": int(e.player.nunique()),
                "n_events": int(len(e)),
            })
            curves.append(reliability_curve(
                e, seed=C.EVENT_SAMPLE_SEED,
                label=f"{ev_label}, {tier_label}"))

    C.ensure_dirs(C.DERIVED)
    if rows:
        t2 = pd.DataFrame(rows)
        path = os.path.join(C.DERIVED, "reliability.csv")
        t2.to_csv(path, index=False)
        print(f"\n→ {path}")
        print(t2[["label", "reliability", "sigma_b", "events_per_player",
                  "n_players"]].to_string(index=False))
    if curves:
        cv = pd.concat(curves, ignore_index=True)
        path = os.path.join(C.DERIVED, "reliability_curve.csv")
        cv.to_csv(path, index=False)
        print(f"→ {path}")


def cmd_bins(args):
    """
    합법수 변화량 구간별 효과 — 논문 Fig 2b.

    재료가 사라진 방향(자기 손실 / 상대 손실)에 따라 합법수는 반대로 움직인다.
    그래서 합법수 변화를 고정한 구간 안에서 두 방향을 비교하면, "국면이
    단순해져서 빨라진다"는 설명을 배제할 수 있다.

    --size 9 를 주면 9점(퀸 값) 손실만 남긴다. 원고의 "widens the
    separation to -1.04 at its maximum" 이 그 실행의 값인데, 그 파일을
    만드는 단계가 저장소에 없었다. legal_bins_queen.csv 는 출처 없는
    파일로 남아 있었다.

    출력: data/derived/legal_bins.csv          (--size 없음)
          data/derived/legal_bins_queen.csv    (--size 9)
    """
    from src.analysis import bin_effects, lagwise
    from src.prepare import net_legal_change

    _need(C.PREPARED, "먼저 python run.py prep 을 돌릴 것.")
    d = pd.read_parquet(C.PREPARED).dropna(subset=[C.PRESPEED_VAR])
    d = net_legal_change(d)
    cal, _ = _calipers_from_main()

    bins = [-np.inf, -12, -8, -4, -1, 1, 4, 8, np.inf]
    sz = getattr(args, "size", None)
    if sz:
        masks = [("self_lost", d.net_mat == -sz), ("opp_lost", d.net_mat == sz)]
        print(f"  {sz}점 손실만", flush=True)
    else:
        masks = [("self_lost", d.net_mat < 0), ("opp_lost", d.net_mat > 0)]
    out = []
    for label, m in masks:
        ev = d[m].sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > args.max_events:
            ev = ev.sample(args.max_events, random_state=C.EVENT_SAMPLE_SEED)
            ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) == 0:
            print(f"  건너뜀 {label}: 해당 사건이 없다")
            continue
        print(f"  {label}: 사건 {len(ev):,}건 처리 중...", flush=True)

        # The effect is the t+1 difference, as in every other stage.
        #
        # This used to take build_with_pretrend's z_post, which averages
        # t+1..t+3. Because t+2 and t+3 sit near zero, that divides the
        # effect by roughly three: the same events gave -0.19 here and -0.51
        # under se split. Figure 2b shares its axis with the other panels, so
        # it has to be the same quantity.
        prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        e = prof[["player", "lag+1"]].copy()
        e.columns = ["player", "effect"]
        e["net_legal"] = ev["net_legal"].values
        out.append(bin_effects(e, "net_legal", bins, label=label))

        mean_change = ev.groupby(ev.net_mat.abs())["net_legal"].mean()
        for v in (1, 3, 5, 9):
            if v in mean_change.index:
                print(f"     {v}점 손실 시 합법수 평균 변화 "
                      f"{mean_change.loc[v]:+.2f}")

    if not out:
        sys.exit("구간별 산출이 없다 — 사건이 충분하지 않다.")
    C.ensure_dirs(C.DERIVED)
    res = pd.concat(out, ignore_index=True)
    path = os.path.join(C.DERIVED,
                        "legal_bins_queen.csv" if sz else "legal_bins.csv")
    res.to_csv(path, index=False)
    print(f"\n→ {path}")
    print(res[["label", "bin", "effect", "se", "n_events"]].to_string(index=False))


def cmd_describe(args):
    """
    사건 특성 — 논문 Results 의 기술 통계.

      · 세 집단의 승률 하락폭과 잃은 재료
      · 9점 이상 손실의 비중
      · 블런더 직후 다시 블런더가 나오는 비율
        (Discussion 의 "followed by further blunders at an elevated rate".
         원고는 이 값을 Results 에 보고하지 않는다)

    출력: data/derived/event_characteristics.csv
    """
    _need(C.PREPARED, "먼저 python run.py prep 을 돌릴 것.")
    d = pd.read_parquet(C.PREPARED).dropna(subset=[C.PRESPEED_VAR])
    hasA, hasB = mask_A(d), mask_B(d)
    groups = {"blunder_only": hasA & ~hasB,
              "blunder_loss": hasA & hasB,
              "loss_only": hasB & ~hasA}

    # 같은 플레이어의 다음 수가 블런더인가
    nxt_blunder = pd.Series(
        dict(zip(zip(d.game_id, d.player, d.ply), hasA.values)))
    key_next = list(zip(d.game_id, d.player, d.ply + 2))
    d = d.assign(next_is_blunder=[nxt_blunder.get(k, np.nan) for k in key_next])

    rows = []
    for name, m in groups.items():
        g = d[m]
        loss = g.net_mat[g.net_mat < 0].abs()
        nb = g.next_is_blunder.dropna()
        rows.append({
            "group": name,
            "n_events": int(len(g)),
            "wp_delta_mean": float(g.wp_delta.mean()),
            "material_lost_mean": float(loss.mean()) if len(loss) else np.nan,
            "material_lost_median": float(loss.median()) if len(loss) else np.nan,
            "share_ge_9": float((loss >= 9).mean()) if len(loss) else np.nan,
            "next_blunder_rate": float(nb.mean()) if len(nb) else np.nan,
            "n_next_obs": int(len(nb)),
        })
    base = d.next_is_blunder.dropna()
    rows.append({"group": "(all moves)", "n_events": int(len(d)),
                 "wp_delta_mean": float(d.wp_delta.mean()),
                 "material_lost_mean": np.nan, "material_lost_median": np.nan,
                 "share_ge_9": np.nan,
                 "next_blunder_rate": float(base.mean()) if len(base) else np.nan,
                 "n_next_obs": int(len(base))})

    C.ensure_dirs(C.DERIVED)
    res = pd.DataFrame(rows)
    path = os.path.join(C.DERIVED, "event_characteristics.csv")
    res.to_csv(path, index=False)
    print(f"→ {path}")
    print(res.to_string(index=False))
    print("\n주의: next_blunder_rate 는 엔진 평가가 있는 수에서만 정의된다. "
          "기저율((all moves))과 함께 봐야 한다.")


def cmd_robustness(args):
    """
    사전등록된 여섯 사양을 대안 수준에서 재분석 — 논문 Robustness 절.

      오프닝 컷      10 / 14 / 20 플라이
      측정 하한      1 / 2 / 3 초        ← z 를 다시 표준화한다
      블런더 임계    10 / 20 / 30 %p
      매칭 캘리퍼    0.1 / 0.2 / 0.4 / 0.8 SD
      복잡도 통제    run.py lag 의 none / nlegal / nlegal_cx 로 본다
      사전 속도      run.py lag zpre / run.py did 로 본다

    출력: data/derived/robustness.csv
    """
    _need(C.PREPARED, "먼저 python run.py prep 을 돌릴 것.")
    base = pd.read_parquet(C.PREPARED)

    def effect_at_t1(d, cal, which, n=None):
        d = d.dropna(subset=[C.PRESPEED_VAR])
        ev, _ = _events(d, which)
        ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        n = n or args.max_events
        if len(ev) > n:
            ev = ev.sample(n, random_state=C.EVENT_SAMPLE_SEED)
            ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) < C.MIN_EVENTS:
            return None
        prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        return cluster_stats(prof[["player", "lag+1"]].dropna(), "lag+1")

    def restandardise(d, tau):
        """측정 하한을 바꾸면 z 를 다시 계산해야 한다."""
        from src.prepare import add_pre_speed
        x = d[d.move_time >= tau].copy()
        x["y"] = np.log(x.move_time + C.LOG_OFFSET)
        g = x.groupby("player")["y"]
        x["z"] = (x.y - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        x = x.dropna(subset=["z"]).reset_index(drop=True)
        return add_pre_speed(x, k=C.PRESPEED_K)

    rows = []

    def record(spec, level, which, st):
        if st is None:
            return
        rows.append({"spec": spec, "level": level, "event": which,
                     "effect": st["effect"], "se": st["se"],
                     "effect_pw": st["effect_pw"],
                     "n_events": st["n_events"], "n_players": st["n_players"]})
        print(f"  {spec:<18}{str(level):<8}{which:<4}"
              f"{st['effect']:+.4f} ± {st['se']:.4f}", flush=True)

    print("오프닝 컷")
    for cut in C.SENS_OPENING_CUT:
        d = base[base.ply >= cut]
        cal, _ = _calipers_from_main()
        record("opening_cut", cut, "B", effect_at_t1(d, cal, "B"))

    print("측정 하한")
    for tau in C.SENS_TAU:
        d = base if tau == C.TAU else restandardise(base, tau)
        cal, _ = _calipers_from_main()
        record("measurement_floor", tau, "B", effect_at_t1(d, cal, "B"))

    print("블런더 임계")
    d = base.dropna(subset=[C.PRESPEED_VAR])
    cal, _ = _calipers_from_main()
    for th in C.GRID_BLUNDER_THRESH:
        ev = events_A(d, thresh=th).sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) > args.max_events:
            ev = ev.sample(args.max_events, random_state=C.EVENT_SAMPLE_SEED)
            ev = ev.sort_values(SORT_KEY).reset_index(drop=True)
        if len(ev) < C.MIN_EVENTS:
            continue
        prof = lagwise(d, ev, cal, seed=C.MATCH_SEED)
        record("blunder_threshold", th, "A",
               cluster_stats(prof[["player", "lag+1"]].dropna(), "lag+1"))

    print("매칭 캘리퍼")
    _, sd = _calipers_from_main()
    for mult in C.SENS_CALIPER:
        cal = {v: mult * sd[v] for v in C.MATCH_VARS}
        cal[C.PRESPEED_VAR] = C.PRESPEED_CALIPER_SD * sd[C.PRESPEED_VAR]
        record("caliper", mult, "B", effect_at_t1(base, cal, "B"))

    # The registered six-variable caliper, reported as a deviation.
    #
    # Computed on the engine-evaluated subsample, where both specifications
    # exist, so the comparison is within one sample rather than across two.
    print("등록 사양 (승률 매칭 포함)")
    cal_wp, _ = _calipers_from_main(with_wp=True)
    cal_5, _ = _calipers_from_main()
    sub = base.dropna(subset=["wp_before"])
    for tag, cal in [("5 vars (main)", cal_5), ("6 vars (registered)", cal_wp)]:
        st = effect_at_t1(sub, cal, "B")
        record("caliper_vars", tag, "B", st)
        if st:
            print(f"  {tag:<22} {st['effect']:+.4f} +/- {st['se']:.4f}  "
                  f"플레이어 {st['n_players']:,}", flush=True)

    C.ensure_dirs(C.DERIVED)
    res = pd.DataFrame(rows)
    path = os.path.join(C.DERIVED, "robustness.csv")
    res.to_csv(path, index=False)
    print(f"\n→ {path}")
    print("복잡도 통제와 사전 속도는 run.py lag <B> <none|nlegal|nlegal_cx|zpre> "
          "및 run.py did 로 본다.")


def cmd_verify_paper(args):
    """
    논문에 실린 수치와 data/derived/ 를 대조한다.

    전사값은 src/paper_check.py 에 있다. 논문을 고치면 거기도 고칠 것.
    불일치는 DATA(저장소 재생성 필요) / PAPER(원고 표기 확인) 로 표시된다.
    """
    from src.paper_check import run as verify
    rep = verify()
    sys.exit(1 if rep.bad else 0)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        prog="run.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<stage>")

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_, description=fn.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        s.set_defaults(fn=fn)
        return s

    s = add("scan", cmd_scan, "stage 1 - account metadata")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=C.MAIN_N_SHARDS)

    s = add("scan-titled", cmd_scan_titled,
            "stage 1, titled accounts (all six months if omitted)")
    s.add_argument("month", nargs="?", type=int, default=None,
                   choices=sorted(C.TITLED_MONTHS))
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=None)

    add("sample", cmd_sample, "fix the player sample -> out/ (not distributed)")

    s = add("extract", cmd_extract, "stage 2 - board replay (slowest stage)")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=C.MAIN_N_SHARDS)
    s.add_argument("--force", action="store_true",
                   help="표본이 논문과 달라도 진행한다")

    s = add("extract-titled", cmd_extract_titled, "stage 2, titled tier, 2024-01 .. 2024-06")
    s.add_argument("month", nargs="?", type=int, default=None)
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=None)

    s = add("prelim", cmd_prelim, "preliminary extraction (shards 0-23)")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=24)

    s = add("prep", cmd_prep, "analysis table (net_mat, z, z_pre3)")
    s.add_argument("fm", nargs="?", default=False,
                   type=lambda v: str(v).lower().startswith("fm"))

    s = add("se", cmd_se, "clustered SEs -> data/derived/se_*.csv")
    s.add_argument("--tiers", choices=["five", "four"], default="five",
                   help="five: 타이틀 층 포함(기본) · four: 하위 네 층만")
    s.add_argument("mode", choices=["split", "dose", "tier"])

    s = add("lag", cmd_lag, "one lag profile")
    s.add_argument("which", choices=["A", "B", "fmA", "fmB"])
    s.add_argument("spec", choices=list(SPECS))
    s.add_argument("--nmax", type=int, default=C.MAX_EVENTS_LAG)
    s.add_argument("--see", action="store_true",
                   help="철회된 SEE 정의로 사건 B 를 잡는다 (부록 비교 전용)")
    s.add_argument("--by-tier", action="store_true", dest="by_tier",
                   help="층별로 따로 낸다 (논문 Fig 1a/1b 의 다섯 곡선)")

    add("lag-all", cmd_lag_all, "every lag profile, then aggregate")

    s = add("lag-split", cmd_lag_split, "three-group lag decomposition")
    s.add_argument("which", choices=["lower", "fm"])
    s.add_argument("grp", choices=["Aonly", "Bonly", "AB"])
    s.add_argument("--no-prespeed", action="store_true", dest="no_prespeed",
                   help="z_pre3 를 캘리퍼에서 빼 옛 사양을 재현한다")

    for name, fn, help_ in [("ps6", cmd_ps6, "pre-speed matched, with pre-trend"),
                            ("did", cmd_did, "difference-in-differences")]:
        s = add(name, fn, help_)
        s.add_argument("which", nargs="?", default="B",
                       choices=["A", "B", "fmA", "fmB"])
        s.add_argument("start", nargs="?", type=int, default=0)
        s.add_argument("end", nargs="?", type=int, default=None)

    s = add("mixed", cmd_mixed, "preregistered mixed-effects model")
    s.add_argument("which", nargs="?", default="B",
                   choices=["A", "B", "fmA", "fmB"])
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_MIXED,
                   dest="max_events")

    s = add("cx-fens", cmd_cx_fens, "sample positions for complexity validation")
    s.add_argument("start", nargs="?", type=int, default=24)
    s.add_argument("end", nargs="?", type=int, default=34)

    s = add("cx", cmd_cx, "measure complexity with Stockfish")
    s.add_argument("start", nargs="?", type=int, default=0)
    s.add_argument("end", nargs="?", type=int, default=None)
    s.add_argument("--depth", type=int,
                   default=int(os.environ.get("CX_DEPTH", 12)))

    add("cx-model", cmd_cx_model, "complexity prediction model, adds cx_pred")
    add("aggregate", cmd_aggregate, "fold per-run output into data/derived/*.csv")

    s = add("anonymize", cmd_anonymize, "replace account names with salted hashes")
    s.add_argument("--dry-run", action="store_true", dest="dry_run")

    s = add("reliability", cmd_reliability, "per-player reliability (Table 2)")
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_SE,
                   dest="max_events")
    s.add_argument("--min-events", type=int, default=10, dest="min_events")
    s.add_argument("--select", type=int, default=None,
                   help="매칭 전 사건 수 기준 플레이어 선별. 생략하면 "
                        "config.RELIABILITY_SELECT (기물 237 · 블런더 40)")

    s = add("bins", cmd_bins, "effect within legal-move-count bins")
    s.add_argument("--size", type=int, default=None,
                   help="이 점수의 손실만 (예: 9 = 퀸). 생략하면 전체")
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_SE,
                   dest="max_events")

    add("describe", cmd_describe, "event descriptives")

    s = add("robustness", cmd_robustness, "the six registered specifications")
    s.add_argument("--max-events", type=int, default=C.MAX_EVENTS_SE,
                   dest="max_events")

    add("verify-paper", cmd_verify_paper, "manuscript values vs data/derived/")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
