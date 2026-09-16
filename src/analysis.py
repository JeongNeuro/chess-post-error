"""
Matching, lag decomposition, pre-trend, mixed-effects models

  1. caliper matching and clustered SEs        caliper_mask / cluster_stats
  2. decomposition by lag                      lagwise / summarize
  3. pre-trend and difference-in-differences   build_with_pretrend
  4. preregistered mixed-effects model         build_pairs / fit_mixed

Driver: `run.py`.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE ★ the standard errors come from here
#   - Clustered SEs are computed in exactly one place, cluster_stats:
#     groupby('player').mean() followed by .sem(). Computing them at the
#     event level underestimates them — that was wrong once.
#   - Both effect (event-weighted) and effect_pw (player-weighted) are
#     reported. A large gap between them signals that events per player are
#     badly unbalanced.
#   - caliper_mask DISCARDS an event that is missing any matching variable
#     (strict=True). Previously that variable alone was skipped and the event
#     matched on the rest, so some events entered on looser criteria than the
#     manuscript describes ("all six within the caliper").
#   - The control pool is not "moves with no event"; it is "moves that are not
#     the event currently under analysis". This belongs in the limitations.
#   - Windows count only the player's own moves, hence ply ± 2k.
# ───────────────────────────────────────────────────────────

import warnings

import numpy as np
import pandas as pd

from .config import EPOCH_WINDOW

LAGS = list(range(-6, 4))          # −6 … +3


# ════════════════════════════════════════════════════════════════════
# 1. 캘리퍼 매칭과 클러스터 SE
# lagwise / pretrend / mixed 세 곳이 같은 매칭 로직을 각자 복사해 갖고
# 있었다. 세 벌이 조금씩 달라질 위험이 있어 여기로 합쳤다.
# 
# 캘리퍼 결측 처리 (중요)
#     strict=True 면 매칭 변수 중 하나라도 결측인 사건은 통째로 버린다.
#     대조 풀은 dropna 로 걸러지므로, 이게 아니면 사건 쪽만 조용히
#     사양이 바뀐다.
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 점검 안내
# 점검 포인트 ★
#   - caliper_mask: 사건 하나에 대해 대조 후보를 고르는 마스크.
#     strict=True 면 매칭 변수 중 하나라도 결측인 사건은 통째로 버린다.
#   - window_mean: 자기 수만 세므로 ply ± 2k 로 이동한다.
# ─────────────────────────────────────────────────────────────


def is_missing(v):
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def caliper_mask(row, cand, cal, strict=True):
    """
    사건 row 에 대해 대조 후보 DataFrame cand 의 행별 채택 여부.

    반환: (mask, ok)
      mask : np.ndarray[bool]  — cand 와 같은 길이
      ok   : False 면 이 사건은 매칭 불가 (strict 에서 결측 변수 발생)
    """
    m = np.ones(len(cand), bool)
    for v, w in cal.items():
        rv = getattr(row, v, None)
        if is_missing(rv):
            if strict:
                return m, False
            continue
        if v not in cand.columns:
            if strict:
                return m, False
            continue
        m &= np.abs(cand[v].values - rv) <= w
    return m, True


def build_zmap(d):
    """(game_id, player, ply) → z 조회표"""
    return dict(zip(zip(d.game_id.values, d.player.values, d.ply.values),
                    d.z.values))


def window_mean(zmap, gid, pl, ply, window, sign=1):
    """
    사건 기준 플라이에서 ±window 구간의 z 평균.
    같은 플레이어의 수만 세므로 실제 플라이는 ply + sign*2k.
    반환: (평균 또는 nan, 실제로 관측된 개수)
    """
    zs = []
    for k in range(1, window + 1):
        v = zmap.get((gid, pl, ply + sign * 2 * k))
        if v is not None and not np.isnan(v):
            zs.append(v)
    return (float(np.mean(zs)) if zs else np.nan), len(zs)


def control_pool(d, events, cal):
    """
    대조 후보 풀: 사건 자신을 제외하고, 매칭 변수가 모두 관측된 행.

    주의 — 이 풀은 "무사건 수"가 아니라 "지금 분석 중인 사건이 아닌 수"다.
    다른 종류의 사건(예: 블런더만 볼 때의 재료 손실)이나 다른 사건의
    사후 창에 걸린 수가 후보로 남는다. 한계 섹션에 명시할 것.
    """
    ev_idx = set(zip(events.game_id.values, events.ply.values))
    keep = ~np.fromiter(
        (k in ev_idx for k in zip(d.game_id.values, d.ply.values)),
        dtype=bool, count=len(d))
    cols = [v for v in cal if v in d.columns]
    pool = d[keep]
    if cols:
        pool = pool.dropna(subset=cols)
    return pool


def cluster_stats(df, value_col, player_col="player"):
    """
    플레이어 클러스터 기준 요약.

    effect      사건 가중 평균 (논문 보고값)
    se          플레이어 평균들의 SEM (클러스터 SE)
    effect_pw   플레이어 동일가중 평균 — effect 와 se 의 가중이 다르므로
                둘이 크게 어긋나면 클러스터 크기 불균형이 심하다는 신호다
    """
    s = df[[player_col, value_col]].dropna()
    if len(s) == 0:
        return None
    per = s.groupby(player_col)[value_col].mean()
    se = per.sem()
    eff = s[value_col].mean()
    return {
        "effect": eff,
        "se": se,
        "effect_pw": per.mean(),
        "t": eff / se if se and se > 0 else np.nan,
        "n_events": len(s),
        "n_players": len(per),
    }

# ════════════════════════════════════════════════════════════════════
# 2. lag별 분해
# 기존 Stage 3는 사건 전 구간(t−3~t−1)을 평균 하나로 보고했다.
# 그 평균은 두 상황을 구분하지 못한다:
# 
#   계획 수립 후 실행:  t−3=0, t−2=큰 양수, t−1=음수  → 평균 +0.02
#   국면 난이도:        t−3=+0.02, t−2=+0.02, t−1=+0.02 → 평균 +0.02
# 
# lag k = −6 … +3 각각에 대해 사건-대조 차이를 산출해 사전 구간의
# **모양**을 관측한다. 창이 대국 밖으로 벗어나면 결측 → lag마다 n이 다르다.
# ════════════════════════════════════════════════════════════════════



EMPTY_LAG_COLS = ["player", "tier", "n_ctrl", "z_at"] +                  [f"lag{k:+d}" for k in LAGS]
EMPTY_EPOCH_COLS = ["player", "tier", "game_id", "ply", "z_post", "z_ctrl",
                    "z_pre", "z_ctrl_pre", "z_at", "z_ctrl_at",
                    "n_obs", "n_before", "n_ctrl",
                    "effect", "pretrend", "at_event", "did"]


def lagwise(d, events, cal, lags=LAGS, seed=0, max_events=None, strict=True):
    """
    사건별 lag 프로파일을 산출한다.
    반환: DataFrame — 한 행이 한 사건, 컬럼 lag_-6 … lag_+3
    """
    if len(events) == 0:
        return pd.DataFrame(columns=["player", "tier", "n_ctrl", "z_at"]
                            + [f"lag{k:+d}" for k in lags])
    rng = np.random.default_rng(seed)
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    # 메모리 절약: 사건이 있는 플레이어의 대국만 남긴다
    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)

    pool = control_pool(d, events, cal) if cal else d
    by_player = {p: g for p, g in pool.groupby("player")}

    def prof(gid, pl, ply):
        """기준 플라이에서 각 lag의 z (없으면 nan)"""
        return [zmap.get((gid, pl, ply + 2 * k), np.nan) for k in lags]

    recs = []
    for r in events.itertuples():
        ze = prof(r.game_id, r.player, r.ply)
        g = by_player.get(r.player)
        zc = [np.nan] * len(lags)
        ncand = 0
        if g is not None and len(g):
            if cal:
                m, ok = caliper_mask(r, g, cal, strict=strict)
                c = g[m] if ok else g.iloc[:0]
            else:
                # 무통제: 같은 플레이어의 다른 플라이 중 무작위 20개
                c = g.iloc[rng.choice(len(g), min(20, len(g)), replace=False)]
            ncand = len(c)
            if ncand:
                mat = np.array([prof(x.game_id, x.player, x.ply)
                                for x in c.itertuples()], dtype=float)
                # 어떤 lag 는 후보 전부가 결측일 수 있다 (창이 대국 밖).
                # nanmean 의 "Mean of empty slice" 경고를 삼킨다.
                with np.errstate(invalid="ignore"),                         warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    zc = np.nanmean(mat, axis=0)
        row = {"player": r.player, "tier": r.tier, "n_ctrl": ncand,
               "z_at": ze[lags.index(0)]}
        for i, k in enumerate(lags):
            row[f"lag{k:+d}"] = ze[i] - zc[i]
        recs.append(row)
    return pd.DataFrame(recs)


def summarize(prof_df, event_type, tier, spec, lags=LAGS, min_n=30):
    """lag별 effect / SE(플레이어 클러스터) / n"""
    out = []
    for k in lags:
        col = f"lag{k:+d}"
        s = prof_df[["player", col]].dropna()
        if len(s) < min_n:
            continue
        st = cluster_stats(s, col)
        out.append({"event_type": event_type, "tier": tier, "spec": spec,
                    "lag": k, **st})
    return pd.DataFrame(out)


def per_player_at(prof_df, lag=1, label=None):
    """
    특정 lag 의 플레이어별 평균 효과.
    Fig 2a 의 분포 패널과 data/derived/per_player_t1.csv 의 원천이다.
    """
    col = f"lag{lag:+d}"
    s = prof_df[["player", col]].dropna()
    per = s.groupby("player")[col].mean().rename("eff").reset_index()
    if label is not None:
        per.insert(0, "group", label)
    return per

# ════════════════════════════════════════════════════════════════════
# 3. 사전 추세 검정 (pre-trend check)
# Pfister & Foerster(2021)가 지적한 pre-error speeding 문제에 대응한다.
# 사람은 오류를 범하기 전부터 이미 빨라져 있을 수 있고, 그 경우
# "오류 후 정상 복귀"가 slowing으로 잘못 읽힌다.
# 
#   사건 전 차이 없음 → 매칭 적절. 결과 강화
#   사건 전 차이 있음 → 매칭 불완전. 보정 필요
# ════════════════════════════════════════════════════════════════════

def build_with_pretrend(d, events, cal, window=EPOCH_WINDOW, seed=0,
                        max_events=None, strict=True):
    """
    사건/대조 에폭의 창 z를 사건 전후 모두 산출한다.

    post : t+1 ~ t+window   (기존 종속변수)
    pre  : t−window ~ t−1   (사전 추세 검정용)

    같은 플레이어의 수이므로 실제 플라이는 ±2k.
    """
    if len(events) == 0:
        return pd.DataFrame(columns=EMPTY_EPOCH_COLS)
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)
    by_player = {p: g for p, g in control_pool(d, events, cal).groupby("player")}

    recs = []
    for r in events.itertuples():
        zp, nobs = window_mean(zmap, r.game_id, r.player, r.ply, window, +1)
        zb, nbef = window_mean(zmap, r.game_id, r.player, r.ply, window, -1)
        z_at = zmap.get((r.game_id, r.player, r.ply))     # 사건 수 자체

        g = by_player.get(r.player)
        zc = zcb = zc_at = np.nan
        ncand = 0
        if g is not None and len(g):
            m, ok = caliper_mask(r, g, cal, strict=strict)
            c = g[m] if ok else g.iloc[:0]
            ncand = len(c)
            if ncand:
                post, pre, at = [], [], []
                for x in c.itertuples():
                    a, _ = window_mean(zmap, x.game_id, x.player, x.ply,
                                       window, +1)
                    b, _ = window_mean(zmap, x.game_id, x.player, x.ply,
                                       window, -1)
                    if not np.isnan(a):
                        post.append(a)
                    if not np.isnan(b):
                        pre.append(b)
                    v = zmap.get((x.game_id, x.player, x.ply))
                    if v is not None and not np.isnan(v):
                        at.append(v)
                zc = float(np.mean(post)) if post else np.nan
                zcb = float(np.mean(pre)) if pre else np.nan
                zc_at = float(np.mean(at)) if at else np.nan

        recs.append({
            "player": r.player, "tier": r.tier,
            "game_id": r.game_id, "ply": r.ply,
            "z_post": zp, "z_ctrl": zc,
            "z_pre": zb, "z_ctrl_pre": zcb,
            "z_at": z_at, "z_ctrl_at": zc_at,
            "n_obs": nobs, "n_before": nbef, "n_ctrl": ncand,
        })
    e = pd.DataFrame(recs)
    e["effect"] = e.z_post - e.z_ctrl          # 사후 (기존 종속변수)
    e["pretrend"] = e.z_pre - e.z_ctrl_pre     # 사전 (0이어야 정상)
    e["at_event"] = e.z_at - e.z_ctrl_at       # 사건 수 자체
    e["did"] = e.effect - e.pretrend           # 이중차분 보정
    return e


def pretrend_report(e, label=""):
    """
    SE 는 플레이어 클러스터다. 예전 구현은 .sem() 을 사건 단위로 썼다.
    """
    ok = e.dropna(subset=["effect", "pretrend"])
    out = {"label": label, "n": len(ok)}
    for k in ["pretrend", "at_event", "effect", "did"]:
        st = cluster_stats(ok, k)
        out[k] = (np.nan, np.nan) if st is None else (st["effect"], st["se"])
    return out

# ════════════════════════════════════════════════════════════════════
# 4. 사전등록된 혼합효과 모형
# OSF Statistical models 사양:
#     z_DV ~ event_condition + wp_level + remaining_time + legal_move_count
#            + log(total_games) + (event_condition | player)
# 
# 각 사건 에폭이 두 행을 낳는다.
#   event_condition = 1 : 사후 창(t+1~t+3)의 z 평균
#   event_condition = 0 : 매칭된 대조 창들의 z 평균
# ════════════════════════════════════════════════════════════════════

def build_pairs(d, events, cal, seed=0, max_events=None, strict=True,
                window=1):
    """사건/대조 쌍을 장형으로 반환 — 공변량 포함"""
    if len(events) == 0:
        return pd.DataFrame(columns=["player", "tier", "wp_level",
                                     "remaining_time", "legal_moves", "ply",
                                     "pre_speed", "eid",
                                     "z_dv", "event_condition"])
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)
    by_player = {p: g for p, g in control_pool(d, events, cal).groupby("player")}

    def win(gid, pl, ply):
        """
        The dependent variable is the move at t+1.

        It used to be the mean of t+1..t+3. Every other stage reports t+1, and
        because t+2 and t+3 sit near zero the average ran about 0.2 where the
        matched estimate ran about 0.5 -- two numbers for one quantity, side
        by side in the same Results section.

        ps6 and did keep the window: a difference-in-differences needs the
        pre- and post-windows to be symmetric, so there the window is the
        design rather than a choice.
        """
        if window == 1:
            v = zmap.get((gid, pl, ply + 2))
            return np.nan if v is None else v
        return window_mean(zmap, gid, pl, ply, window, +1)[0]

    recs = []
    for r in events.itertuples():
        zp = win(r.game_id, r.player, r.ply)
        if np.isnan(zp):
            continue
        g = by_player.get(r.player)
        if g is None or not len(g):
            continue
        m, ok = caliper_mask(r, g, cal, strict=strict)
        if not ok:
            continue
        c = g[m]
        if not len(c):
            continue
        zs = [win(x.game_id, x.player, x.ply) for x in c.itertuples()]
        zs = [v for v in zs if not np.isnan(v)]
        if not zs:
            continue
        zc = float(np.mean(zs))
        common = {"player": r.player, "tier": r.tier,
                  "wp_level": r.wp_before, "remaining_time": r.clk_before,
                  "legal_moves": r.n_legal, "ply": r.ply,
                  "pre_speed": getattr(r, "z_pre3", np.nan),
                  "eid": len(recs)}
        recs.append({**common, "z_dv": zp, "event_condition": 1})
        recs.append({**common, "z_dv": zc, "event_condition": 0})
    return pd.DataFrame(recs)


def fit_mixed(long_df, activity=None, with_wp=False):
    """
    무선 절편 + 무선 기울기 모형.
    비수렴 시 (1) 무선 절편만 (2) 쌍 차이 순으로 후퇴한다.
    """
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")

    df = long_df.copy()

    # Controls follow config.MATCH_VARS, the main specification.
    #
    # The formula used to name wp_level unconditionally. Win probability is
    # missing for 83% of untitled moves, and statsmodels drops those rows, so
    # the "main" model was silently fitted on 5,426 of 32,782 rows -- too few
    # per player to converge. It then fell back to the paired difference and
    # reported that as the model.
    #
    # Pass with_wp=True for the registered specification, which is only
    # computed on the engine-evaluated subsample where it is present.
    ctrl = ["remaining_time", "legal_moves", "ply", "pre_speed"]
    if with_wp:
        ctrl = ["wp_level"] + ctrl
    ctrl = [c for c in ctrl if c in df.columns and df[c].notna().any()]
    for c in ctrl:
        sd = df[c].std()
        df[c] = (df[c] - df[c].mean()) / (sd if sd else 1.0)
    df = df.dropna(subset=["z_dv"] + ctrl)
    formula = "z_dv ~ event_condition" + "".join(f" + {c}" for c in ctrl)
    if activity is not None:
        df = df.merge(activity, on="player", how="left")
        df["log_games"] = np.log(df["n_games"].fillna(df["n_games"].median()))
        df["log_games"] = (df.log_games - df.log_games.mean()) / df.log_games.std()
        formula += " + log_games"

    out = {"n_obs": len(df), "n_players": df.player.nunique(),
           "controls": " + ".join(ctrl)}
    try:
        md = smf.mixedlm(formula, df, groups=df["player"],
                         re_formula="~event_condition")
        fit = md.fit(method="lbfgs", maxiter=200)
        out["path"] = "random slope"
    except Exception:
        fit = None
    if fit is None or not fit.converged:
        try:
            md = smf.mixedlm(formula, df, groups=df["player"])
            fit = md.fit(method="lbfgs", maxiter=200)
            out["path"] = "random intercept (fallback 1)"
        except Exception:
            fit = None
    if fit is None or not fit.converged:
        # fallback 2: 쌍 차이
        w = df.pivot_table(index="eid", columns="event_condition",
                           values="z_dv").dropna()
        diff = w[1] - w[0]
        pl = df.drop_duplicates("eid").set_index("eid").player
        per = diff.groupby(pl).mean()
        out.update(path="paired difference (fallback 2)",
                   coef=diff.mean(), se=per.sem(), converged=False)
        return out, None

    out.update(coef=fit.params["event_condition"],
               se=fit.bse["event_condition"],
               z=fit.tvalues["event_condition"],
               p=fit.pvalues["event_condition"],
               converged=fit.converged)
    return out, fit


# ══════════════════════════════════════════════════════════════
# 5. 개인 수준 신뢰도 — 관측수 의존성
# ══════════════════════════════════════════════════════════════
# 논문 Table 2 와 Fig 2d/2e 가 여기서 나온다.
#
# 핵심 주장은 "블런더는 관측을 더 모아도 신뢰도가 오르지 않는다" 이므로,
# **관측수를 맞춰놓고** 비교해야 한다. 그래서 플레이어마다 사건을 N개로
# 잘라가며 신뢰도를 다시 잰다.
#
# 주의: 반분은 **대국 단위**로 나눈다. 같은 대국의 사건끼리는 상관되므로
# 사건 단위로 나누면 신뢰도가 부풀려진다.


def restrict_events(epochs, n, seed=0):
    """플레이어마다 사건을 무작위 n개로 제한한다. n개 미만인 사람은 제외."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in epochs.groupby("player", sort=False):
        if len(g) < n:
            continue
        keep.append(g.iloc[rng.choice(len(g), n, replace=False)])
    if not keep:
        return epochs.iloc[:0]
    return pd.concat(keep, ignore_index=True)


def reliability_curve(epochs, counts=(10, 20, 30, 40, 50, 60, 80, 100,
                                      150, 200, 237, 300),
                      seed=0, min_players=20, label=""):
    """
    관측수별 반분 신뢰도 (Spearman-Brown 보정).

    반환 열: label, n_events_per_player, r, sb, n_players
    플레이어가 min_players 미만으로 남는 관측수는 건너뛴다 — 논문이
    "beyond that count fewer than 25 players met the criterion, so no stable
    estimate could be obtained" 라고 적은 상황이 이것이다.
    """
    from .prepare import reliability
    out = []
    for n in counts:
        sub = restrict_events(epochs, n, seed=seed)
        n_pl = sub.player.nunique()
        if n_pl < min_players:
            out.append({"label": label, "n_events_per_player": n,
                        "r": np.nan, "sb": np.nan, "n_players": n_pl,
                        "note": f"플레이어 {n_pl}명 (<{min_players}) — 추정 불가"})
            continue
        rel = reliability(sub, min_events=max(2, n // 2))
        out.append({"label": label, "n_events_per_player": n,
                    "r": (rel or {}).get("r", np.nan),
                    "sb": (rel or {}).get("sb", np.nan),
                    "n_players": n_pl, "note": ""})
    return pd.DataFrame(out)


def bin_effects(ev, value_col, bins, effect_col="effect", label=""):
    """
    구간별 효과와 플레이어 클러스터 SE.

    Fig 2b — 합법수 변화량 구간별로 효과를 낸다.
    """
    d = ev.dropna(subset=[value_col, effect_col]).copy()
    d["_bin"] = pd.cut(d[value_col], bins)
    rows = []
    for b, g in d.groupby("_bin", observed=True):
        st = cluster_stats(g, effect_col)
        if st is None:
            continue
        rows.append({"label": label, "bin": str(b),
                     "bin_mid": float(b.mid), **st})
    return pd.DataFrame(rows)
