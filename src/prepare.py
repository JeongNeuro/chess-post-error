"""
Ply table → analysis table, and the event definitions

  1. net_mat    net material change. The manuscript's Event B is defined here
     net_legal  net change in legal move count (control variable)
  2. z_pre3     pre-event speed control (ply-indexed)
  3. prepare    measurement floor, then within-player z standardisation
  4. events_*   Event A (blunder) and Event B (net material loss)
  5. calipers   matching caliper construction

★ Label warning — the preregistration and the manuscript use different names.

    Registered   Manuscript   Code           Meaning
    A            A            events_A       blunder (win-probability drop)
    B            (dropped)    —              error under time pressure; too
                                             few events survived matching
    C            B            events_B       net material loss

  The manuscript uses two events, A and B. The code follows the manuscript.
  The withdrawn SEE criterion is events_B_see. See the table in
  docs/definitions.md.

Driver: `run.py prep`.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE ★ the revised event definition is built here
#   - net_mat = mat_diff[ply+2] − mat_diff[ply], two consecutive moves by the
#     same player. Indexed by ply, not by row position.
#   - Event condition is net_mat < 0, strictly. Even exchanges give 0 and are
#     therefore excluded.
#   - Confirm that z standardisation is within player.
#   - z_pre3 is **ply-indexed** (ply−2, ply−4, ply−6). The earlier
#     implementation used groupby().shift(), which is row position; when the
#     measurement floor removes an intervening move the two diverge. The lag
#     windows are ply-indexed, so one analysis was mixing both conventions.
#   - events_B uses net_mat (the manuscript); events_B_see uses SEE (withdrawn,
#     supplementary comparison only, never called from the main path).
#   - Order matters: net_mat is computed BEFORE the measurement floor,
#     z_pre3 AFTER standardisation.
# ───────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from .config import (TAU, EPOCH_WINDOW, LOG_OFFSET, CALIPER_SD, MATCH_VARS,
                     BLUNDER_THRESH, WP_LO, WP_HI, PRESPEED_K)
from .analysis import (caliper_mask, build_zmap, window_mean, control_pool,
                       cluster_stats)

KEY = ["game_id", "player", "ply"]


# ════════════════════════════════════════════════════════════════════
# 파생 변수 — net_mat / z_pre3
# 사전 추세 검정에서 사건 에폭이 사건 직전부터 이미 느렸음이 확인되었다
# (하위 4개 층 +0.021, FM+ +0.038). 승률·시간·복잡도·플라이를 맞춰도
# "그 사람이 직전에 얼마나 빨리 두고 있었는가"가 통제되지 않았기 때문이다.
# 
#   z_pre3(t) = mean( z(t−2), z(t−4), z(t−6) )     ← 플라이 인덱스 기준
# 
# 사전등록에 없던 변수이므로 강건성 검증으로 보고한다.
# ════════════════════════════════════════════════════════════════════

def add_pre_speed(d, k=3, col="z_pre3"):
    """
    각 플라이에 직전 k개 자기 수의 z 평균을 붙인다 (플라이 인덱스 기준).

    같은 플레이어의 연속한 수는 플라이가 2씩 차이나므로
    ply−2, ply−4, …, ply−2k 를 찾는다. 하나라도 없으면 NaN.
    """
    d = d.copy()
    dup = d.duplicated(subset=KEY).sum()
    if dup:
        raise ValueError(
            f"(game_id, player, ply) 가 중복이다 ({dup}행). "
            "z_pre3 를 플라이 기준으로 찾으려면 유일해야 한다.")

    s = d.set_index(KEY)["z"]
    acc = np.zeros(len(d), dtype=float)
    for i in range(1, k + 1):
        key = pd.MultiIndex.from_arrays(
            [d["game_id"].values, d["player"].values, d["ply"].values - 2 * i],
            names=KEY)
        acc += s.reindex(key).to_numpy(dtype=float)   # 없으면 NaN → 전파
    d[col] = acc / k
    return d


def _next_own(d, col):
    """같은 플레이어의 다음 수(ply+2)에서의 col 값. 없으면 NaN."""
    dup = d.duplicated(subset=KEY).sum()
    if dup:
        raise ValueError(f"(game_id, player, ply) 가 중복이다 ({dup}행).")
    s = d.set_index(KEY)[col]
    key = pd.MultiIndex.from_arrays(
        [d["game_id"].values, d["player"].values, d["ply"].values + 2],
        names=KEY)
    return s.reindex(key).to_numpy(dtype=float)


def net_legal_change(d, col="net_legal"):
    """
    같은 플레이어의 연속한 두 수 사이 합법수 변화.

        net_legal(t) = n_legal(t+2) − n_legal(t)

    논문 "Number of available options" 절이 쓰는 양이다. 재료를 잃은 쪽과
    상대가 잃은 쪽에서 이 값이 반대 방향으로 움직이므로(자기 9점 손실은
    평균 −8.52, 상대 9점 손실은 +6.86), 이 값을 고정한 채 효과를 비교하면
    "국면이 단순해져서"라는 설명을 배제할 수 있다.
    """
    d = d.copy()
    d[col] = _next_own(d, "n_legal") - d["n_legal"].to_numpy(dtype=float)
    return d


def net_material_change(d, col="net_mat"):
    """
    같은 플레이어의 연속한 두 수 사이 재료 차이 변화.

        net_mat(t) = mat_diff(t+2) − mat_diff(t)

    mat_diff 는 stage2v2 가 **수를 두기 직전**, **수를 두는 쪽 관점**으로
    기록한 재료 차이다. 따라서 net_mat < 0 이면 그 수 이후 다음 자기 차례가
    올 때까지 재료를 순손실한 것이다. 등가 교환은 0이 되어 사건이 아니다.

    docs/definitions.md 의 "Event B — Material net loss" 구현이다.
    """
    d = d.copy()
    d[col] = _next_own(d, "mat_diff") - d["mat_diff"].to_numpy(dtype=float)
    return d

# ════════════════════════════════════════════════════════════════════
# 사건 정의와 매칭
# 사건 A    승률 10%p 하락, 국면 10~90% (민감도 20/30%p)
#   사건 B    재료 순손실 (net_mat < 0). 등가 교환은 사건이 아니다
#   매칭      동일 플레이어 + 각 변수 0.2 SD 캘리퍼 (Austin 2011), 1:N 평균
#   종속변수  z = (ln(move_time+1) − μ_p)/σ_p, 효과 = z(사건) − z(대조)
# ════════════════════════════════════════════════════════════════════

def prepare(df):
    """측정 하한 적용 + 플레이어 내 z 표준화"""
    d = df[df.move_time >= TAU].copy()
    d["y"] = np.log(d.move_time + LOG_OFFSET)
    g = d.groupby("player")["y"]
    d["z"] = (d.y - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return d.dropna(subset=["z"])


def calipers(d, sd_mult=CALIPER_SD, vars_=MATCH_VARS):
    """Austin(2011): 각 변수 SD의 0.2배"""
    return {v: sd_mult * d[v].std() for v in vars_ if v in d.columns}


def events_A(d, thresh=BLUNDER_THRESH, wp_lo=WP_LO, wp_hi=WP_HI):
    """블런더. 진 쪽은 정의상 자동 배제(실측 0건), 상한을 대칭 적용"""
    m = d.wp_delta.notna() & (d.wp_delta <= -thresh)
    m &= d.wp_before.between(wp_lo, wp_hi)
    return d[m]


def mask_A(d, thresh=BLUNDER_THRESH, wp_lo=WP_LO, wp_hi=WP_HI):
    """events_A 와 같은 조건의 불리언 마스크 (집단 분할용)"""
    return (d.wp_delta.notna() & (d.wp_delta <= -thresh)
            & d.wp_before.between(wp_lo, wp_hi))


def events_B(d, min_loss=1):
    """
    재료 순손실 — 논문 본문의 사건 B (사전등록 라벨로는 C).

    net_mat 은 prespeed.net_material_change 가 mat_diff 에서 만든다.
    등가 교환은 net_mat == 0 이므로 자동으로 빠진다.
    """
    _require_net_mat(d)
    return d[d.net_mat.notna() & (d.net_mat <= -min_loss)]


def mask_B(d, min_loss=1):
    _require_net_mat(d)
    return d.net_mat.notna() & (d.net_mat <= -min_loss)


def events_B_see(d, min_value=1):
    """
    사전등록판 사건 C = 논문의 사건 B (SEE > 0). **철회된 정의다.**

    등가 교환을 배제하지 못해 사건의 55.7%가 순변화 0이었다.
    부록 비교를 재생성할 때만 쓴다. docs/definitions.md 참조.
    """
    if "see_loss" not in d.columns:
        raise KeyError("see_loss 열이 없다. stage2v2 를 do_see=True 로 돌렸는가.")
    return d[d.see_loss.notna() & (d.see_loss >= min_value)]


def _require_net_mat(d):
    if "net_mat" not in d.columns:
        raise KeyError(
            "net_mat 열이 없다. scripts/run_prep.py 로 분석 테이블을 먼저 만들 것. "
            "사전등록판 SEE 정의가 필요하면 events_B_see() 를 쓴다.")


def build_and_match(d, events, cal, window=EPOCH_WINDOW, seed=0,
                    max_events=None, strict=True):
    """
    1:N 매칭 — 캘리퍼 조건을 만족하는 모든 후보의 창 z를 평균.

    후보 상한을 두지 않는다. 상한 50과 비교한 결과 처리 시간이 동일하고
    (병목은 후보 필터링이지 창 조회가 아님) 효과 차이도 1e-5 수준이었다.
    상한이 없으면 무작위 선택이 사라져 재현성도 개선된다.
    """
    if len(events) == 0:
        return pd.DataFrame(columns=[
            "player", "tier", "game_id", "ply", "z_post", "z_ctrl",
            "n_obs", "n_ctrl", "clk_before", "wp_before", "size", "effect"])
    if max_events and len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    players = set(events.player.unique())
    d = d[d.player.isin(players)]
    zmap = build_zmap(d)
    by_player = {p: g for p, g in control_pool(d, events, cal).groupby("player")}

    recs = []
    for r in events.itertuples():
        zp, nobs = window_mean(zmap, r.game_id, r.player, r.ply, window, +1)
        g = by_player.get(r.player)
        zc, ncand = np.nan, 0
        if g is not None and len(g):
            m, ok = caliper_mask(r, g, cal, strict=strict)
            if ok:
                c = g[m]
                ncand = len(c)
                if ncand:
                    zs = [window_mean(zmap, x.game_id, x.player, x.ply,
                                      window, +1)[0] for x in c.itertuples()]
                    zs = [v for v in zs if not np.isnan(v)]
                    if zs:
                        zc = float(np.mean(zs))
        recs.append({
            "player": r.player, "tier": r.tier, "game_id": r.game_id, "ply": r.ply,
            "z_post": zp, "z_ctrl": zc, "n_obs": nobs, "n_ctrl": ncand,
            "clk_before": getattr(r, "clk_before", np.nan),
            "wp_before": getattr(r, "wp_before", np.nan),
            "size": getattr(r, "net_mat", np.nan),
        })
    e = pd.DataFrame(recs)
    e["effect"] = e.z_post - e.z_ctrl
    return e


def reliability(epochs, min_events=10):
    """홀짝 게임 분할 신뢰도 + Spearman-Brown, 분산 성분"""
    e = epochs.dropna(subset=["effect"]).copy()
    if len(e) < 50:
        return None
    e["half"] = pd.factorize(e.game_id)[0] % 2
    piv = e.groupby(["player", "half"])["effect"].agg(["mean", "size"]).unstack()
    ok = (piv["size"].fillna(0) >= min_events / 2).all(axis=1)
    piv = piv[ok]

    per = e.groupby("player")["effect"].agg(["mean", "var", "size"])
    per = per[per["size"] >= min_events]
    out = {"n_players_split": len(piv), "n_players_var": len(per)}
    if len(piv) >= 20:
        a, b = piv["mean"][0], piv["mean"][1]
        r = np.corrcoef(a, b)[0, 1]
        out["r"] = r
        out["sb"] = 2 * r / (1 + r) if r > -1 else np.nan
    if len(per) >= 20:
        var_w = per["var"].mean()
        sb2 = max(per["mean"].var() - var_w / per["size"].mean(), 0)
        out.update(sigma_b=np.sqrt(sb2), sigma_w=np.sqrt(var_w),
                   mean_events=per["size"].mean())
    return out


def moderation_by_time(epochs, bins=(0, 30, 60, 120, 180, 300, 600)):
    """
    잔여 시간 구간별 효과 — 시간 압박은 사건이 아니라 맥락.

    주의: 30초 미만은 산술적 제약이 강하므로 해석에서 분리할 것.
    SE 는 **플레이어 클러스터**다. 예전 구현은 사건 단위 .sem() 이었는데,
    같은 플레이어의 사건끼리 상관되므로 과소추정된다.
    """
    e = epochs.dropna(subset=["effect"]).copy()
    e["clk_bin"] = pd.cut(e.clk_before, bins)
    rows = []
    for b, g in e.groupby("clk_bin", observed=True):
        st = cluster_stats(g, "effect")
        if st is None:
            continue
        rows.append({"clk_bin": b, **st})
    return pd.DataFrame(rows).set_index("clk_bin")
