"""
Manuscript values vs repository output

Every value the manuscript reports is transcribed here, with the section it
came from, and compared mechanically against `data/derived/` and
`src/external_values.py`. Run it with:

    python run.py verify-paper

Why this exists
    A manuscript and its repository drift apart easily. In the pre-release
    review the effects agreed but every standard error was out by a factor of
    1.6, and that was found by reading the two side by side. A comparison like
    that should not depend on someone noticing.

Transcription rules
    - Record which section, table, or figure of the manuscript each value
      came from.
    - When the manuscript changes, change this file. A disagreement between
      the two is itself the bug.
    - Set tol slightly wider than the rounding in the manuscript.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE
#   - Check the constants below against the manuscript word for word. If this
#     file is wrong, the whole check is meaningless.
#   - A passing check does not mean the analysis is correct. It means the
#     numbers in the manuscript and the numbers this code produces agree.
#   - Failures come in three kinds:
#       DATA   the repository output needs regenerating
#       PAPER  the manuscript needs correcting
#       BOTH   someone has to decide which is right
#     The decision is a person's. This file only shows the difference.
#   - The final section compares derived files against ONE ANOTHER rather
#     than against the manuscript. That is what catches two files produced by
#     different runs; comparison against the manuscript alone does not.
# ───────────────────────────────────────────────────────────

# Transcribed from manuscript v14 (five-variable matching, five tiers).
#
# Values are read off the manuscript, never copied from data/derived. A
# transcription taken from the repository would agree with it by construction
# and could not catch a number mistyped on its way into the paper.

import os

import numpy as np
import pandas as pd

from .config import DERIVED, SAMPLE, TIERS, TITLE_TIER
from .config import BLUNDER_THRESH as BLUNDER_THRESH_PP

# ══════════════════════════════════════════════════════════════
# 논문 전사값
# ══════════════════════════════════════════════════════════════

# Table 1 — 표본 구성 (tier: 전체, ≥30판, 통과율%, 표집)
TABLE1 = {
    "1300-1600": (201_949, 41_882, 20.7, 500),
    "1600-1900": (148_391, 40_643, 27.4, 500),
    "1900-2100": (54_140, 15_106, 27.9, 500),
    "2100+":     (27_189, 5_764, 21.2, 500),
    TITLE_TIER:  (712, 136, 19.1, 136),
}
N_PLAYERS = 2_136          # Abstract, Results
# Title: "Evidence From 3.9 Million Moves".
#
# ★ The analysis now spans five tiers, so this has to be the two prepared
#   tables together: 3,390,227 untitled plus 553,312 titled is 3,943,539.
#   The check used to read prepared.parquet alone and so agreed with the old
#   title while the paper had grown past it.
N_MOVES_TITLE = 3_943_539

# Results -- event counts.
#
# ★ Two populations are in play and the manuscript mixes them.
#
#   N_PLAYERS (2,136) is how many players were *sampled*. The analysis tables
#   hold 2,014: 122 of the sampled players have no game in the 208 shards that
#   were extracted.
#
#   The event counts below imply about 211,000 games (887,287 / 4.21), which
#   is roughly 397 shards. The analysis tables hold 129,791 games. So the
#   counts describe the full January scan while every effect comes from the
#   208-shard subset -- the same split that produced the standard-error
#   mismatch and the title count.
#
#   Events per game agrees (4.21 against 4.19), so the subset is
#   representative; it is the absolute counts that belong to another run.
N_EVENTS_MATERIAL = 543_987     # 순손실 정의, 두 파일 합산
N_EVENTS_BLUNDER = 88_548
EVENTS_PER_GAME = 4.19
N_PLAYERS_ANALYSED = 2_014      # rows in prepared + prepared_fm
N_GAMES_ANALYSED = 129_791
# The withdrawn SEE criterion on the same tables. Nothing checked these
# four until now -- they sat in the file as constants with no comparison
# attached, which is how a stale value survives even when the data is present.
N_EVENTS_MATERIAL_REGISTERED = 975_871

# Method — 개정 검증 (원정의 사건 중)
REVISION_SHARE_ZERO = 54.1      # %
REVISION_SHARE_GAIN = 12.3      # %
# The verification subsample is gone: the shares are computed over all
# 975,871 registered-criterion events now, which is the stronger claim.

# Results, "Effects by Co-occurrence of Material Loss" / Fig 1c / Fig 2f
#   label: (효과, SE, t)
# label: (효과, SE, t, 플레이어 수)
# 2026-09 개정판에서 플레이어 수가 추가되었다. SE 와 함께 보면 클러스터
# 표준편차가 역산되므로, 저장소 산출물이 같은 실행인지 바로 확인할 수 있다.
GROUPS = {
    "blunder_only": (+0.209, 0.017, 12.3, 1_629),
    "blunder_loss": (-0.407, 0.017, -23.4, 1_642),
    "loss_only":    (-0.537, 0.015, -35.3, 1_781),
}
# 같은 절 — 집단 간 차이.
#
# 방향: **blunder_only − blunder_loss**.
# 원고 v6 이 본문에 이 방향을 명시한다 ("Taking the difference as
# blunder-only minus blunder-with-material-loss"). v5 까지는 방향이 적혀
# 있지 않아 값을 옮길 때마다 부호가 뒤집혔고, 실제로 한 문장 안에 두
# 방향이 섞인 적이 있다. 아래 GROUP_DIFF_ORDER 가 계산 방향을 고정한다.
GROUP_DIFF_ORDER = ("blunder_only", "blunder_loss")
GROUP_DIFF = {-3: -0.001, -2: +0.010, -1: -0.018,
              +1: +0.616, +2: +0.018, +3: +0.048}
GROUP_DIFF_SE_T1 = 0.024
EVENT_MOVE_BLUNDER_LOSS = +0.486

# Results, "When the Effect Occurs" / Fig 1a·1b — 다섯 층의 범위
# Results, tier-wise profiles. Five tiers now, and blunders changed sign:
# the manuscript reports "slower than control in every tier".
#
# ★ These come from lag_profiles.csv, which Figure 1 also draws. se_tier.csv
#   measures the same quantity under a smaller event cap (16,000 against
#   25,000) and reads about 0.01 lower. Both are correct; the manuscript
#   quotes the figure's source.
TIER_RANGES = {
    "material_t+1": (-0.563, -0.490),
    # t+2 crosses zero in the titled tier (+0.008), so the manuscript
    # dropped its "a seventh to a twentieth of t+1" ratio and speaks of
    # absolute size instead.
    "material_t+2": (-0.030, +0.008),
    "material_t+3": (-0.038, -0.013),
    "material_pre": (-0.019, +0.037),   # t−3 ~ t−1
    "blunder_t0":   (+0.279, +0.372),
    "blunder_t+1":  (+0.059, +0.078),
}
TIER_RANGE_WIDTH = {"material": 0.073, "blunder": 0.019}
# "all |t| > 32"
TIER_MATERIAL_T_MIN = 32

# 원고 v6 은 층별 lag 을 "the four untitled tiers" 로 서술한다.
# 타이틀 층은 더 긴 기간에서 뽑혀 별도 보고된다 (Table 1·2 는 다섯 층).
TIER_COUNT_IN_TEXT = 5

# Results, "Magnitude of Material Lost" / "Direction of loss" / Fig 2c
DOSE_SELF = {1: (-0.301, 0.015), 9: (-0.979, 0.016)}
DOSE_RANGE = (-1.006, -0.520)       # "all |t| > 25"
DOSE_T_MIN = 25
EVENT_MOVE_DOSE = {"self": +0.307, "opp": -0.455}
# The nine-point gain is "indistinguishable from zero" in the manuscript.
DOSE_OPP = {1: (+0.219, 0.014), 9: (+0.027, 0.016)}

# Results, "Number of available options" / Fig 2b
LEGAL_BIN_DIFF_RANGE = (-0.693, -0.382)
LEGAL_BIN_DIFF_FLAT = -0.689     # 합법수가 거의 안 변하는 구간
# "Restricting to nine-point losses ... widens the separation to −1.04"
# 이 값은 9점 제한 실행에서 나온 것으로 legal_bins.csv 에 없다 — 대조 불가.
LEGAL_BIN_NINE_POINT_MAX = -0.990

# Table 2 — 개인 수준 신뢰도 (신뢰도, sigma_b, 인당 사건 중앙값, 인원)
# v8 에서 두 행으로 줄었다 (층 합침). 선별 규칙이 Method 에 명시됐다.
# Table 2, four rows. The manuscript labels the untitled rows "untitled";
# the CSV labels them "lower".
RELIABILITY = {
    "Material loss, lower":  (+0.685, 0.115, 198, 607),
    "Blunder, lower":        (+0.103, 0.048, 34, 431),
    "Material loss, titled": (+0.739, 0.122, 173, 62),
    "Blunder, titled":       (-0.032, 0.051, 32, 122),
}
# Method / Table 2 캡션이 말하는 선별 기준
RELIABILITY_THRESHOLD = {"Material loss, lower": 237, "Blunder, lower": 40,
                         "Material loss, titled": 237, "Blunder, titled": 40}
# Results — 관측수별 신뢰도
# Results, "Reliability depended on the number of events retained per player"
# The manuscript quotes 50 events and above: below that the truncated
# sample changes composition and the curve is not monotone (.263 at 10,
# .078 at 20, .335 at 30).
RELIABILITY_CURVE = {50: 0.352, 60: 0.366, 80: 0.441,
                     100: 0.546, 150: 0.631, 200: 0.738}
RELIABILITY_CURVE_TITLED = {150: 0.828}
# Blunders could not be traced beyond 30 events per player.
RELIABILITY_CURVE_BLUNDER = {10: -0.039, 20: -0.036, 30: -0.051}
# "crossing the conventional .70 threshold between 150 and 237 events"
# "crossing the conventional .70 threshold between 150 and 200 events"
RELIABILITY_CROSSING = (150, 200)
RELIABILITY_THRESHOLD_VALUE = 0.70
# Discussion — 같은 관측 수에서의 대비
# The manuscript no longer contrasts the two at matched counts:
# blunders stop at 30 events and material loss starts at 50.
CONTRAST_AT = {}   # (블런더, 기물손실)
# "At matched observation counts the two event types differ by roughly .17."
CONTRAST_ROUGHLY = 0.17
# Abstract 가 인용하는 값
ABSTRACT_RELIABILITY = 0.685
FLANKER_CURVE = {10: 0.233, 30: 0.358, 50: 0.544, 80: 0.676}

# Results, "Comparison With a Laboratory Task" / Fig 2g
FLANKER = {
    "post_error_slowing": 0.131,
    "err_rate_after_correct": 10.2,
    "err_rate_after_error": 21.4,
    "n_trials": 91_741,
    "n_trials_post_error": 91_569,   # 각 참가자 첫 시행 제외
    "error_rate": 11.4,
    "n_participants": 172,
}

# Results, "Robustness" — 사전등록 여섯 사양
# ★ 이 값들은 `run.py robustness` 의 산출물이다. 그 명령은 원래 저장소에
#   없어서 재구성한 것이고, 원 분석의 강건성 코드는 남아 있지 않다.
#   Method 가 이 사양을 기술하는지 확인할 것. docs/corrections.md 참조.
#
# level 은 robustness.csv 의 표기를 그대로 쓴다 (문자열 비교).
# Baseline for these is -0.513: all Event B occurrences, not the
# material-loss-only subset (-0.537).
ROBUSTNESS = {
    ("opening_cut", "10"): -0.513, ("opening_cut", "14"): -0.513,
    ("opening_cut", "20"): -0.534,
    ("measurement_floor", "1"): -0.513,
    ("measurement_floor", "2"): -0.515,
    ("measurement_floor", "3"): -0.452,
    # Positive and significant at every threshold under this specification.
    ("blunder_threshold", "0.1"): +0.066,
    ("blunder_threshold", "0.2"): +0.063,
    ("blunder_threshold", "0.3"): +0.078,
}
# The registered six-variable caliper against the five used here, on the
# subsample where both exist. Its own baseline, -0.560, is not the -0.513
# above: a different sample.
CALIPER_VARS = {"5 vars (main)": -0.560, "6 vars (registered)": -0.587}
CALIPER_VARS_MAX_GAP = 0.04
# "Matching caliper produced values from −0.527 to −0.568 across 0.1 to 0.8 SD"
CALIPER_RANGE = (-0.525, -0.493)
# "removing the pre-event speed control changed the estimate from
#  −0.568 to −0.562"
# The pre-event speed comparison and the complexity control moved to the
# supplementary material: they use a three-move window, so their values do
# not sit beside the single-move numbers here.

# Discussion — 구성 논증
#   "Weighting +0.121 and −0.567 by their observed frequencies gives −0.053"
# "Weighting +0.209 and -0.407 by their observed frequencies gives +0.055
#  at the 10-point threshold, close to the +0.066 observed."
#
# The manuscript rests this on the level alone. The trend does not follow:
# a higher threshold should raise the share that costs material and pull the
# average down, but the observed values are flat or rising.
COMPOSITION = {"share_loss_10pp": 0.25, "share_loss_30pp": 0.31,
               "predicted_10pp": +0.055, "observed_10pp": +0.066}

# "Restricting to nine-point losses ... −1.04 at its maximum"
QUEEN_BIN_MAX = -0.990
# 원고는 "Raising the measurement floor to 3 s reduced the effect by 18%"
# 라고 쓴다. 아래 값들에서 계산한 감쇠율이 그것과 맞는지 확인한다.
# (전체 표본 robustness 실행 전까지는 원고 값을 그대로 둔다)
ROBUSTNESS_TAU3_ATTENUATION = 0.12

# Results, "Mixed-effects models"
#   "−0.175 with the win-probability covariate and −0.230 without it;
#    the former is estimated on the 20% of matched events for which engine
#    evaluation exists."
#   "The paired-difference estimate agrees with each on its own subset:
#    −0.173 on the events with engine evaluation and −0.229 on all matched
#    events. The two approaches therefore differ by less than 0.003 once the
#    sample is held constant."
MIXED = {
    "mixed model (main)": -0.513,
    "paired difference (main)": -0.513,
    "mixed model (with wp, engine-eval subset)": -0.596,
    "paired difference (with wp, engine-eval subset)": -0.593,
}
MIXED_SE = {"mixed model (main)": 0.008,
            "paired difference (main)": 0.015,
            "mixed model (with wp, engine-eval subset)": 0.014,
            "paired difference (with wp, engine-eval subset)": 0.021}
MIXED_MAIN_PLAYERS = 1_724
# "The model with random slopes did not converge, so we report the
#  random-intercept specification."
MIXED_MAIN_PATH = "random intercept (fallback 1)"
# 같은 표본에서 두 접근이 얼마나 가까운가
MIXED_VS_PAIRED_MAX = 0.005
MIXED_PAIRS = [("mixed model (main)", "paired difference (main)"),
               ("mixed model (with wp, engine-eval subset)",
                "paired difference (with wp, engine-eval subset)")]

TOL = 0.002          # 효과·SE 기본 허용 오차
TOL_RANGE = 0.003    # 범위 끝값


# ══════════════════════════════════════════════════════════════
# 대조
# ══════════════════════════════════════════════════════════════

class Report:
    def __init__(self):
        self.ok, self.bad, self.note = [], [], []

    def check(self, label, paper, repo, tol=TOL, kind="DATA", names=None):
        """
        names lets a comparison say what the two sides actually are.

        Most rows compare the manuscript against the repository, so the
        default reads "paper" and "repo". The cross-file section at the end
        compares two derived files against each other, where those words
        would be wrong.
        """
        lhs, rhs = names or ("paper", "repo")
        if repo is None or (isinstance(repo, float) and np.isnan(repo)):
            self.note.append(
                f"{label}: no corresponding value in the repository "
                f"({lhs} {paper})")
            return
        d = abs(paper - repo)
        line = (f"{label:<46} {lhs} {paper:>9.4f}  {rhs} {repo:>9.4f}  "
                f"diff {d:.4f}")
        (self.ok if d <= tol else self.bad).append((kind, line))

    def check_range(self, label, paper_range, vals, tol=TOL_RANGE, kind="DATA"):
        lo, hi = min(vals), max(vals)
        plo, phi = paper_range
        d = max(abs(plo - lo), abs(phi - hi))
        line = (f"{label:<46} paper [{plo:+.3f}, {phi:+.3f}]  "
                f"repo [{lo:+.3f}, {hi:+.3f}]")
        (self.ok if d <= tol else self.bad).append((kind, line))

    def remark(self, msg):
        self.note.append(msg)


def _read(name):
    p = os.path.join(DERIVED, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def run(verbose=True):
    try:
        from . import external_values as FD
    except ImportError:
        FD = None

    r = Report()

    # ── 표본 ────────────────────────────────────────────────
    for tier, (tot, elig, rate, n) in TABLE1.items():
        r.check(f"Table 1 eligibility rate {tier}", rate, 100 * elig / tot,
                tol=0.06, kind="PAPER")
    if os.path.exists(SAMPLE):
        samp = pd.read_parquet(SAMPLE)
        r.check("sample size, total", N_PLAYERS, len(samp), tol=0.5)
        vc = samp.tier.value_counts().to_dict()
        for tier, (_, _, _, n) in TABLE1.items():
            r.check(f"sampled players {tier}", n, vc.get(tier, 0), tol=0.5)
    else:
        r.remark(f"{SAMPLE} is absent - skipping the sample comparison")

    # ── 세 집단 (주 결과) ───────────────────────────────────
    sp = _read("se_split.csv")
    pp = _read("per_player_t1.csv")
    if sp is not None:
        if "np1" not in sp.columns:
            r.remark("se_split.csv has no np (cluster count) column - it is not the "
                     "output of the current code. Regenerate with python run.py se split.")
        for g, (pe, pse, pt, pn) in GROUPS.items():
            row = sp[sp.label == g]
            if row.empty:
                r.remark(f"se_split.csv has no row for {g}")
                continue
            row = row.iloc[0]
            r.check(f"effect at t+1, {g}", pe, float(row.e1), tol=0.001)
            r.check(f"SE   {g} t+1", pse, float(row.se1))
            r.check(f"internal consistency of t, {g}", pt, pe / pse, tol=0.6, kind="PAPER")
            if "np1" in sp.columns and not pd.isna(row.get("np1")):
                r.check(f"cluster count, {g}", pn, float(row["np1"]), tol=1)
        lhs, rhs = GROUP_DIFF_ORDER
        for k, pv in GROUP_DIFF.items():
            a = sp[sp.label == lhs]
            b = sp[sp.label == rhs]
            if a.empty or b.empty:
                continue
            r.check(f"group difference t{k:+d} ({lhs} minus {rhs})", pv,
                    float(a.iloc[0][f"e{k}"]) - float(b.iloc[0][f"e{k}"]),
                    tol=0.004)
        a = sp[sp.label == "blunder_loss"]
        if not a.empty:
            r.check("event count, blunder_loss", EVENT_MOVE_BLUNDER_LOSS,
                    float(a.iloc[0]["e0"]))
    else:
        r.remark("se_split.csv is absent - skipping the main-result comparison")

    _f = {k[1]: v for k, v in ROBUSTNESS.items()
          if k[0] == "measurement_floor"}
    _a = (1 - _f[max(_f)] / _f[min(_f)]) if len(_f) >= 2 else float("nan")
    r.check("measurement-floor attenuation (within paper)", ROBUSTNESS_TAU3_ATTENUATION, _a,
            tol=0.01, kind="PAPER")

    r.check("SE of the group difference at t+1 (within paper)", GROUP_DIFF_SE_T1,
            float(np.hypot(GROUPS["blunder_loss"][1], GROUPS["blunder_only"][1])),
            tol=0.001, kind="PAPER")

    # 논문이 보고한 (SE, n) 이 함축하는 플레이어간 표준편차.
    # 저장소 산출물과 견주면 같은 실행인지 즉시 드러난다.
    for g, (pe, pse, pt, pn) in GROUPS.items():
        r.remark(f"{g}: paper SE {pse:.3f} x sqrt({pn:,}) => between-player SD "
                 f"{pse * np.sqrt(pn):.3f}")

    if sp is not None and pp is not None:
        for g in GROUPS:
            v = pp[pp.group == g].eff
            row = sp[sp.label == g]
            if v.empty or row.empty:
                continue
            if abs(float(v.mean()) - float(row.iloc[0].e1)) > 0.02:
                r.remark(f"{g}: per_player_t1 mean {v.mean():+.3f} and "
                         f"se_split e1 {float(row.iloc[0].e1):+.3f} disagree - "
                         f"the two files came from different runs")

    # ── 층별 범위 ───────────────────────────────────────────
    # 층별 프로파일: lag_profiles.csv 가 층별 행을 가지고 있으면 그것을 쓰고,
    # 없으면 src/external_values.py 의 전사값으로 떨어진다.
    lp_t = _read("lag_profiles.csv")
    M = B = None
    if lp_t is not None and {"event_type", "tier", "lag", "effect"} <= set(lp_t.columns):
        # The manuscript describes five tiers, so the titled one belongs in
        # the range. "lower" is an aggregate row, not a tier.
        tiers = [t for t in lp_t.tier.unique() if t != "lower"]
        if tiers:
            LK = [-3, -2, -1, 0, 1, 2, 3]
            # lag_profiles.csv carries three control specifications per
            # tier. Without this filter .loc returns three rows per lag and
            # float() fails on the Series.
            SPEC = "zpre"
            def _rows(ev):
                out = []
                for t in tiers:
                    g = lp_t[(lp_t.event_type == ev) & (lp_t.tier == t)
                             & (lp_t.spec == SPEC)].set_index("lag")
                    if not set(LK) <= set(g.index):
                        return None
                    out.append([float(g.loc[k, "effect"]) for k in LK])
                return out
            M, B = _rows("material"), _rows("blunder")
            if M and B:
                r.remark(f"tier ranges computed from the {len(tiers)} tiers in lag_profiles.csv"
                         f": {', '.join(map(str, tiers))}")
    if M is None and FD is not None:
        M, B = FD.MATERIAL_BY_TIER["rows"], FD.BLUNDER_BY_TIER["rows"]
        r.remark("tier ranges read from transcribed values in external_values - "
                 "lag_profiles.csv has no per-tier rows")

    if FD is not None:
        col = lambda rows, i: [x[i] for x in rows]
        r.check_range("tier range, material t+1", TIER_RANGES["material_t+1"], col(M, 4))
        if len(M) != TIER_COUNT_IN_TEXT:
            r.remark(f"the tier profile has {len(M)} tiers but the manuscript describes "
                     f"{TIER_COUNT_IN_TEXT}.")
        r.check_range("tier range, material t+2", TIER_RANGES["material_t+2"], col(M, 5))
        r.check_range("tier range, material t+3", TIER_RANGES["material_t+3"], col(M, 6))
        r.check_range("tier range, material pre-event", TIER_RANGES["material_pre"],
                      col(M, 0) + col(M, 1) + col(M, 2))
        r.check_range("tier range, blunder at event", TIER_RANGES["blunder_t0"], col(B, 3))
        r.check_range("tier range, blunder t+1", TIER_RANGES["blunder_t+1"], col(B, 4))

        lp_t = _read("lag_profiles.csv")
        if lp_t is not None and {"t", "lag", "event_type"} <= set(lp_t.columns):
            # Without the spec and tier filters this swept in the other
            # control specifications and the "lower" aggregate row, and
            # reported their smallest |t| instead of the five tiers'.
            g = lp_t[(lp_t.event_type == "material") & (lp_t.lag == 1)
                     & (lp_t.spec == "zpre") & (lp_t.tier != "lower")]
            if not g.empty:
                got = float(g.t.abs().min())
                line_ok = got >= TIER_MATERIAL_T_MIN
                (r.ok if line_ok else r.bad).append(
                    ("DATA", f"{'tier material t+1, smallest |t|':<46} "
                             f"paper >{TIER_MATERIAL_T_MIN}      repo {got:>9.1f}"))

        # 합법수 구간 — legal_bins.csv 가 있으면 그것을, 없으면 전사값을.
        lb = _read("legal_bins.csv")
        if lb is not None and {"label", "bin", "effect"} <= set(lb.columns):
            piv = lb.pivot_table(index="bin", columns="label", values="effect")
            if {"self_lost", "opp_lost"} <= set(piv.columns):
                mids = lb.drop_duplicates("bin").set_index("bin")["bin_mid"]
                piv = piv.loc[mids.sort_values().index]
                d = (piv["self_lost"] - piv["opp_lost"]).tolist()
                r.remark(f"legal-move bin differences computed from legal_bins.csv "
                         f"({len(d)} bins)")
            else:
                d = None
        else:
            d = None
        if d is None:
            d = [x - y for x, y in zip(FD.LEGAL_MOVE_BINS["self_lost"],
                                       FD.LEGAL_MOVE_BINS["opp_lost"])]
            r.remark("legal-move bin differences read from transcribed values in external_values")
        r.check_range("legal-move bin difference range", LEGAL_BIN_DIFF_RANGE, d, tol=0.006)
        flat = d[len(d) // 2] if len(d) >= 5 else d[0]
        r.check("legal-move flat bin", LEGAL_BIN_DIFF_FLAT, flat, tol=0.006)

        # Table 2 — reliability.csv 에 네 층 행이 있으면 그것을 본다.
        rl = _read("reliability.csv")
        T2_KEYS = list(RELIABILITY)          # 원고 Table 2 의 행 이름
        csv_t2 = None
        if rl is not None and "label" in rl.columns:
            have = set(rl.label)
            if set(T2_KEYS) <= have:
                csv_t2 = rl.set_index("label")
            else:
                r.remark(
                    f"reliability.csv lacks the Table 2 rows "
                    f"(present: {', '.join(sorted(have))}). "
                    f"Table 2 cannot be compared.")
                for lab in sorted(have):
                    row = rl[rl.label == lab].iloc[0]
                    r.remark(f"  {lab}: reliability {row.reliability:+.3f}, "
                             f"sigma_b {row.sigma_b:.3f}, "
                             f"{int(row.n_players)} players, "
                             f"{row.events_per_player:.0f} events each")
        for key, (pv, psig, pev, pn) in RELIABILITY.items():
            if csv_t2 is None or key not in csv_t2.index:
                continue
            row = csv_t2.loc[key]
            r.check(f"reliability, {key}", pv, float(row["reliability"]), tol=0.002)
            r.check(f"sigma_b {key}", psig, float(row["sigma_b"]), tol=0.002)
            r.check(f"events per player, {key}", pev, float(row["events_per_player"]),
                    tol=1)
            r.check(f"player count, {key}", pn, float(row["n_players"]), tol=1)
            # v9 의 Method 는 선별이 **매칭 전** 사건 수이고 매칭이 일부를
            # 걸러내 중앙값이 기준보다 낮아진다고 설명한다. 그 설명이
            # 매칭 잔존율(원고의 37~39% 제외 = 61~63% 잔존)과 맞는지 본다.
            thr = RELIABILITY_THRESHOLD.get(key)
            if thr:
                keep = float(row["events_per_player"]) / thr
                if not (0.55 <= keep <= 0.70):
                    r.note.append(
                        f"retention after matching, {key}: threshold {thr} "
                        f"-> median {float(row['events_per_player']):.0f} "
                        f"({keep:.0%} kept). The manuscript states only that "
                        f"the median falls below the threshold.")

        # 관측수별 곡선 — reliability_curve.csv 우선
        rc = _read("reliability_curve.csv")
        Rc = None
        if rc is not None and {"label", "n_events_per_player", "sb"} <= set(rc.columns):
            # startswith("Material") matches both the untitled and the titled
            # row, and dict(zip(...)) then keeps whichever comes last -- the
            # titled one. The manuscript quotes the untitled curve, so the
            # check was comparing two different curves and reporting the
            # difference as an error.
            g = rc[rc.label.str.contains("lower|untitled", case=False)]
            g = g[g.label.str.startswith("Material")].dropna(subset=["sb"])
            if not g.empty:
                Rc = dict(zip(g.n_events_per_player.astype(int), g.sb))
                r.remark("reliability curve read from reliability_curve.csv "
                         "(untitled rows)")
        if Rc is None:
            Rc = dict(zip(FD.RELIABILITY_CHESS["n"], FD.RELIABILITY_CHESS["sb"]))
            r.remark("reliability curve read from transcribed values in external_values")
        for n, v in RELIABILITY_CURVE.items():
            r.check(f"reliability curve n={n}", v, Rc.get(n), tol=0.002)

        # Discussion 의 대비 — v8 이 인용하는 네 값
        if rc is not None and "label" in rc.columns:
            gb = rc[rc.label.str.contains("lower|untitled", case=False)]
            gb = gb[gb.label.str.startswith("Blunder")]
            bl = dict(zip(gb.n_events_per_player.astype(int), gb.sb))
            for n, (pb, pm) in CONTRAST_AT.items():
                r.check(f"contrast, blunder n={n}", pb, bl.get(n), tol=0.002)
                r.check(f"contrast, material loss n={n}", pm, Rc.get(n), tol=0.002)
            for n, v in RELIABILITY_CURVE_BLUNDER.items():
                r.check(f"blunder curve n={n}", v, bl.get(n), tol=0.002)

            # "gap between the two event types" — 가장 큰 공통 관측수에서 본다
            common = sorted(set(bl) & set(Rc))
            if common:
                n = common[-1]
                r.check(f"contrast magnitude (n={n})", CONTRAST_ROUGHLY,
                        float(Rc[n]) - float(bl[n]), tol=0.03)

        # "interval where .70 is crossed" — 곡선에서 실제로 어디서 넘는지
        if Rc:
            ns = sorted(Rc)
            over = [n for n in ns if Rc[n] >= RELIABILITY_THRESHOLD_VALUE]
            if over:
                first = over[0]
                prev = max([n for n in ns if n < first], default=None)
                lo, hi = RELIABILITY_CROSSING
                if not (prev is not None and prev == lo and first == hi):
                    r.bad.append(("PAPER",
                        f"{'.70 crossing interval':<46} "
                        f"paper {lo}-{hi}      repo "
                        f"{prev}~{first} ({Rc[first]:.3f} at {first})"))

        # Abstract 가 인용하는 값이 Table 2 와 같은지
        mt = "Material loss, lower"
        if csv_t2 is not None and mt in csv_t2.index:
            r.check("reliability quoted in the Abstract", ABSTRACT_RELIABILITY,
                    float(csv_t2.loc[mt, "reliability"]), tol=0.002,
                    kind="PAPER")
        Rf = dict(zip(FD.RELIABILITY_FLANKER["n"], FD.RELIABILITY_FLANKER["sb"]))
        for n, v in FLANKER_CURVE.items():
            r.check(f"flanker reliability n={n}", v, Rf.get(n), tol=0.001)

        F = FD.FLANKER
        r.check("flanker post-error slowing", FLANKER["post_error_slowing"],
                F["post_error_slowing"], tol=0.001)
        r.check("flanker error rate after correct", FLANKER["err_rate_after_correct"],
                F["err_rate_after_correct"], tol=0.05)
        r.check("flanker error rate after error", FLANKER["err_rate_after_error"],
                F["err_rate_after_error"], tol=0.05)
        # 원고는 전체 시행수(91,741)와 사후분석 시행수(91,569)를 둘 다 적는다.
        # external_values 의 합계는 후자와 맞아야 한다.
        trials = F["n_correct"] + F["n_error"]
        r.check("flanker post-error trial count", FLANKER["n_trials_post_error"],
                float(trials), tol=0.5)
        gap = FLANKER["n_trials"] - FLANKER["n_trials_post_error"]
        if gap != FLANKER["n_participants"]:
            r.remark(f"flanker: total {FLANKER['n_trials']:,} minus post-error "
                     f"{FLANKER['n_trials_post_error']:,} = {gap}, but there are "
                     f"{FLANKER['n_participants']} participants. If one trial per "
                     f"participant is dropped, the two should match.")

        for i, v in enumerate(FD.DOSE["x"]):
            if v in DOSE_SELF:
                r.check(f"self_{v} (external_values)", DOSE_SELF[v][0],
                        FD.DOSE["self"][i], tol=0.001)
    else:
        r.remark("src/external_values.py could not be read - skipping external reference values")

    # ── dose ────────────────────────────────────────────────
    dose = _read("se_dose.csv")
    if dose is not None:
        for v, (pe, pse) in DOSE_SELF.items():
            row = dose[dose.label == f"self_{v}"]
            if row.empty:
                continue
            r.check(f"effect, self_{v}", pe, float(row.iloc[0].e1), tol=0.003)
            r.check(f"SE   self_{v}", pse, float(row.iloc[0].se1))
        for v, (pe, pse) in DOSE_OPP.items():
            row = dose[dose.label == f"opp_{v}"]
            if row.empty:
                continue
            r.check(f"effect, opp_{v}", pe, float(row.iloc[0].e1), tol=0.003)
    else:
        r.remark("se_dose.csv is absent - skipping the dose comparison")

    # ── 강건성 ─────────────────────────────────────────────
    rb = _read("robustness.csv")
    if rb is not None and not rb.empty:
        rb = rb.copy()
        rb["level"] = rb["level"].astype(str)
        for (spec, level), pv in ROBUSTNESS.items():
            row = rb[(rb.spec == spec) & (rb.level == level)]
            if row.empty:
                r.remark(f"robustness.csv has no row for {spec}={level}")
                continue
            r.check(f"robustness {spec}={level}", pv, float(row.iloc[0].effect),
                    tol=0.004)
        f1 = rb[(rb.spec == "measurement_floor") & (rb.level == "1")]
        f3 = rb[(rb.spec == "measurement_floor") & (rb.level == "3")]
        if not f1.empty and not f3.empty:
            a = 1 - float(f3.iloc[0].effect) / float(f1.iloc[0].effect)
            r.check("attenuation at a 3 s measurement floor", ROBUSTNESS_TAU3_ATTENUATION, a,
                    tol=0.03)
        cal = rb[rb.spec == "caliper"]
        if not cal.empty:
            lo, hi = float(cal.effect.min()), float(cal.effect.max())
            r.check_range("caliper range", CALIPER_RANGE, [lo, hi], tol=0.004)
            r.remark(f"caliper range [{lo:+.3f}, {hi:+.3f}] - "
                     f"output of prepared_net.parquet (the four untitled tiers)")
        # "The effect of blunders was small at every threshold and reached
        #  significance at none"
        bt = rb[rb.spec == "blunder_threshold"]
        if not bt.empty and "se" in bt.columns:
            worst = float((bt.effect.abs() / bt.se).max())
            # The manuscript now reports these as positive at every
            # threshold, so the test is that they are, not that they are not.
            smallest = float((bt.effect / bt.se).abs().min())
            allpos = bool((bt.effect > 0).all())
            line = (f"{'blunder threshold, all positive':<46} "
                    f"paper  yes, all significant   "
                    f"repo {'yes' if allpos else 'no'}, min |t| {smallest:>5.2f}")
            (r.ok if allpos and smallest >= 1.96
             else r.bad).append(("DATA", line))

        cv = rb[rb.spec == "caliper_vars"]
        if not cv.empty:
            cvm = cv.set_index("level")
            for lvl, pv in CALIPER_VARS.items():
                if lvl in cvm.index:
                    r.check(f"caliper variables, {lvl}", pv,
                            float(cvm.loc[lvl, "effect"]), tol=0.004)
            if len(cvm) == 2:
                gap = abs(float(cvm.effect.max() - cvm.effect.min()))
                line = (f"{'5 vars vs 6 vars, same subsample':<46} "
                        f"paper  < {CALIPER_VARS_MAX_GAP}      repo {gap:>9.4f}")
                (r.ok if gap <= CALIPER_VARS_MAX_GAP
                 else r.bad).append(("DATA", line))
    else:
        r.remark("robustness.csv is absent - create it with python run.py robustness")

    # ── 혼합효과 모형 ──────────────────────────────────────
    mx = _read("mixed_B.csv")
    if mx is not None and not mx.empty:
        col = ("coefficient" if "coefficient" in mx.columns
               else "coef" if "coef" in mx.columns else None)
        if col is None:
            r.remark("mixed_B.csv has no coefficient column")
        else:
            m = mx.set_index("spec")
            for spec, pv in MIXED.items():
                if spec in m.index:
                    r.check(f"mixed model, {spec}", pv,
                            float(m.loc[spec, col]), tol=0.004)
                else:
                    r.remark(f"mixed_B.csv has no \"{spec}\" row")
            # "with the sample held constant the two approaches agree within 0.003"
            for a, b in MIXED_PAIRS:
                if a in m.index and b in m.index:
                    d = abs(float(m.loc[a, col]) - float(m.loc[b, col]))
                    tag = a.replace(" wp", "")
                    (r.ok if d <= MIXED_VS_PAIRED_MAX else r.bad).append(
                        ("DATA", f"{'mixed vs paired difference (' + tag + ')':<46} "
                                 f"paper  < {MIXED_VS_PAIRED_MAX}      "
                                 f"repo {d:>9.4f}"))
            # Which path the fit took is a reported quantity: a model that
            # falls back has not converged, and the manuscript says so.
            main = "mixed model (main)"
            if main in m.index and "convergence" in mx.columns:
                got = str(m.loc[main, "convergence"])
                line = (f"{'mixed model convergence path':<46} "
                        f"paper {MIXED_MAIN_PATH}   repo {got}")
                (r.ok if got == MIXED_MAIN_PATH
                 else r.bad).append(("DATA", line))
            for spec, pse in MIXED_SE.items():
                if spec in m.index and "se" in mx.columns:
                    r.check(f"SE, {spec}", pse, float(m.loc[spec, "se"]),
                            tol=0.0015)
            if main in m.index and "n_players" in mx.columns:
                r.check("mixed model players", MIXED_MAIN_PLAYERS,
                        float(m.loc[main, "n_players"]), tol=1)
        pd_rows = [i for i in m.index if str(i).startswith("paired difference")]
        if "n_obs" in mx.columns and pd_rows:
            r.remark("paired difference rows: " + ", ".join(
                f"{i} n={int(m.loc[i, 'n_obs']):,}" for i in pd_rows))
    else:
        r.remark("mixed_B.csv is absent - create it with python run.py mixed B")

    # ── 제목의 분석 수 ─────────────────────────────────────
    from .config import PREPARED, PREPARED_FM
    import pyarrow.parquet as pq
    parts, missing = {}, []
    for nm, path in [("untitled", PREPARED), ("titled", PREPARED_FM)]:
        if os.path.exists(path):
            parts[nm] = pq.ParquetFile(path).metadata.num_rows
        else:
            missing.append(os.path.basename(path))
    if missing:
        r.remark(f"{', '.join(missing)} absent - the title's "
                 f"{N_MOVES_TITLE:,} moves cannot be checked in full; "
                 f"run run.py prep and run.py prep fm")
    # Sample-wide counts are five-tier quantities, so they are only
    # comparable when both prepared tables are present. With one of them
    # missing the totals fall short by exactly the other tier and every
    # comparison reads as a mismatch -- six false failures, which is the same
    # error as comparing two different things and calling the difference a
    # defect. Skip instead, and say why.
    BOTH_TABLES = os.path.exists(PREPARED) and os.path.exists(PREPARED_FM)
    if not BOTH_TABLES:
        absent = [os.path.basename(q) for q in (PREPARED, PREPARED_FM)
                  if not os.path.exists(q)]
        r.remark(f"{', '.join(absent)} absent - the sample-wide counts "
                 f"(events, players, games, title) span five tiers and are "
                 f"not comparable from one table; run run.py prep and "
                 f"run.py prep fm")

    # The revision to Event B, on the analysis tables.
    if BOTH_TABLES:
        rev = pd.concat(
            [pd.read_parquet(q, columns=["see_loss", "net_mat"])
             for q in (PREPARED, PREPARED_FM) if os.path.exists(q)],
            ignore_index=True)
        see = rev.see_loss.notna() & (rev.see_loss >= 1)
        if int(see.sum()):
            sub = rev[see]
            r.check("registered-criterion events",
                    N_EVENTS_MATERIAL_REGISTERED, float(see.sum()),
                    tol=20_000)
            r.check("share with no net change", REVISION_SHARE_ZERO,
                    100 * float((sub.net_mat == 0).mean()), tol=1.0)
            r.check("share with a net gain", REVISION_SHARE_GAIN,
                    100 * float((sub.net_mat > 0).mean()), tol=1.0)
            r.remark(f"revision: {int(see.sum()):,} events under the "
                     f"registered criterion, "
                     f"{int((rev.net_mat < 0).sum()):,} under net loss "
                     f"({100 * (1 - (rev.net_mat < 0).sum() / see.sum()):.0f}% "
                     f"fewer)")
        del rev

    # Event counts and analysed players, both prepared tables together.
    cnt = {}
    for nm, path in [("untitled", PREPARED), ("titled", PREPARED_FM)]:
        if os.path.exists(path):
            cnt[nm] = pd.read_parquet(
                path, columns=["player", "game_id", "net_mat",
                               "wp_delta", "wp_before"])
    if cnt and BOTH_TABLES:
        allp = pd.concat(cnt.values(), ignore_index=True)
        r.check("players in the analysis tables", N_PLAYERS_ANALYSED,
                float(allp.player.nunique()), tol=1)
        r.check("material-loss events", N_EVENTS_MATERIAL,
                float((allp.net_mat < 0).sum()), tol=5_000)
        bl = (allp.wp_delta.notna() & (allp.wp_delta <= -BLUNDER_THRESH_PP)
              & allp.wp_before.between(0.10, 0.90))
        r.check("blunder events", N_EVENTS_BLUNDER, float(bl.sum()), tol=2_000)
        r.check("material events per game", EVENTS_PER_GAME,
                float((allp.net_mat < 0).sum()) / allp.game_id.nunique(),
                tol=0.05)
        r.check("games in the analysis tables", N_GAMES_ANALYSED,
                float(allp.game_id.nunique()), tol=1)
        if N_PLAYERS != allp.player.nunique():
            r.remark(f"Table 1 samples {N_PLAYERS:,} players; the analysis "
                     f"tables hold {allp.player.nunique():,}. The manuscript "
                     f"should say which number a given sentence means.")
        del allp

    if parts and BOTH_TABLES:
        total = sum(parts.values())
        detail = " + ".join(f"{k} {v:,}" for k, v in parts.items())
        r.check("moves entering the analysis (title)", N_MOVES_TITLE,
                float(total), tol=max(50_000, 0.02 * N_MOVES_TITLE))
        r.remark(f"title count is {detail} = {total:,}")

    # ── 9점 제한 합법수 구간 ───────────────────────────────
    qb = _read("legal_bins_queen.csv")
    if qb is not None and {"label", "bin", "effect"} <= set(qb.columns):
        piv = qb.pivot_table(index="bin", columns="label", values="effect")
        if {"self_lost", "opp_lost"} <= set(piv.columns):
            d = (piv["self_lost"] - piv["opp_lost"]).dropna()
            if len(d):
                r.check("maximum separation, nine-point losses", QUEEN_BIN_MAX, float(d.min()),
                        tol=0.006)
    else:
        r.remark("legal_bins_queen.csv is absent - cannot check the "
                 "manuscript's \"-1.04 at its maximum\"")

    # ── 구성 논증의 산술 ───────────────────────────────────
    if sp is not None:
        bo = sp[sp.label == "blunder_only"]
        bl2 = sp[sp.label == "blunder_loss"]
        if not bo.empty and not bl2.empty:
            share = COMPOSITION["share_loss_10pp"]
            pred = ((1 - share) * float(bo.iloc[0].e1)
                    + share * float(bl2.iloc[0].e1))
            r.check("compositional weighted average, predicted", COMPOSITION["predicted_10pp"], pred,
                    tol=0.004, kind="PAPER")

    # ── lag_profiles 의 사건 정의 ───────────────────────────
    lp = _read("lag_profiles.csv")
    if lp is not None and "event_type" in lp.columns:
        kinds = set(lp.event_type.unique())
        if "material_see" not in kinds and "material" in kinds:
            sub = lp[(lp.event_type == "material") & (lp.lag == 1)]
            if not sub.empty:
                worst = float(sub.effect.abs().max())
                if worst < 0.4:
                    r.remark(
                        f"the largest material t+1 effect in lag_profiles.csv is only "
                        f"{worst:.3f}; the manuscript reports about {abs(GROUPS['loss_only'][0]):.3f}, "
                        f"so this file may have been built with the withdrawn SEE definition. "
                        f"Regenerate with python run.py lag-all.")

    # ── 파일 사이의 일관성 ─────────────────────────────────
    #
    # 여기서부터는 논문과의 대조가 아니라 **저장소 산출물끼리** 모순이
    # 없는지 본다. 원래 figures/make_figures.py 의 check_consistency() 에
    # 있던 검사인데, 그림 코드를 저장소에서 빼면서 이쪽으로 옮겼다.
    #
    # ★ 이 검사들이 잡는 것: 공개 전 점검에서 효과값은 전부 맞는데
    #   표준오차만 1.6배 다른 상태가 발견된 적이 있다. 두 파일이 서로
    #   다른 실행에서 나왔기 때문이었다. 논문 대조만으로는 안 잡힌다.
    sp2, pp2 = _read("se_split.csv"), _read("per_player_t1.csv")

    if sp2 is not None and "np1" not in sp2.columns:
        r.bad.append(("DATA",
            f"{'se_split.csv columns':<46} no np1 (cluster count) column - "
            f"not the output of the current run.py se"))

    if sp2 is not None and pp2 is not None and "group" in pp2.columns:
        for g in GROUPS:
            v = pp2[pp2.group == g].eff
            row = sp2[sp2.label == g]
            if v.empty or row.empty:
                continue
            r.check(f"{g}: t+1 effect, two files", float(row.iloc[0]["e1"]),
                    float(v.mean()), tol=0.02,
                    names=("se_split", "per_player"))
            sa = float(v.std() / np.sqrt(len(v)))
            sb = float(row.iloc[0]["se1"])
            if sb > 0:
                ratio = sa / sb
                line = (f"{g + ': SE ratio':<46} per_player {sa:>9.4f}  "
                        f"se_split {sb:>9.4f}  ratio {ratio:.2f}")
                (r.ok if 0.7 < ratio < 1.4 else r.bad).append(("DATA", line))

    dose2 = _read("se_dose.csv")
    if dose2 is not None and FD is not None:
        for i, v in enumerate(FD.DOSE["x"]):
            row = dose2[dose2.label == f"self_{v}"]
            if row.empty:
                continue
            r.check(f"dose_{v}: reference table vs CSV", FD.DOSE["self"][i],
                    float(row.iloc[0]["e1"]), tol=0.02,
                    names=("ref", "se_dose"))

    if verbose:
        _print(r)
    return r


def _print(r):
    print("=" * 78)
    print("Manuscript values vs repository output")
    print("=" * 78)
    print(f"  matched {len(r.ok)}   mismatched {len(r.bad)}   notices {len(r.note)}")
    if r.bad:
        print("\n[MISMATCHED]  DATA = regenerate the repository output / PAPER = check the manuscript")
        for kind, line in r.bad:
            print(f"  {kind:<6}{line}")
    if r.note:
        print("\n[NOTICES]")
        for n in r.note:
            print(f"  - {n}")
    if not r.bad and not r.note:
        print("\n  Everything matches.")
    print()
