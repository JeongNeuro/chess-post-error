"""
Reference values the pipeline does not produce.

Some values the manuscript reports do not come from this pipeline. They are
collected here so that `run.py verify-paper` can compare them against the
transcription in `src/paper_check.py`.

Each entry carries a status:
    "external"  produced outside this repository. The flanker comparison
                (OpenNeuro ds004883) is the only case; the code for that
                analysis is not here.
    "pending"   would be replaced by a CSV once the run.py stage named in the
                comment has been run. Every panel the manuscript reports now
                has a generating stage, and `data/derived/` takes precedence;
                these remain only as a fallback.

History of this file
    It was `figures/figure_data.py`. Despite the name it was not figure code
    but a table of values: at the time the plotting code had numbers written
    directly into it, so there was no way to check a figure against the
    output it supposedly came from. Fig 1c once carried error bars 1.6 to 1.8
    times larger than the standard errors in se_split.csv without anything
    catching it. The plotting code has since been removed from the
    repository; this table is still used for verification, so it moved to
    src/.
"""

TIER_NAMES = ["1300–1600", "1600–1900", "1900–2100", "2100+"]

# data/derived/lag_profiles.csv 의 tier 값 ↔ 그림 범례 이름
TIER_ORDER = ["1300-1600", "1600-1900", "1900-2100", "2100+"]
TIER_DISPLAY = dict(zip(TIER_ORDER, TIER_NAMES))

# ★ 논문 본문과 t+1 이 어긋난다.
#   논문: "Values at t+1 ranged from -0.641 to -0.548 across tiers"
#   아래: -0.617 ~ -0.564
#   t+2 / t+3 / blunder 행은 논문과 정확히 일치하므로, t+1 열만 다른 실행에서
#   온 것으로 보인다. run.py verify-paper 가 이 항목을 잡는다.
MATERIAL_BY_TIER = {
    "status": "pending",
    "lags": [-3, -2, -1, 0, 1, 2, 3],
    # 사건 전(t-3~t-1) 범위는 -0.026 ~ +0.023 이다. 논문 본문의
    # "-0.057 to +0.048" 은 se_split.csv 의 blunder_only t-1 / t-3 값과
    # 정확히 일치하므로, 세 집단 패널(Fig 1c)의 수치를 층별 문단에 잘못
    # 옮겨 적은 것으로 보인다. 원고 쪽 확인 필요.
    "rows": [

        [-0.011, -0.001, 0.014, 0.260, -0.564, -0.082, -0.107],

        [0.004, -0.008, 0.000, 0.233, -0.599, -0.082, -0.079],

        [-0.026, -0.001, 0.020, 0.224, -0.617, -0.057, -0.083],

        [-0.008, -0.002, 0.023, 0.229, -0.598, -0.045, -0.038],
    ],
}

BLUNDER_BY_TIER = {
    "status": "pending",
    "lags": [-3, -2, -1, 0, 1, 2, 3],
    "rows": [

        [0.032, 0.005, -0.004, 0.318, -0.029, -0.001, -0.049],

        [0.056, 0.033, -0.040, 0.326, -0.035, 0.004, -0.011],

        [0.019, 0.039, -0.057, 0.335, -0.030, -0.019, -0.016],

        [0.014, 0.026, -0.028, 0.334, -0.007, -0.013, -0.047],
    ],
}

# ─────────────────────────────────────────────────────────────
# Fig 2 (b) — 합법수 변화량 구간별 이동시간 변화
# status: pending
# 대체: python run.py bins        → legal_bins.csv
# ─────────────────────────────────────────────────────────────
LEGAL_MOVE_BINS = {
    "status": "pending",
    "x": [0, 1, 2, 3, 4, 5, 6, 7],
    "xticklabels": ["≤−12", "−8~−4", "−1~+1", "+4~+8"],
    "self_lost": [-1.109, -1.059, -0.911, -0.736, -0.663, -0.592, -0.513, 0.030],
    "self_se":   [0.048, 0.052, 0.039, 0.031, 0.034, 0.038, 0.049, 0.041],
    "opp_lost":  [-0.082, 0.368, 0.535, 0.662, 0.686, 0.728, 0.802, 0.875],
    "opp_se":    [0.061, 0.057, 0.043, 0.033, 0.033, 0.035, 0.039, 0.043],
}

# ─────────────────────────────────────────────────────────────
# Fig 2 (c) — 손실 크기별 효과 (자기 손실 / 상대 손실)
# status: pending
# 대체: python run.py se dose     → se_dose.csv  (이미 읽고 있다)
#       라벨은 self_N / opp_N 이다 (예전 dose_N / gain_N).
# ─────────────────────────────────────────────────────────────
DOSE = {
    "status": "pending",
    "x": [1, 3, 5, 9],
    # Five-variable matching. se_dose.csv is the source; these are the
    # fallback for when it is absent.
    "self": [-0.3005, -0.6972, -0.8053, -0.9786],
    "self_se": [0.0147, 0.0147, 0.0156, 0.0161],
    "opp": [0.2190, 0.1591, 0.1628, 0.0269],
    "opp_se": [0.0141, 0.0144, 0.0163, 0.0158],
}

# ─────────────────────────────────────────────────────────────
# Fig 2 (d)(e) — 신뢰도
# status: pending (체스) / external (플랭커)
# 대체: python run.py reliability  → reliability.csv, reliability_curve.csv
# ─────────────────────────────────────────────────────────────
RELIABILITY_CHESS = {
    "status": "pending",
    "n": [20, 40, 60, 100, 150, 237, 300],
    "sb": [0.378, 0.531, 0.603, 0.596, 0.685, 0.741, 0.754],
}
RELIABILITY_FLANKER = {
    "status": "external",
    "n": [10, 20, 30, 50, 80],
    "sb": [0.233, 0.095, 0.358, 0.544, 0.676],
}
RELIABILITY_BY_CONDITION = {
    "status": "pending",
    # 원고 v8 Table 2 — 층을 합친 두 행.
    "labels": ["Blund.\nuntitled", "Mat.\nuntitled"],
    "sb": [-0.113, 0.781],
    "sigma_b": [0.284, 0.218],
}

# ─────────────────────────────────────────────────────────────
# 플랭커 비교 (OpenNeuro ds004883)
# status: external
# 이 저장소에는 플랭커 분석 코드가 없다. 아래 값은 별도 스크립트에서
# 나온 것이라 status="external" 로 분리해 두었고, verify-paper 와
# figures --check 는 이 값들을 저장소 산출물과 대조하지 않는다.
# 논문에 인용하려면 출처(OpenNeuro ds004883)를 명시해야 한다.
# ─────────────────────────────────────────────────────────────
FLANKER = {
    "status": "external",
    "post_error_slowing": 0.131,
    "post_error_slowing_se": 0.020,
    "n_correct": 81108,
    "n_error": 10461,
    "err_rate_after_correct": 10.2,
    "err_rate_after_error": 21.4,
}
