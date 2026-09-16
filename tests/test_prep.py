"""분석 테이블 생성 단위 테스트 — net_mat, z_pre3, 사건 정의

z_pre3 는 **플라이 인덱스 기준**이어야 한다. 예전 구현은 groupby().shift()
로 행 위치 기준이었는데, 측정 하한(move_time < 1초)으로 중간 수가 빠지면
"직전 3수"가 실제로는 더 먼 과거를 가리킨다.
test_z_pre3_skips_missing_ply 가 그 경우다.
"""

import numpy as np
import pandas as pd
import pytest

from src.prepare import add_pre_speed, net_material_change
from src.prepare import events_A, events_B, events_B_see, mask_A, mask_B


def frame(plies, z=None, mat=None, player="p1", game="g1"):
    n = len(plies)
    return pd.DataFrame({
        "game_id": [game] * n,
        "player": [player] * n,
        "ply": plies,
        "z": np.arange(n, dtype=float) if z is None else z,
        "mat_diff": np.zeros(n) if mat is None else mat,
    })


# ── net_mat ─────────────────────────────────────────────────

def test_net_mat_is_next_own_move():
    d = frame([10, 12, 14, 16], mat=[0.0, -1.0, -1.0, -4.0])
    out = net_material_change(d)
    # ply10: mat(12) - mat(10) = -1 ; ply12: mat(14)-mat(12) = 0
    assert out.net_mat.tolist()[:3] == [-1.0, 0.0, -3.0]
    assert np.isnan(out.net_mat.iloc[-1])   # 다음 자기 수가 없다


def test_even_exchange_is_not_an_event():
    """등가 교환은 순변화 0 → 사건이 아니다. 철회된 SEE 정의와의 차이점."""
    d = frame([10, 12], mat=[0.0, 0.0])
    out = net_material_change(d)
    assert out.net_mat.iloc[0] == 0.0
    assert not mask_B(out).iloc[0]


def test_net_mat_gap_in_plies_gives_nan():
    """ply+2 가 없으면(수가 빠졌으면) 더 먼 수로 건너뛰지 않는다."""
    d = frame([10, 14], mat=[0.0, -3.0])
    out = net_material_change(d)
    assert np.isnan(out.net_mat.iloc[0])


def test_net_mat_separates_players_and_games():
    a = frame([10, 12], mat=[0.0, -1.0], player="p1", game="g1")
    b = frame([10, 12], mat=[5.0, 5.0], player="p2", game="g2")
    out = net_material_change(pd.concat([a, b], ignore_index=True))
    assert out.net_mat.iloc[0] == -1.0
    assert out.net_mat.iloc[2] == 0.0


# ── z_pre3 ──────────────────────────────────────────────────

def test_z_pre3_is_mean_of_three_previous_own_moves():
    d = frame([10, 12, 14, 16], z=[1.0, 2.0, 3.0, 9.0])
    out = add_pre_speed(d, k=3)
    assert np.isnan(out.z_pre3.iloc[0])
    assert np.isnan(out.z_pre3.iloc[2])          # 앞에 두 수뿐
    assert out.z_pre3.iloc[3] == pytest.approx((1.0 + 2.0 + 3.0) / 3)


def test_z_pre3_skips_missing_ply():
    """중간 수가 측정 하한으로 빠지면 z_pre3 는 결측이어야 한다.

    행 위치 기준(shift)이면 ply 8 을 직전 수로 잘못 쓴다.
    """
    d = frame([8, 10, 14, 16], z=[5.0, 1.0, 2.0, 3.0])   # ply 12 가 없다
    out = add_pre_speed(d, k=3)
    assert np.isnan(out.z_pre3.iloc[3]), (
        "ply 기준이 아니라 행 위치 기준으로 shift 하고 있다")


def test_z_pre3_rejects_duplicate_keys():
    d = frame([10, 10], z=[1.0, 2.0])
    with pytest.raises(ValueError):
        add_pre_speed(d)


# ── 사건 정의 ───────────────────────────────────────────────

def test_events_B_uses_net_mat_not_see():
    d = frame([10, 12], mat=[0.0, -3.0])
    d = net_material_change(d)
    d["see_loss"] = [9.0, 9.0]          # SEE 는 크게 나오지만
    d["tier"] = "1300-1600"
    assert len(events_B(d)) == 1        # net 기준으로는 1건
    assert len(events_B_see(d)) == 2    # SEE 기준으로는 2건


def test_events_B_requires_net_mat_column():
    d = frame([10, 12])
    with pytest.raises(KeyError, match="net_mat"):
        events_B(d)


def test_events_A_bounds_are_inclusive_and_symmetric():
    d = pd.DataFrame({
        "wp_delta": [-0.10, -0.09, -0.50, -0.50],
        "wp_before": [0.50, 0.50, 0.05, 0.95],
    })
    m = mask_A(d)
    assert m.tolist() == [True, False, False, False]
    assert len(events_A(d)) == 1


def test_split_groups_are_exclusive():
    d = frame([10, 12, 14, 16], mat=[0.0, -1.0, -1.0, -1.0])
    d = net_material_change(d)
    d["wp_delta"] = [-0.5, -0.5, 0.0, 0.0]
    d["wp_before"] = [0.5, 0.5, 0.5, 0.5]
    hasA, hasB = mask_A(d), mask_B(d)
    groups = [(hasA & ~hasB), (hasA & hasB), (hasB & ~hasA)]
    total = sum(g.sum() for g in groups)
    assert total == (hasA | hasB).sum()      # 겹치지 않고 빠지지 않는다


# ── 빈 사건 집합 (회귀 테스트) ──────────────────────────────

def test_builders_handle_empty_events():
    """사건이 0건이면 빈 프레임을 돌려줘야 한다 (예전엔 AttributeError).

    run.py bins 가 '상대가 재료를 잃은 사건'을 찾지 못하는 데이터에서
    z_post 접근으로 죽었다.
    """
    from src.analysis import lagwise, build_with_pretrend, build_pairs
    from src.prepare import build_and_match

    d = frame([10, 12, 14, 16], z=[1.0, 2.0, 3.0, 4.0])
    d["tier"] = "1300-1600"
    for c in ("wp_before", "clk_before", "n_legal"):
        d[c] = 1.0
    empty = d.iloc[:0]
    cal = {"wp_before": 0.5}

    r = lagwise(d, empty, cal)
    assert len(r) == 0 and "lag+1" in r.columns

    for fn in (build_with_pretrend, build_and_match):
        r = fn(d, empty, cal)
        assert len(r) == 0 and "effect" in r.columns

    r = build_pairs(d, empty, cal)
    assert len(r) == 0 and "event_condition" in r.columns


# ---------------------------------------------------------------------------
# Caliper widths must come from the frame the analyses actually run on.
#
# _calipers_from_main() used to take standard deviations over the whole
# prepared table, while every analysis first drops rows with no z_pre3. Those
# rows are 9.9% of the table and sit at the start of games -- low ply, high
# clock -- so the two conventions disagreed by up to 4%, and the published
# CSVs were produced under the second one.
# ---------------------------------------------------------------------------

def test_caliper_sd_excludes_missing_prespeed():
    import numpy as np
    import pandas as pd

    # Rows missing z_pre3 are deliberately extreme, as the real ones are.
    d = pd.DataFrame({
        "wp_before": [0.5] * 10 + [0.99] * 10,
        "clk_before": [100.0] * 10 + [600.0] * 10,
        "n_legal": [30] * 10 + [5] * 10,
        "ply": [40] * 10 + [2] * 10,
        "z_pre3": [0.1, -0.1] * 5 + [np.nan] * 10,
    })
    kept = d.dropna(subset=["z_pre3"])

    over_all = {v: d[v].std() for v in d.columns}
    over_kept = {v: kept[v].std() for v in kept.columns}

    # The two conventions must actually differ here, or the test proves nothing.
    assert over_all["wp_before"] != over_kept["wp_before"]

    # Every matching variable is constant once the missing rows are dropped.
    for v in ["wp_before", "clk_before", "n_legal", "ply"]:
        assert over_kept[v] == 0, v
        assert over_all[v] > 0, v


# ---------------------------------------------------------------------------
# A shard containing none of the sampled players writes an empty frame, whose
# columns are all dtype object. Concatenating one of those into the analysis
# table turned move_time into object, and np.log then raised
# "loop of ufunc does not support argument 0 of type float".
#
# The four untitled tiers never triggered it: every shard held one of the
# 1,878 players. The titled tier is sparse enough that 119 of its 2,355
# shards came back empty.
# ---------------------------------------------------------------------------

def test_empty_shard_does_not_poison_dtypes():
    import numpy as np
    import pandas as pd

    real = pd.DataFrame({"player": ["a", "a"], "ply": [10, 12],
                         "move_time": [3, 5]})
    blank = pd.DataFrame(columns=["player", "ply", "move_time"])
    assert blank["move_time"].dtype == object

    naive = pd.concat([real, blank], ignore_index=True)
    assert naive["move_time"].dtype == object, "the hazard must still exist"
    with pytest.raises(TypeError):
        np.log(naive["move_time"] + 1.0)

    # What cmd_prep now does: drop the empty frames before concatenating.
    parts = [f for f in (real, blank) if len(f)]
    fixed = pd.concat(parts, ignore_index=True)
    assert fixed["move_time"].dtype != object
    assert np.isfinite(np.log(fixed["move_time"] + 1.0)).all()
