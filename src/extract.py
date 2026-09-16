"""
Raw archives → player sample → ply-level features

Four stages in pipeline order:

  1. PGN movetext parsing and win-probability transform
     parse_movetext / move_times / winprob
  2. Stage 1   account metadata     scan_shard / scan_titled_shard
  3. Stage 1b  stratified sampling  aggregate_players / build_sample
  4. Stage 2   board replay and feature extraction   extract_shard  ← the core

Driver: `run.py`. There is no entry point in this file.
"""

# ───────────────────────────────────────────────────────────
# REVIEW NOTE ★ the reported numbers originate in this file
#
#   [parsing]
#   - Does the win-probability transform and its coefficient (0.00368208)
#     match the equation in the Method section?
#   - Perspective normalisation: the sign of the centipawn score is flipped
#     for Black. Omitting this loses every blunder by Black and keeps only
#     White's.
#   - Handling of mate scores (described in the manuscript as 100%).
#   - move_time = clk(i−2) − clk(i). Increment is zero, so the plain
#     difference is correct. A single missing clock removes two move times.
#
#   [sampling]
#   - Tier definitions: 500 players from each of the four untitled tiers,
#     all titled accounts that qualify.
#   - Where the minimum of 30 games is applied.
#   - That the seed is fixed (reproducibility).
#   - The titled tier is scanned across 2024-01 to 2024-06; one month yields
#     only 26 accounts with 30 or more games.
#
#   [extraction] ★★
#   - mat_diff: piece values (pawn 1, knight 3, bishop 3, rook 5, queen 9),
#     king excluded; confirm the sign is from the moving player's
#     perspective. All of Event B rests on this.
#   - mat_diff is filled only when do_see=True.
#   - Which point in the move n_legal, ply, wp_before and wp_delta refer to.
#   - The surest check is to follow one game by hand and count mat_diff.
#   - This is the slowest stage (about 45 minutes).
# ───────────────────────────────────────────────────────────

import hashlib
import os
import re
import subprocess

import chess
import duckdb
import numpy as np
import pandas as pd

from .config import *          # noqa: F403  임계값·경로·층 정의
from .see import see, VALUES


# ════════════════════════════════════════════════════════════════════
# 1. PGN movetext 파싱 + 승률 변환
# 리체스 승률 함수:
#     winprob(cp) = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
#     → 0~100 (백 관점). 여기서는 0~1로 정규화해 사용.
# 
# 관점 정규화: 플레이어가 흑이면 cp의 부호를 뒤집은 뒤 변환한다.
# 이것을 빠뜨리면 흑의 블런더를 전부 놓치고 백의 블런더만 잡게 된다.
# ════════════════════════════════════════════════════════════════════

K = 0.00368208

CLK_RE  = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+)\]")
EVAL_RE = re.compile(r"\[%eval\s+(#?-?[\d.]+)\]")
# 각 수 + 뒤따르는 주석 블록을 함께 잡는다
MOVE_RE = re.compile(
    r"(?:\d+\.+\s*)?"                     # 수 번호 (선택)
    r"((?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?)"
    r"([?!]*)"                            # NAG: 리체스가 ?/??/!/!? 를 붙여둠
    r"\s*(\{[^}]*\})?"                    # 주석 블록
)
NAG_MAP = {"?!": "inaccuracy", "?": "mistake", "??": "blunder"}

MATE_CP = 10000   # 메이트 스코어 클리핑


def winprob(cp):
    """센티폰(백 관점) → 승률 0~1 (백 관점)"""
    return 1.0 / (1.0 + np.exp(-K * np.asarray(cp, dtype=float)))


def parse_eval(tok):
    """'#-3' 또는 '0.25' → 센티폰(백 관점) float"""
    if tok.startswith("#"):
        v = tok[1:]
        sign = -1.0 if v.startswith("-") else 1.0
        return sign * MATE_CP
    return float(tok) * 100.0


def parse_movetext(mt):
    """
    movetext → (clks, evals, sans, nags)
    세 배열은 모두 플라이 인덱스 정렬. 값이 없으면 None.
    clk는 정수 초 (리체스가 H:MM:SS로만 기록).
    """
    clks, evals, sans, nags = [], [], [], []
    for m in MOVE_RE.finditer(mt):
        san, nag, comment = m.group(1), m.group(2), m.group(3)
        sans.append(san)
        nags.append(NAG_MAP.get(nag))          # 리체스 자체 분류 (교차검증용)
        c = e = None
        if comment:
            cm = CLK_RE.search(comment)
            if cm:
                h, mi, s = cm.groups()
                c = int(h) * 3600 + int(mi) * 60 + int(s)
            em = EVAL_RE.search(comment)
            if em:
                e = parse_eval(em.group(1))
        clks.append(c)
        evals.append(e)
    return clks, evals, sans, nags


def move_times(clks, color):
    """
    한 플레이어의 수별 소요 시간.
    color: 0=백(짝수 플라이 인덱스), 1=흑(홀수)
    소요시간(i) = clk(i-2) - clk(i)   ← 같은 플레이어의 연속 clk
    증분 0이므로 보정항 없음.
    각 플레이어의 첫 수는 이전 clk가 없어 정의 불가 → None
    반환: {ply_index: seconds}
    """
    out = {}
    idxs = list(range(color, len(clks), 2))
    for j, i in enumerate(idxs):
        if j == 0:
            continue                       # 첫 수: 정의 불가
        prev = idxs[j - 1]
        if clks[i] is None or clks[prev] is None:
            continue
        d = clks[prev] - clks[i]
        if d < 0:
            continue                       # 음수: 기록 오류 (실측 0.011%)
        out[i] = d
    return out


def player_winprobs(evals, color):
    """
    플레이어 관점 승률 배열.
    evals는 백 관점 센티폰. 흑이면 부호 반전 후 변환.
    반환: {ply_index: winprob(0~1)}  — 해당 수를 둔 직후 국면
    """
    out = {}
    sign = 1.0 if color == 0 else -1.0
    for i, e in enumerate(evals):
        if e is None:
            continue
        out[i] = float(winprob(sign * e))
    return out

# ════════════════════════════════════════════════════════════════════
# 2. Stage 1 — 계정 메타데이터 집계 (movetext 미사용)
# 층 배정과 표집 후보 확정. 대국 내용은 읽지 않는다.
# 샤드 단위로 다운로드 → 집계 → 즉시 삭제. 재시작 가능.
# 산출: {WORK}/players/sNNNNN.parquet, {WORK}/titled/mMM_sNNNNN.parquet
# ════════════════════════════════════════════════════════════════════

def shard_url(year, month, i, n):
    return ARCHIVE_URL.format(y=year, m=month, i=i, n=n)


def download(url, dst):
    r = subprocess.run(["curl", "-sL", "--fail", "-o", dst, url],
                       capture_output=True)
    return r.returncode == 0 and os.path.getsize(dst) > 1_000_000


SQL_PLAYERS = f"""
WITH g AS (
  SELECT White AS player, WhiteTitle AS title, WhiteElo AS elo,
         (movetext LIKE '%eval%') AS has_eval
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}'
    AND Event LIKE 'Rated%'
    AND (WhiteTitle IS NULL OR WhiteTitle <> 'BOT')
    AND (BlackTitle IS NULL OR BlackTitle <> 'BOT')
  UNION ALL
  SELECT Black, BlackTitle, BlackElo, (movetext LIKE '%eval%')
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}'
    AND Event LIKE 'Rated%'
    AND (WhiteTitle IS NULL OR WhiteTitle <> 'BOT')
    AND (BlackTitle IS NULL OR BlackTitle <> 'BOT')
)
SELECT player,
       any_value(title)            AS title,
       count(*)                    AS n_games,
       sum(has_eval::INT)          AS n_eval_games,
       median(elo)                 AS elo_median,
       min(elo) AS elo_min, max(elo) AS elo_max
FROM g
WHERE elo IS NOT NULL
GROUP BY player
"""


def scan_shard(path, out_path):
    con = duckdb.connect()
    con.execute(f"COPY ({SQL_PLAYERS.format(f=path)}) TO '{out_path}' (FORMAT PARQUET)")
    con.close()




STAGE1_FM = STAGE1_FM

TITLED_MONTHS = TITLED_MONTHS          # {1: 421, 2: 388, ... 6: 379}

SQL_TITLED = f"""
WITH g AS (
  SELECT White AS player, WhiteTitle AS title, WhiteElo AS elo,
         (movetext LIKE '%eval%') AS has_eval
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}' AND Event LIKE 'Rated%'
    AND WhiteTitle IN ('FM','IM','GM','WFM','WIM','WGM')
    AND (BlackTitle IS NULL OR BlackTitle <> 'BOT')
  UNION ALL
  SELECT Black, BlackTitle, BlackElo, (movetext LIKE '%eval%')
  FROM read_parquet('{{f}}')
  WHERE TimeControl = '{TIME_CONTROL}' AND Event LIKE 'Rated%'
    AND BlackTitle IN ('FM','IM','GM','WFM','WIM','WGM')
    AND (WhiteTitle IS NULL OR WhiteTitle <> 'BOT')
)
SELECT player, any_value(title) AS title, count(*) AS n_games,
       sum(has_eval::INT) AS n_eval_games,
       median(elo) AS elo_median, min(elo) AS elo_min, max(elo) AS elo_max
FROM g WHERE elo IS NOT NULL GROUP BY player
"""


def scan_titled_shard(args):
    m, i, n = args
    p = f"{STAGE1_FM}/m{m:02d}_s{i:05d}.parquet"
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return "skip"
    tmp = os.path.join(WORK, f"_t{m}_{i}.parquet")
    url = ARCHIVE_2024_URL.format(m=m, i=i, n=n)
    r = subprocess.run(["curl", "-sL", "--fail", "-o", tmp, url], capture_output=True)
    if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1_000_000:
        if os.path.exists(tmp):
            os.remove(tmp)
        return "fail"
    try:
        con = duckdb.connect()
        con.execute(f"COPY ({SQL_TITLED.format(f=tmp)}) TO '{p}' (FORMAT PARQUET)")
        con.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return "ok"

# ════════════════════════════════════════════════════════════════════
# 3. Stage 1b — 플레이어 집계 및 층화 표집
# 표집 규칙 (Phase 1 확정):
#   - 최소 30판, 전 층 동일
#   - 층 배정은 기간 내 레이팅 중앙값
#   - 하위 4개 층 각 500명 무작위, FM+ 전수
#   - 1300 미만 제외, BOT 제외(Stage 1에서 이미 처리)
# ════════════════════════════════════════════════════════════════════

SEED = 20260816

_SQL_AGG = """
    SELECT player,
           max(title)                       AS title,
           sum(n_games)                     AS n_games,
           sum(n_eval_games)                AS n_eval_games,
           -- 샤드별 중앙값의 가중평균으로 기간 중앙값 근사
           sum(elo_median * n_games) / sum(n_games) AS elo_median,
           min(elo_min) AS elo_min, max(elo_max) AS elo_max
    FROM read_parquet('{glob}')
    GROUP BY player
"""


def aggregate_players(players_glob=None):
    """샤드별 집계를 플레이어 단위로 합산"""
    players_glob = players_glob or f"{STAGE1}/*.parquet"
    con = duckdb.connect()
    df = con.execute(_SQL_AGG.format(glob=players_glob)).df()
    con.close()
    return df


def aggregate_titled(players_glob=None):
    """타이틀 전용 집계 (stage1_titled 산출물)"""
    players_glob = players_glob or f"{STAGE1_FM}/*.parquet"
    return aggregate_players(players_glob)


def assign_tier_row(row):
    t = row["title"]
    if t in TITLES_DROP:
        return None
    if t in TITLES_KEEP:
        return TITLE_TIER
    e = row["elo_median"]
    if pd.isna(e) or e < 1300:
        return None
    for name, lo, hi in TIERS:
        if lo <= e < hi:
            return name
    return None


def build_sample(df, min_games=MIN_GAMES, per_tier=PER_TIER, seed=SEED):
    df = df.copy()
    df["tier"] = df.apply(assign_tier_row, axis=1)
    df = df[df.tier.notna()]

    elig = df[df.n_games >= min_games]
    out = []
    for name, _, _ in TIERS:
        pool = elig[elig.tier == name]
        n = min(per_tier, len(pool))
        out.append(pool.sample(n=n, random_state=seed))
    tpool = elig[elig.tier == TITLE_TIER]
    out.append(tpool)                     # 타이틀 층 전수
    return df, elig, pd.concat(out, ignore_index=True)


def build_titled_sample(df_titled, min_games=MIN_GAMES):
    """타이틀 층 전수 표본. 층 배정은 타이틀 기준이므로 레이팅과 무관하다."""
    d = df_titled.copy()
    d = d[d.title.isin(TITLES_KEEP)]
    d = d[d.n_games >= min_games]
    d["tier"] = TITLE_TIER
    return d.reset_index(drop=True)


def sample_report(df, elig, samp):
    lines = []
    lines.append(f"전체 플레이어(층 배정됨): {len(df):,}")
    lines.append(f"≥{MIN_GAMES}판 통과:        {len(elig):,}  ({100*len(elig)/len(df):.1f}%)")
    lines.append("")
    lines.append(f"{'층':>12} {'전체':>9} {'≥30판':>8} {'통과율':>7} {'표집':>7} {'eval≥30':>8}")
    lines.append("-" * 58)
    for name in [t[0] for t in TIERS] + [TITLE_TIER]:
        a = df[df.tier == name]
        e = elig[elig.tier == name]
        s = samp[samp.tier == name]
        ev = a[a.n_eval_games >= MIN_GAMES]
        rate = 100 * len(e) / len(a) if len(a) else 0
        lines.append(f"{name:>12} {len(a):>9,} {len(e):>8,} {rate:>6.1f}% {len(s):>7,} {len(ev):>8,}")
    lines.append("-" * 58)
    lines.append(f"{'합계':>12} {len(df):>9,} {len(elig):>8,} {'':>7} {len(samp):>7,}")
    return "\n".join(lines)


def build_all():
    ensure_dirs(SAMPLE, SAMPLE_FM, PLAYERS_ALL)
    df = aggregate_players()
    df, elig, samp = build_sample(df)

    # 타이틀 층: 별도 집계가 있으면 그것으로 교체한다
    fm = None
    if os.path.isdir(STAGE1_FM) and any(
            f.endswith(".parquet") for f in os.listdir(STAGE1_FM)):
        fm = build_titled_sample(aggregate_titled())
        samp = pd.concat([samp[samp.tier != TITLE_TIER], fm], ignore_index=True)
        fm.to_parquet(SAMPLE_FM, index=False)
        print(f"타이틀 층 {len(fm):,}명 → {SAMPLE_FM}")
    else:
        print(f"경고: {STAGE1_FM} 이 비어 있다. "
              f"src/stage1_titled.py 를 먼저 돌려야 타이틀 층이 채워진다.")

    print(sample_report(df, elig, samp))
    samp.to_parquet(SAMPLE, index=False)
    df.to_parquet(PLAYERS_ALL, index=False)
    print(f"\n표본 {len(samp):,}명 → {SAMPLE}")

# ════════════════════════════════════════════════════════════════════
# 4. Stage 2 — 대국 재현 + 플라이 수준 특징
# 산출 컬럼:
#   game_id, player, color, tier, ply, move_time, clk_before, clk_after,
#   wp_before, wp_after, wp_delta,   ← 사건 A용 (eval 있는 대국만)
#   see_loss,                        ← 사전등록판(철회) 정의용
#   mat_diff,                        ← 사건 B용. 수를 두기 직전,
#                                      수를 두는 쪽 관점의 재료 차이
#   n_legal, max_see_mine, n_checks, ← 복잡도 대리 및 예측 모형 입력
#   nag, is_castle, is_capture, ply_total, termination
# ════════════════════════════════════════════════════════════════════

GAME_COLS = ["Site", "White", "Black", "WhiteTitle", "BlackTitle",
        "WhiteElo", "BlackElo", "Termination", "movetext"]

SCHEMA = ["game_id", "player", "color", "tier", "ply", "move_time",
          "clk_before", "clk_after", "wp_before", "wp_after", "wp_delta",
          "see_loss", "n_legal", "nag", "is_castle", "is_capture",
          "ply_total", "termination",
          "max_see_mine", "mat_diff", "n_checks"]


def assign_tier_from_game(elo, title):
    if title in TITLES_DROP:
        return None
    if title in TITLES_KEEP:
        return TITLE_TIER
    if elo is None or elo < 1300:
        return None
    for name, lo, hi in TIERS:
        if lo <= elo < hi:
            return name
    return None


def pick_side(site):
    h = int(hashlib.md5(str(site).encode()).hexdigest()[:8], 16)
    return h % 2


def max_opponent_capture_see(board):
    """
    현재 국면에서 수를 둘 쪽(=상대)이 얻을 수 있는 최대 SEE.
    > 0 이면 직전에 둔 플레이어가 재료를 순손실한 것.

    최적화: 잡기 대상 칸별로 한 번만 SEE를 계산한다.
    같은 칸을 여러 기물이 잡을 수 있어도 SEE는 내부적으로
    가장 값싼 공격자부터 순서대로 사용하므로 결과가 같다.
    값이 큰 표적부터 검사해 조기 종료한다.
    """
    me = board.turn
    targets = []
    for sq in chess.scan_forward(board.occupied_co[not me]):
        p = board.piece_at(sq)
        if p is None or VALUES[p.piece_type] == 0:
            continue
        if board.attackers_mask(me, sq):
            targets.append((VALUES[p.piece_type], sq))
    if not targets:
        best = 0
    else:
        targets.sort(reverse=True)
        best = 0
        for v, sq in targets:
            if v <= best:
                break                   # 남은 표적은 현재 최선을 못 넘김
            for mv in board.generate_legal_moves(chess.BB_ALL, chess.BB_SQUARES[sq]):
                if not board.is_capture(mv):
                    continue            # 승격 등 비잡기 수 제외
                s = see(board, mv)
                if s > best:
                    best = s

    # 앙파상: 목표 칸이 비어 있어 위 표적 탐색에 잡히지 않음
    if board.ep_square is not None:
        for mv in board.generate_legal_moves(chess.BB_ALL,
                                             chess.BB_SQUARES[board.ep_square]):
            if board.is_en_passant(mv):
                s = see(board, mv)
                if s > best:
                    best = s
    return best


def extract_game(site, white, black, wt, bt, we, be, term, mt,
                 keep_players=None, do_see=True, tier_map=None):
    color = pick_side(site)
    player = white if color == 0 else black
    if keep_players is not None and player not in keep_players:
        other = black if color == 0 else white
        if other in keep_players:
            color, player = 1 - color, other
        else:
            return None

    # 층 배정: 표집 시 확정된 층(기간 내 레이팅 중앙값 기준)을 사용한다.
    # 게임별 레이팅으로 매기면 레이팅이 오르내릴 때 같은 사람이 여러 층에
    # 걸쳐 층 간 비교가 오염된다. tier_map이 없을 때만 게임별로 배정한다.
    if tier_map is not None:
        tier = tier_map.get(player)
    else:
        title = wt if color == 0 else bt
        elo = we if color == 0 else be
        tier = assign_tier_from_game(elo, title)
    if tier is None:
        return None

    clks, evals, sans, nags = parse_movetext(mt)
    if len(sans) < OPENING_CUT + 4:
        return None

    mtimes = move_times(clks, color)
    wps = player_winprobs(evals, color)
    if not mtimes:
        return None

    # 보드 추적: SEE + 합법수 + 복잡도 예측용 특징
    see_by_ply, legal_by_ply, cap_by_ply = {}, {}, {}
    feat_by_ply = {}
    if do_see:
        board = chess.Board()
        for i, san in enumerate(sans):
            try:
                mv = board.parse_san(san)
            except Exception:
                break
            if i % 2 == color and i >= OPENING_CUT and i in mtimes:
                legal_by_ply[i] = board.legal_moves.count()
                cap_by_ply[i] = board.is_capture(mv)
                # 복잡도 예측용 (강건성 검증 전용, 사전등록 외)
                caps = [m for m in board.legal_moves if board.is_capture(m)]
                best_mine = 0
                for m in caps:
                    v = see(board, m)
                    if v > best_mine:
                        best_mine = v
                me = board.turn
                mat = {}
                for c in (True, False):
                    mat[c] = sum(VALUES[p] * len(board.pieces(p, c))
                                 for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                           chess.ROOK, chess.QUEEN))
                feat_by_ply[i] = (
                    best_mine,
                    mat[me] - mat[not me],
                    sum(1 for m in board.legal_moves if board.gives_check(m)),
                )
                board.push(mv)
                see_by_ply[i] = max_opponent_capture_see(board)
                board.pop()
            board.push(mv)

    rows = []
    for ply, sec in mtimes.items():
        if ply < OPENING_CUT:
            continue
        wp_a, wp_b = wps.get(ply), wps.get(ply - 1)
        clk_a = clks[ply]
        rows.append((
            str(site), player, color, tier, ply, sec,
            (clk_a + sec) if clk_a is not None else None, clk_a,
            wp_b, wp_a,
            (wp_a - wp_b) if (wp_a is not None and wp_b is not None) else None,
            see_by_ply.get(ply), legal_by_ply.get(ply),
            nags[ply], sans[ply].startswith("O-O"), cap_by_ply.get(ply),
            len(sans), term,
            *(feat_by_ply.get(ply) or (None, None, None)),
        ))
    return rows


def sample_fens(path, n=2000, seed=0, keep_players=None, tier_map=None):
    """
    복잡도 검증용 국면 표집.

    plies 에는 FEN 을 저장하지 않으므로(용량) 여기서 다시 재현해 뽑는다.
    반환 열: game_id, player, tier, ply, fen, n_legal, max_see_mine,
             mat_diff, n_checks
    """
    rng = np.random.default_rng(seed)
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT {','.join(GAME_COLS)} FROM read_parquet('{path}') "
        f"WHERE TimeControl='{TIME_CONTROL}' AND Event LIKE 'Rated%' "
        f"AND (WhiteTitle IS NULL OR WhiteTitle<>'BOT') "
        f"AND (BlackTitle IS NULL OR BlackTitle<>'BOT')").fetchall()
    con.close()

    out = []
    for r in rows:
        site, white, black, wt, bt, we, be, term, mt = r
        color = pick_side(site)
        player = white if color == 0 else black
        if keep_players is not None and player not in keep_players:
            other = black if color == 0 else white
            if other not in keep_players:
                continue
            color, player = 1 - color, other
        tier = (tier_map or {}).get(player) or assign_tier_from_game(
            we if color == 0 else be, wt if color == 0 else bt)
        if tier is None:
            continue
        clks, evals, sans, nags = parse_movetext(mt)
        if len(sans) < OPENING_CUT + 4:
            continue
        mtimes = move_times(clks, color)
        board = chess.Board()
        for i, san in enumerate(sans):
            try:
                mv = board.parse_san(san)
            except Exception:
                break
            if i % 2 == color and i >= OPENING_CUT and i in mtimes                     and rng.random() < 0.02:
                caps = [m for m in board.legal_moves if board.is_capture(m)]
                best_mine = max((see(board, m) for m in caps), default=0)
                me = board.turn
                mat = {c: sum(VALUES[pt] * len(board.pieces(pt, c))
                              for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                         chess.ROOK, chess.QUEEN))
                       for c in (True, False)}
                out.append({
                    "game_id": str(site), "player": player, "tier": tier,
                    "ply": i, "fen": board.fen(),
                    "n_legal": board.legal_moves.count(),
                    "max_see_mine": best_mine,
                    "mat_diff": mat[me] - mat[not me],
                    "n_checks": sum(1 for m in board.legal_moves
                                    if board.gives_check(m)),
                })
            board.push(mv)
        if len(out) >= n:
            break
    return pd.DataFrame(out[:n])


def extract_shard(path, out_path, keep_players=None, eval_only=False, do_see=True,
              tier_map=None):
    con = duckdb.connect()
    q = (f"SELECT {','.join(GAME_COLS)} FROM read_parquet('{path}') "
         f"WHERE TimeControl='{TIME_CONTROL}' AND Event LIKE 'Rated%' "
         f"AND (WhiteTitle IS NULL OR WhiteTitle<>'BOT') "
         f"AND (BlackTitle IS NULL OR BlackTitle<>'BOT')")
    if eval_only:
        q += " AND movetext LIKE '%eval%'"
    rows = con.execute(q).fetchall()
    con.close()

    out = []
    for r in rows:
        g = extract_game(*r, keep_players=keep_players, do_see=do_see,
                         tier_map=tier_map)
        if g:
            out.extend(g)
    df = pd.DataFrame(out, columns=SCHEMA)
    df.to_parquet(out_path, index=False)
    return len(df), df["game_id"].nunique()
