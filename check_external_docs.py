#!/usr/bin/env python3
"""
원고 밖 문서의 수치를 원고와 대조한다.

  python check_external_docs.py --tex paper.tex

verify-paper 는 저장소 안만 본다. cover letter, README, 프리프린트 표지처럼
저장소 밖에 있으면서 원고 수치를 인용하는 문서는 어떤 검사에도 걸리지
않는다. 이 스크립트가 그 틈을 메운다.

대조 대상은 파일이 아니라 **주장**이다. 제목의 이동 수, 사건 수, 플레이어
수처럼 원고가 말하는 것을 다른 곳에서 다르게 말하고 있는지 본다.
"""
import argparse
import io
import os
import re
import sys

# 원고에서 뽑을 주장과 그 정규식
CLAIMS = {
    "title_moves":   (r'Evidence From ([\d.]+) Million Moves', "제목의 이동 수"),
    "n_material":    (r'([\d,]+)\s*\n?material-loss events', "기물 손실 사건"),
    "n_blunder":     (r'and ([\d,]+) blunders from', "블런더 사건"),
    "n_players":     (r'from ([\d,]+) players across five', "분석 플레이어"),
    "n_tiers":       (r'across (five|four) rating\s*\n?tiers', "층 수"),
}

# 원고가 버린 값. 표현이 달라도 이 숫자가 남아 있으면 낡은 문서다.
#
# 문구 기반 비교만으로는 "our study of 3.4 million moves" 처럼 달리 쓴
# 문장을 잡지 못한다. 원고를 고칠 때 여기에 옛 값을 한 줄 더하면, 표현을
# 어떻게 바꾸든 남아 있는 한 걸린다.
SUPERSEDED = {
    "title_moves": ["3.4"],
    "n_material":  ["887,287", "887287", "1.585", "1,585,000"],
    "n_blunder":   ["137,490", "137490"],
    "n_players":   ["2,136", "2136", "1,878", "1878"],
}

# 검사할 문서. 없으면 조용히 건너뛴다.
#
# cover letter 는 원고 옆에 있으므로 현재 폴더에서 찾는다. 저장소의 README
# 는 다르다 — 이 검사는 보통 원고 폴더에서 `--tex paper.tex` 로 돌리고,
# 그때 "README.md" 는 저장소의 것을 가리키지 않는다. 그래서 조용히 건너뛰고,
# 실제로 저장소 README 에 낡은 3.4 Million 이 그대로 남아 있었다.
#
# 저장소 README 는 현재 폴더가 아니라 **이 스크립트의 위치**에서 찾는다.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_README = os.path.join(HERE, "README.md")

DOCS = [
    "cover_letter.md",
    "cover_letter_EN_plain.txt",
    "cover_letter_KR.md",
    REPO_README,
]


def read(p):
    try:
        return io.open(p, encoding="utf-8").read()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="paper.tex")
    ap.add_argument("--docs", nargs="*", default=None)
    a = ap.parse_args()

    tex = read(a.tex)
    if tex is None:
        sys.exit(f"원고를 열 수 없다: {a.tex}")

    # 원고에서 주장을 뽑는다
    truth = {}
    for key, (pat, label) in CLAIMS.items():
        m = re.search(pat, tex)
        if m:
            truth[key] = (m.group(1).replace(",", ""), label)

    if not truth:
        sys.exit("원고에서 주장을 하나도 뽑지 못했다. 정규식을 확인할 것.")

    print("원고가 말하는 것")
    print("-" * 52)
    for k, (v, label) in truth.items():
        print(f"  {label:<16} {v}")

    docs = a.docs if a.docs else DOCS
    bad = 0
    checked = 0
    seen = set()

    for d in docs:
        key = os.path.realpath(d)
        if key in seen:
            continue
        seen.add(key)
        s = read(d)
        if s is None:
            # 저장소 README 는 늘 있어야 한다. 없다면 건너뛸 일이 아니라
            # 검사가 제 대상을 놓치고 있다는 뜻이다.
            if d == REPO_README:
                print("")
                print(f"✗ 저장소 README 를 열 수 없다: {d}")
                bad += 1
            continue
        checked += 1
        issues = []

        # (1) 같은 문구를 쓰는 경우 — 값을 직접 비교한다.
        for key, (pat, label) in CLAIMS.items():
            if key not in truth:
                continue
            for m in re.finditer(pat, s):
                got = m.group(1).replace(",", "")
                if got != truth[key][0]:
                    issues.append(f"{label}: {got} (원고 {truth[key][0]})")

        # (2) 문구가 다른 경우 — 값만 보고 낡은 수치를 찾는다.
        #
        # (1) 은 원고의 문장 형태를 그대로 요구한다. Cover letter 는 같은
        # 사실을 "our study of 3.4 million chess moves" 처럼 달리 쓰므로
        # 정규식이 아무것도 잡지 못하고 조용히 통과한다. 실제로 그렇게
        # 통과하는 것을 확인했다.
        #
        # 그래서 표현과 무관하게, 원고가 **버린** 값이 문서에 남아 있는지
        # 본다. 원고에서 뽑은 현재 값과 짝이 되는 옛 값 목록이 필요하다.
        for key, (cur, label) in truth.items():
            for stale in SUPERSEDED.get(key, []):
                if re.search(rf'(?<![\d.]){re.escape(stale)}(?![\d])', s):
                    issues.append(
                        f"{label}: 낡은 값 {stale} 이 남아 있다 (현재 {cur})")

        # 원고에 없는 숫자가 주장처럼 쓰였는지
        if issues:
            bad += len(issues)
            print(f"\n✗ {d}")
            for i in issues:
                print(f"    {i}")
        else:
            print(f"\n○ {os.path.basename(d)}")

    print("\n" + "=" * 52)
    if checked == 0:
        sys.exit("검사한 문서가 없다. --docs 로 경로를 줄 것.")
    print(f"{checked}개 문서 · 불일치 {bad}건")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
