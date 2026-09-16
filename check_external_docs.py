#!/usr/bin/env python3
"""
Check documents outside the manuscript against the manuscript.

  python check_external_docs.py --tex paper.tex

verify-paper sees only what is inside this repository. A cover letter, a
preprint title page or this README quotes the same figures from outside it and
is checked by nothing. This script closes that gap.

What it compares is not files but **claims**: the move count in the title, the
event counts, the number of players — whether something the manuscript states
is stated differently somewhere else.
"""
import argparse
import io
import os
import re
import sys

# The claims to read out of the manuscript, and the pattern that finds each.
CLAIMS = {
    "title_moves":   (r'Evidence From ([\d.]+) Million Moves', "moves in title"),
    "n_material":    (r'([\d,]+)\s*\n?material-loss events', "material-loss events"),
    "n_blunder":     (r'and ([\d,]+) blunders from', "blunder events"),
    "n_players":     (r'from ([\d,]+) players across five', "players analysed"),
    "n_tiers":       (r'across (five|four) rating\s*\n?tiers', "rating tiers"),
}

# Values the manuscript has discarded. However a document words it, if one of
# these numbers is still in it, the document is stale.
#
# Comparing wording alone cannot catch a sentence like "our study of 3.4
# million moves", which states the superseded figure in a form the manuscript
# never uses. Adding the old value here when the manuscript changes makes it
# catchable no matter how it is phrased.
SUPERSEDED = {
    "title_moves": ["3.4"],
    "n_material":  ["887,287", "887287", "1.585", "1,585,000"],
    "n_blunder":   ["137,490", "137490"],
    "n_players":   ["2,136", "2136", "1,878", "1878"],
}

# Documents to check. A missing one is skipped without comment.
#
# The cover letters sit beside the manuscript, so they are looked up in the
# working directory. The repository README is different: this check is normally
# run from the manuscript's folder with `--tex paper.tex`, and "README.md" does
# not resolve to the repository's copy there. It was skipped in silence that
# way, and the README carried a superseded move count for as long as it was.
#
# So the repository README is located from **this script's own directory**,
# not from the working directory.
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
        sys.exit(f"cannot open the manuscript: {a.tex}")

    # Read the claims out of the manuscript.
    truth = {}
    for key, (pat, label) in CLAIMS.items():
        m = re.search(pat, tex)
        if m:
            truth[key] = (m.group(1).replace(",", ""), label)

    if not truth:
        sys.exit("no claim could be read from the manuscript - check the patterns.")

    print("What the manuscript says")
    print("-" * 52)
    for k, (v, label) in truth.items():
        print(f"  {label:<22} {v}")

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
            # The repository README is always present. Its absence is not
            # something to skip over: it means the check is missing its subject.
            if d == REPO_README:
                print("")
                print(f"X cannot open the repository README: {d}")
                bad += 1
            continue
        checked += 1
        issues = []

        # (1) Where the wording is shared, compare the values directly.
        for key, (pat, label) in CLAIMS.items():
            if key not in truth:
                continue
            for m in re.finditer(pat, s):
                got = m.group(1).replace(",", "")
                if got != truth[key][0]:
                    issues.append(f"{label}: {got} (manuscript says {truth[key][0]})")

        # (2) Where it is not, look at the values alone.
        #
        # Pass (1) requires the manuscript's own sentence form. A cover letter
        # states the same fact differently -- "our study of 3.4 million chess
        # moves" -- so no pattern matches and it passes in silence. That was
        # confirmed against a document written exactly that way.
        #
        # This pass ignores wording and asks whether a value the manuscript has
        # **discarded** is still present, which is what SUPERSEDED holds.
        for key, (cur, label) in truth.items():
            for stale in SUPERSEDED.get(key, []):
                if re.search(rf'(?<![\d.]){re.escape(stale)}(?![\d])', s):
                    issues.append(
                        f"{label}: superseded value {stale} is still here "
                        f"(now {cur})")

        if issues:
            bad += len(issues)
            print(f"\nX {d}")
            for i in issues:
                print(f"    {i}")
        else:
            print(f"\nok {os.path.basename(d)}")

    print("\n" + "=" * 52)
    if checked == 0:
        sys.exit("no document was checked. Pass paths with --docs.")
    print(f"{checked} documents - {bad} mismatches")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
