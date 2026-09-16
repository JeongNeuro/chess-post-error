"""
Every transcribed value must appear in the manuscript.

src/paper_check.py holds the manuscript's numbers so they can be compared
against data/derived. Those numbers are typed in by hand, which is the one
step in the whole chain with no check on it — and it failed in exactly that
way during preparation: a value was filled in from the repository instead of
the paper, so the comparison agreed with itself and a stale section went
unnoticed.

docs/paper_values.txt is extracted mechanically from the manuscript source.
This test asserts that every number in paper_check appears there, which makes
"I typed a number the paper does not contain" impossible to miss.

It cannot check the reverse — a paper value nobody transcribed is simply not
compared — nor whether a value sits in the right constant. It closes the one
gap where a number can be invented.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
VALUES = ROOT / "docs" / "paper_values.txt"


def _manuscript_numbers():
    """Every number the manuscript states, as a set of rounded floats."""
    if not VALUES.exists():
        pytest.skip(f"{VALUES.name} is absent")
    out = set()
    # The paper writes ".366" without its leading zero, so the pattern has to
    # allow a bare decimal point as well as a leading digit.
    for tok in re.findall(r"[+-]?(?:\d[\d,]*\.?\d*|\.\d+)",
                          VALUES.read_text(encoding="utf-8")):
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        out.add(round(abs(v), 3))
        # ".685" is written without its leading zero in the paper
        if abs(v) < 1:
            out.add(round(abs(v), 3))
    return out


def _transcribed():
    """(name, value) for every number in paper_check's constants."""
    from src import paper_check as pc

    out = []

    def walk(name, obj):
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            out.append((name, float(obj)))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{name}[{k!r}]", v)
        elif isinstance(obj, (list, tuple, set)):
            for i, v in enumerate(obj):
                walk(f"{name}[{i}]", v)

    for nm in dir(pc):
        if nm.isupper() and not nm.startswith("_"):
            walk(nm, getattr(pc, nm))
    return out


# Numbers that are not claims in the manuscript: tolerances the check itself
# uses, and structural counts.
NOT_FROM_PAPER = {
    "TOL", "TOL_RANGE", "MIXED_VS_PAIRED_MAX", "CALIPER_VARS_MAX_GAP",
    "RELIABILITY_SIGMA_B_RATIO_MAX", "RELIABILITY_THRESHOLD_VALUE",
    "TIER_COUNT_IN_TEXT", "RELIABILITY_CROSSING", "CONTRAST_ROUGHLY",
    "N_MOVES_TITLE", "N_EVENTS_MATERIAL_REGISTERED", "EVENTS_PER_GAME",
    "TABLE1", "RELIABILITY_THRESHOLD", "MIXED_MAIN_PLAYERS",
    "TIERS", "TITLE_TIER", "TITLES_KEEP", "SEED",
    "DOSE_T_MIN", "TIER_MATERIAL_T_MIN", "ROBUSTNESS_TAU3_ATTENUATION",
    "COMPOSITION", "FLANKER", "FLANKER_CURVE",
    "REVISION_SHARE_ZERO", "REVISION_SHARE_GAIN",
    "GROUP_DIFF_SE_T1", "MIXED_SE",
    # N_EVENTS_MATERIAL, N_EVENTS_BLUNDER, EVENTS_PER_GAME and N_PLAYERS were
    # exempt here. They are claims the abstract makes, so exempting them let
    # the counts be filled from the repository without anything objecting --
    # the same vacuous comparison this file exists to prevent. They are
    # checked now.
    # Facts about the repository, not claims in the paper. N_PLAYERS_ANALYSED
    # exists precisely because it differs from the manuscript's 2,136, and
    # BLUNDER_THRESH_PP is imported from config to define the event.
    "N_PLAYERS_ANALYSED", "BLUNDER_THRESH_PP", "N_GAMES_ANALYSED",
}


def test_every_transcribed_value_appears_in_the_manuscript():
    paper = _manuscript_numbers()
    missing = []
    for name, v in _transcribed():
        root = name.split("[")[0]
        if root in NOT_FROM_PAPER:
            continue
        if root == "RELIABILITY" and name.endswith(("[2]", "[3]")):
            # Events per player and player counts sit in Table 2. The
            # extractor reads running text and does not see table cells.
            continue
        if round(abs(v), 3) not in paper:
            missing.append(f"{name} = {v}")
    assert not missing, (
        "these are in paper_check but not in the manuscript:\n  "
        + "\n  ".join(sorted(missing)))


def test_the_extract_is_not_empty():
    """A broken extract would make the test above pass by accident."""
    paper = _manuscript_numbers()
    assert len(paper) > 80, f"only {len(paper)} numbers found — check the extract"
    for known in (0.209, 0.407, 0.537, 0.685, 0.103):
        assert known in paper, f"{known} missing — the extract looks wrong"


# ---------------------------------------------------------------------------
# Section-scoped check.
#
# The test above only asks whether a number appears somewhere in the paper,
# which a value swapped between constants would survive: +0.219 belongs to the
# dose-response section, so writing it into GROUPS passes. Tying each constant
# to the section it was read from closes that.
# ---------------------------------------------------------------------------

SECTION_OF = {
    "GROUPS": "Effects by Co-occurrence of Material Loss",
    "GROUP_DIFF": "Effects by Co-occurrence of Material Loss",
    "EVENT_MOVE_BLUNDER_LOSS": "Effects by Co-occurrence of Material Loss",
    "TIER_RANGES": "When the Effect Occurs",
    "TIER_RANGE_WIDTH": "When the Effect Occurs",
    "DOSE_SELF": "Magnitude of Material Lost",
    "DOSE_OPP": "Direction of loss",
    "DOSE_RANGE": "Direction of loss",
    "EVENT_MOVE_DOSE": "Direction of loss",
    "LEGAL_BIN_DIFF_RANGE": "Number of available options",
    "LEGAL_BIN_DIFF_FLAT": "Number of available options",
    "QUEEN_BIN_MAX": "Number of available options",
    "LEGAL_BIN_NINE_POINT_MAX": "Number of available options",
    "RELIABILITY": "Individual-Level Reliability",
    "RELIABILITY_CURVE": "Individual-Level Reliability",
    "RELIABILITY_CURVE_TITLED": "Individual-Level Reliability",
    "RELIABILITY_CURVE_BLUNDER": "Individual-Level Reliability",
    "ABSTRACT_RELIABILITY": "Individual-Level Reliability",
    "ROBUSTNESS": "Robustness",
    "CALIPER_RANGE": "Robustness",
    "CALIPER_VARS": "Robustness",
    "MIXED": "Robustness",
}


def _by_section():
    text = VALUES.read_text(encoding="utf-8")
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"\[(.+?)\]", line.strip())
        if m:
            cur = m.group(1)
            out.setdefault(cur, set())
            continue
        if cur:
            for tok in re.findall(r"[+-]?(?:\d[\d,]*\.?\d*|\.\d+)", line):
                try:
                    out[cur].add(round(abs(float(tok.replace(",", ""))), 3))
                except ValueError:
                    pass
    return out


def test_values_come_from_the_section_they_claim():
    from src import paper_check as pc

    sections = _by_section()
    missing = []
    for const, sec in SECTION_OF.items():
        if sec not in sections:
            continue
        obj = getattr(pc, const, None)
        if obj is None:
            continue
        vals = []

        def walk(o):
            if isinstance(o, bool):
                return
            if isinstance(o, (int, float)):
                vals.append(float(o))
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple, set)):
                for v in o:
                    walk(v)

        walk(obj)
        for v in vals:
            if abs(v) >= 1000 or float(v).is_integer():
                continue          # counts and player totals live in tables
            if abs(v) > 2:
                # t values. The section extract reads the LaTeX source and
                # does not capture "t = 12.3", and a t is effect over SE --
                # derived from two numbers this test already checks, not an
                # independent claim. The whole-paper test above still sees it.
                continue
            if round(abs(v), 3) not in sections[sec]:
                missing.append(f"{const} = {v} not in [{sec}]")
    assert not missing, "\n  " + "\n  ".join(sorted(missing))
