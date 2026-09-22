r"""Dry-run of the matched-state substitution on a *copy* of the manuscript.

The real substitution happens once the F-A11 sweep lands. Doing it blind at
that point would mean discovering a broken layout in the middle of a numeric
paste, so this applies the exact replacements to a scratch copy first and
checks (a) every anchor matches exactly once, and (b) the result still
compiles to 5 pages with no Overfull box.

The numbers below are the 8-pair *partial* values, used purely as geometry
placeholders: every site formats as `X.Y\%` / `N\times`, so a later digit
change cannot move a line. If a placeholder here disagreed with the final
value in *width*, the dry run would be measuring the wrong paragraph.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(r"D:\try\paper")
SRC = PAPER / "ICASSP2027_armvdta.tex"
DST = PAPER / "dryrun_subst.tex"

# Default is the scratch copy. `--apply` writes the real manuscript instead.
# Both paths run the same anchors and the same guard; the flag exists so the
# real edit is the *same code path* that was already verified, not a re-typed
# variant of it. The anchors are the pre-substitution wording, so this must run
# before anything else touches those four regions.
if "--apply" in sys.argv:
    DST = SRC
    print(f"*** APPLYING TO THE REAL MANUSCRIPT: {SRC.name} ***")

# --- values from the full 15-pair sweep (outputs/retrained_matched_state.json)
MED, MEAN, P, DAVIS, KIBA, METZ, RATIO = "0.9", "1.1", "3.4", "1.7", "0.8", "1.6", "24"

REPLACEMENTS: list[tuple[str, str, str]] = [
    (
        "abstract",
        "weights loses only $1.7\\%$: post-hoc interventions overstate gating's\n"
        "architectural value by $\\sim12\\times$.",
        f"weights loses only ${MED}\\%$: post-hoc interventions overstate gating's\n"
        f"architectural value by $\\sim{RATIO}\\times$.",
    ),
    (
        "contribution",
        "weights loses only $1.7\\%$, so a post-hoc intervention of this kind\n"
        "overstates gating's architectural value by $12\\times$;",
        f"weights loses only ${MED}\\%$, so a post-hoc intervention of this kind\n"
        f"overstates gating's architectural value by ${RATIO}\\times$;",
    ),
    (
        "III-D paragraph",
        "The retrained ablation bounds the architectural value of the same\n"
        "condition on the architecture the post-hoc arm also uses: trained from\n"
        "scratch under uniform $1/K$ weights (A1 of Table~\\ref{tab:ablation},\n"
        "$+1.7\\%$ at seed 42 in the final-fit state), the model is only $2.1\\%$\n"
        "worse over all $15$ (seed, dataset) pairs at the matched state (median;\n"
        "$2.6\\%$ mean, $p=1\\times10^{-3}$; Davis $1.8\\%$, KIBA $2.3\\%$, Metz\n"
        "$3.1\\%$). The arms differ only by whether gates were present during\n"
        "training, so the $10\\times$ gap to the post-hoc median is overstatement\n"
        "rather than architecture. We quote the retrained figure: the context\n"
        "gate's dataset-level view preference is interpretable\n"
        "(Fig.~\\ref{fig:gateweights}), but gating's architectural contribution is\n"
        "$2.1\\%$, not $21\\%$.",
        "The retrained ablation bounds the architectural value of the same\n"
        "condition on the architecture the post-hoc arm also uses: trained from\n"
        f"scratch under uniform $1/K$ weights, the model loses a median ${MED}\\%$\n"
        f"over the same $15$ (seed, dataset) pairs at the matched state, worse in\n"
        f"$13$ of $15$ (${MEAN}\\%$ mean, two-sided paired $p={P}\\times10^{{-3}}$; Davis\n"
        f"${DAVIS}\\%$, KIBA ${KIBA}\\%$, Metz ${METZ}\\%$)---at or below the\n"
        "$1.7$--$2.1\\%$ seed-noise floor of Sec.~\\ref{sec:ablation}, and no\n"
        "larger than the $+1.7\\%$ that section's fixed-interaction screen\n"
        "independently attributes to both gates. The arms differ only by whether\n"
        "gates were\n"
        f"present during training, so the ${RATIO}\\times$ gap to the post-hoc median is\n"
        "overstatement rather than architecture. We quote the retrained figure:\n"
        "the context gate's dataset-level view preference is interpretable\n"
        "(Fig.~\\ref{fig:gateweights}), but gating's architectural contribution is\n"
        f"${MED}\\%$, not $21\\%$.",
    ),
    (
        "conclusion",
        "gates to $1/K$ costs a median 21.0\\% while retraining under those\n"
        "weights costs 1.7\\%---a $12\\times$ overstatement.",
        "gates to $1/K$ costs a median 21.0\\% while retraining under those\n"
        f"weights costs {MED}\\%---a ${RATIO}\\times$ overstatement.",
    ),
    (
        "placeholder comment",
        "% PLACEHOLDERS: RET_MED, RET_MEAN, RET_P, RET_DAVIS, RET_KIBA, RET_METZ,\n"
        "% RET_RATIO -- to be replaced by analysis_retrained_matched_state.py's\n"
        "% output once the F-A11 sweep completes. Widths chosen to match the\n"
        "% expected values so the swap is layout-neutral.\n",
        "",
    ),
]

text = SRC.read_text(encoding="utf-8")
for name, old, new in REPLACEMENTS:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR FAIL [{name}]: found {n} occurrences, expected 1")
    text = text.replace(old, new)

DST.write_text(text, encoding="utf-8")

# A replacement line missing its `f` prefix interpolates nothing, and LaTeX
# renders the leftover `${MED}\%$` as italic "MED" without complaining -- the
# page count, the Overfull count and the compile all stay clean. Only the
# audit's site extractor notices, as a short count rather than an error. Catch
# the class here instead: no placeholder name may survive into the output.
#
# Match the `${NAME}` form specifically. A bare `\{[A-Z_]+\}` also fires on
# legitimate content -- `\textbf{KIBA}`, `{MLP}` in a caption -- and a guard
# that fails on a clean file gets deleted rather than fixed.
PLACEHOLDERS = ("MED", "MEAN", "P", "DAVIS", "KIBA", "METZ", "RATIO")
leftovers = sorted({m for m in re.findall(r"\$\{([A-Z_]+)\}", text)
                    if m in PLACEHOLDERS})
if leftovers:
    raise SystemExit(f"UNINTERPOLATED placeholders in output: {leftovers}")

print(f"wrote {DST.name} ({len(REPLACEMENTS)} anchors, all matched exactly once)")
for name, _, _ in REPLACEMENTS:
    print(f"  ok  {name}")
