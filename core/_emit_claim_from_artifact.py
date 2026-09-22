r"""Re-run the claim emitter against the canonical artefact, then verify.

`emit_snippet` only runs at the end of a full sweep, which costs a GPU pass
over 15 (seed, dataset) pairs. The artefact already holds every row, so the
emitter can be driven from it directly -- and it must be, because the literal
is pasted into the audit verbatim. Re-typing it by hand is the transcription
error the emitter exists to prevent, and the p value in particular is a
*rendered* two-significant-figure number, not the raw statistic.

Then the fixed point is closed: the emitted claim is checked against the two
things the audit will check it against.

  claim vs artefact    tol 5e-2          (is the number still true?)
  claim vs manuscript  tol 1e-9          (is it the number the paper prints?)

The second is the binding one and the one a human transcription breaks: the
claim has to be `pct()`'s one-decimal rounding, not the raw statistic, and
the site extractor is what turns the printed text back into a float. Reading
the sites here rather than trusting the paste means a claim that agrees with
the artefact but disagrees with the paper gets caught before the audit runs.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import audit_paper_numbers as A  # noqa: E402
import analysis_retrained_matched_state as M  # noqa: E402

# Where the manuscript may sit, resolved relative to this file rather than
# hard-coded to one machine's path: the bundle-relative `paper/` the other
# audits look in, then the source tree this bundle was cut from (two levels
# above the bundle root), which is where it actually lives during development.
# Neither present is a loud `no manuscript found` below, never a silent read
# of whatever happens to sit at some developer's path.
_BUNDLE = Path(__file__).resolve().parent.parent
_SOURCE = _BUNDLE.parent.parent
# Default to the real manuscript: after `--apply` it *is* the substituted
# text, and that is the file the audit reads. Pass a path to check the dry-run
# copy instead -- which is what the pre-paste verification does, since the
# site patterns are all anchored on post-substitution wording and would
# (correctly) miss on the unmodified manuscript.
_CLI = [a for a in sys.argv[1:] if not a.startswith("-")]
CANDIDATES = ([Path(_CLI[0])] if _CLI
              else [_BUNDLE / "paper" / "ICASSP2027_armvdta.tex",
                    _SOURCE / "ICASSP2027_armvdta.tex"])

ART = M.OUT / "retrained_matched_state.json"
art = json.loads(ART.read_text(encoding="utf-8"))
assert art.get("complete"), "artefact is not a complete 15-pair sweep"
rows = art["rows"]
print(f"driving emitter from {ART.name}: {len(rows)} pairs, "
      f"p={art['p_value']:.6g}\n")

M.emit_snippet(rows, float(art["p_value"]))

# `emit_snippet`'s own guard already compared each value against the raw
# statistic at the audit's tolerance and printed nothing, so reaching here
# means the numeric half is closed. Re-derive the literal from the same
# formulas so the site half below has something to compare against.
deltas = [r["delta_pct"] for r in rows]
med = M.statistics.median(deltas)
p_rendered = float(f"{float(art['p_value']):.1e}".split("e")[0]) * 10 ** int(
    f"{float(art['p_value']):.1e}".split("e")[1])
claim = {
    "median": float(f"{med:.1f}"),
    "mean": float(f"{M.statistics.mean(deltas):.1f}"),
    "p": p_rendered,
    **{ds: float(f"{M.statistics.median([r['delta_pct'] for r in rows if r['dataset'] == ds]):.1f}")
       for ds in M.DATASETS},
    "ratio": float(round(21.0 / med)),
}

src = next((p for p in CANDIDATES if p.exists()), None)
if src is None:
    raise SystemExit("no manuscript found")
sites = A.retrained_sites(src.read_text(encoding="utf-8"))
print(f"\n--- claim vs manuscript sites ({src.name}) ---")
bad = []
for kind, (want_n, _) in A.RETRAINED_SITES.items():
    got = sites[kind]
    if len(got) != want_n:
        bad.append(kind)
        print(f"  [MISS] {kind:7s} {len(got)}/{want_n} sites found")
        continue
    uniq = sorted(set(got))
    # ratio is emitted as a whole number and parsed as a float by the site
    # regex; compare on the value, not the formatting.
    ok = len(uniq) == 1 and abs(uniq[0] - claim[kind]) <= 1e-9
    if not ok:
        bad.append(kind)
    print(f"  [{'ok ' if ok else 'FAIL'}] {kind:7s} {len(got)}/{want_n} "
          f"manuscript={uniq} claim={claim[kind]:g}")

print()
if bad:
    print(f"NOT READY -- claim disagrees with the manuscript on: {', '.join(bad)}")
    sys.exit(1)
print("CLAIM CHAIN VERIFIED -- artefact, emitter and manuscript agree on all "
      f"{len(A.RETRAINED_SITES)} values")
