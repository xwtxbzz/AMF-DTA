"""Which numeric literals in the paper does no audit check actually pin?

`audit_paper_numbers.py` and `audit_baseline_rows.py` each verify a long
list of claims, but nothing says the list is *complete*: adding a sentence
with a fresh number leaves 100+ passing checks and one unverified figure.
Reading the paper looking for mistakes finds the ones one happens to read
(that is how the missing "23% of uniform's per-pair cost" and a wrong
ensemble ratio survived an earlier pass).

Read the pinned-value lists the two audits write when they run, and report
the paper's literals that appear in none of them. A literal that shows up
is not necessarily wrong -- it may be a hyperparameter, a formula constant,
a dataset property, or a claim the audit reproduces in a different form (a
percentage where the paper quotes a ratio) -- but each one is then a
decision made deliberately rather than by omission.

Run the two audits first; this only reads their output.

Usage: python audit_coverage.py            # list uncovered literals
       python audit_coverage.py --context  # with the sentence each sits in
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED = ("audit_pinned_paper_numbers.json", "audit_pinned_baseline_rows.json",
          "audit_pinned_architecture.json")
TEX = HERE.parent / "paper" / "ICASSP2027_armvdta.tex"
# Floats anywhere, plus *integer* percentages ("47\%"), which the float
# pattern would miss entirely -- 47% of KIBA samples switching view is a
# claim, and it is the only form several of them take.
NUM = re.compile(r"\d+\.\d+|\d+(?=\\?%)")


def matches(value: float, pinned: set[float]) -> bool:
    """Is this literal pinned, allowing for sign and percent-vs-fraction?

    The paper writes improvements as positive ("$1.2\\%$") while an audit
    that reports a delta as a difference claims -1.2, and it writes a
    coverage as "69\\%" where the audit computes 0.69. Comparing only
    identical representations would report those as gaps and bury the
    ones that are real.
    """
    for h in pinned:
        for a, b in ((abs(value), abs(h)), (abs(value), abs(100 * h)),
                     (abs(100 * value), abs(h))):
            if abs(a - b) <= max(5e-4, 0.005 * a):
                return True
    return False


def claimed_values() -> set[float]:
    """The values the audits pinned, as they recorded them when last run."""
    values: set[float] = set()
    for name in PINNED:
        path = HERE / "outputs" / name
        if not path.exists():
            raise SystemExit(f"missing {path.name}: run the audits first")
        values.update(json.loads(path.read_text(encoding="utf-8")))
    return values


def paper_literals() -> list[tuple[float, int, str]]:
    """(value, line number, line text) for the body, references excluded."""
    # The manuscript is deliberately not shipped with this bundle, and this
    # script is listed as a reproduction command, so skip rather than traceback
    # -- the same way audit_paper_numbers.py degrades on the same missing file.
    if not TEX.exists():
        print(f"  [skip] manuscript not found at {TEX}")
        raise SystemExit(0)
    src = TEX.read_text(encoding="utf-8")
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("%"))
    body = body.split("Acknowledgment")[0]
    out = []
    for i, line in enumerate(body.splitlines(), 1):
        for tok in NUM.findall(line):
            out.append((float(tok), i, line.strip()))
    return out


def main() -> None:
    have = claimed_values()
    lits = paper_literals()
    uncovered = [(v, n, t) for v, n, t in lits if not matches(v, have)]

    print(f"{len(lits)} numeric literals in the paper body, "
          f"{len({v for v, _, _ in lits})} distinct")
    print(f"{len(have)} distinct values pinned by the two audits")
    print(f"{len(uncovered)} literal occurrences are not pinned by any "
          f"audit check:\n")
    ctx = "--context" in sys.argv
    for v, n, text in uncovered:
        if ctx:
            print(f"  {v:>8} line {n:4d}  {text[:92]}")
        else:
            print(f"  {v:>8}  (line {n})")
    print("\nEach line above is either a non-claim (hyperparameter, formula "
          "constant,\ndataset property) or a gap. Values in the paper that "
          "the audit recomputes in a\ndifferent form (ratios, percentages) "
          "appear here by construction.")


if __name__ == "__main__":
    main()
