"""List every numeric literal in the paper body, for audit-coverage checking.

Written after a patch of manual reading found two numbers that no existing
check covered and that were both wrong (the ensemble Q4/Q1 ratios, and a
claim that M1 was the best of three interaction slots when Table I shows
M2 with the lower average). Reading the paper looking for errors finds the
errors one happens to read; this enumerates what there is to check.

Usage: python extract_paper_numbers.py [--with-context]
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

TEX = Path(__file__).resolve().parent.parent / "paper" / "ICASSP2027_armvdta.tex"

# `$0.4327$`, `0.89`, `$-0.0029$`, and \allowbreak-joined runs.
NUM = re.compile(r"\d+\.\d+")


def main() -> None:
    src = TEX.read_text(encoding="utf-8")
    body = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("%"))
    # References and the funding note are not claims about our experiments.
    body = body.split("Acknowledgment")[0]

    ctx = "--with-context" in sys.argv
    if ctx:
        for i, line in enumerate(body.splitlines(), 1):
            found = NUM.findall(line)
            if found:
                print(f"{i:4d}  {found}  | {line.strip()[:96]}")
        return

    counts = Counter(NUM.findall(body))
    print(f"{len(counts)} distinct numeric literals in the paper body\n")
    for value, n in sorted(counts.items(), key=lambda kv: float(kv[0])):
        print(f"  {value:>9s} x{n}")


if __name__ == "__main__":
    main()
