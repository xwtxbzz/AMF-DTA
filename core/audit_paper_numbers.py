"""Re-derive every number the paper reports, from the run directories.

Written because the paper's own tables and the run artefacts drifted apart
more than once during revision (a protocol description that said
best-validation while the code averaged the final fit; a Table I row that was
the RAIL model rather than M1). Each check prints what the paper claims, what
the artefacts give, and whether they agree -- so a mismatch is visible rather
than inferred.

Sources:
  outputs/rail_ab_F<tag>_seed<s>_200/<ds>/test_metrics.json   F-A* and F-M* runs
  outputs/summary_raildta_5seeds_200.json                     RAIL single + FingerDTA
  outputs/gate_intervention_fa3.json                          post-hoc intervention
  outputs/retrained_matched_state.json                        F-A11 at best_state
  outputs/dataset_gate_matched_state.json                     F-D1 at best_state

Usage: python audit_paper_numbers.py
"""
from __future__ import annotations

import csv
import inspect
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist
from scipy.stats import wilcoxon
from statsmodels.stats.power import TTestPower

OUT = Path(__file__).resolve().parent / "outputs"
DATASETS = ("davis", "kiba", "metz")
SEEDS = (42, 52, 62, 72, 82)
# The five ensemble members. `rail_single_seed*` is the "single" member: its
# name carries the family, unlike the members, which are `rail_m{1..4}_seed*`.
MEMBERS_ENS = ("single", "m1", "m2", "m3", "m4")
# What the manuscript claims for the retrained no-gate arm at matched state,
# in the notation of Sec. III-D. Emitted by
# `analysis_retrained_matched_state.py`'s `emit_snippet` and pasted verbatim,
# not transcribed: the audit compares these to the manuscript at tol 1e-9, so
# each entry must be the number the paper *prints* (`pct()`'s one-decimal
# rounding, and `p` at two significant figures), not the raw statistic. A raw
# statistic here fails by up to 0.05 on every site -- which is exactly the
# transcription error the emitter exists to remove.
RETRAINED_CLAIM: dict[str, float | None] = {
    "median": 0.9, "mean": 1.1, "p": 0.0034,
    "davis": 1.7, "kiba": 0.8, "metz": 1.6, "ratio": 24,
    "dataset_gate": 1.0,
}
OK, BAD = [], []
# Every value this run claimed, for `audit_coverage.py`: a passing 100-check
# audit says nothing about the 101st number unless something compares the
# set of checks against the set of literals the paper contains. Collected
# here at runtime rather than parsed statically, because most claims reach
# check() through loop tuples that a static reader would miss.
PINNED: list[float] = []


# The retrained figures are quoted in four separate places (abstract,
# contribution list, Sec. III-D twice, conclusion). Anchored regexes rather
# than a bare number scan, because `1.7` and `2.1` legitimately appear
# elsewhere -- the seed-noise floor is "1.7--2.1%", the post-hoc per-condition
# deltas are "+0.3%, +1.7%", and Table II's A1 row is "+1.7%" -- so a token
# search would both flag innocent text and miss a partial update. Each pattern
# captures the number from one site; all captures must agree.
RETRAINED_SITES = {
    # kind: (how many copies must be found, [patterns])
    "median": (3, [r"\$([\d.]+)\\%\$: the intervention overstates",   # abstract
                   # `\\%`, not `\%`: the manuscript's percent sign is LaTeX's
                   # `\%`, so the pattern needs a literal backslash *and* a
                   # percent. A single backslash yields the regex `\%`, which
                   # matches a bare `%` and silently finds nothing. The same
                   # doubling applies to every macro below (`\\textbf`, `\\sim`,
                   # `\\times`): bare, the regex reads `\t` and `\s`, i.e. a tab
                   # and a whitespace class, and matches nothing at all.
                   r"costs a median \$([\d.]+)\\%\$ over the",     # III-D
                   r"without them costs ([\d.]+)\\%---an overstatement"]),  # conclusion
    # Every pattern quotes its own sentence's wording, so a reworded site drops
    # out of its list and the check fails on a short list rather than passing on
    # fewer comparisons than intended. That is the guard -- but it also means
    # these anchors go stale whenever the prose is edited, because they are
    # quoted from the manuscript rather than from the claim. An earlier revision
    # of the retrained sentences silently left three of them capturing nothing.
    #
    # Three sites hedge the ratio with `\\sim`, in two LaTeX spellings, and the
    # abstract now does too: the ratio is a quotient of two *printed*
    # one-decimal medians (21.0 / 0.9 = 23.3 -> 23) while the true quotient is
    # 24.08 -> 24, so a reader cannot reproduce it from the operands on the
    # page. "~24x" is reproducible by both readings; a bare "24x" is off by one
    # against the paper's own numbers, which is what a reviewer recomputes first.
    #
    # The abstract says "ratio of medians", not "median": the paper separately
    # reports the median of per-pair ratios (12.5--16.5x), so a bare "median"
    # here reads as that other estimator, which is roughly half as large.
    "ratio": (4, [r"ratio of medians \\textbf\{\$\\sim\$([\d.]+)\$\\times\$\}, never below",  # abstract
                  r"\(\$\\sim\$([\d.]+)\$\\times\$, never below",         # contribution list
                  r"a \$\\sim([\d.]+)\\times\$ overstatement",            # III-D
                  r"\$\\sim\$([\d.]+)\$\\times\$ by our estimator"]),     # conclusion
    # Anchored on the "mean, paired" run: the sentence leads with the median and
    # states the 13-of-15 split, so the old "matched state (median;" anchor no
    # longer exists. `paired` alone, not `two-sided paired`: the manuscript
    # reserves the full protocol name for the p-value clause.
    "mean": (1, [r"\$([\d.]+)\\%\$ mean, paired"]),
    # Anchored on "paired $p=", not "mean, $p=". The manuscript now spells the
    # protocol out ("two-sided paired $p=...") because this p-value sits one
    # clause away from the seed-noise floor, and without the word the two read
    # as contradictory; "paired" is the paper's own term for the protocol
    # (Sec. III-B), so it is the more stable of the two anchors.
    # `[\d.]+`, not `\d+`: the manuscript quotes this p to two significant
    # figures ("3.4\times10^{-3}"). One figure would round by up to a third,
    # which exceeds the 5% tolerance this claim is checked at.
    "p": (1, [r"paired \$p=([\d.]+)\\times10\^\{(-?\d+)\}\$"]),
    "davis": (1, [r"Davis \$([\d.]+)\\%\$, KIBA"]),
    "kiba": (1, [r"KIBA \$([\d.]+)\\%\$, Metz"]),
    "metz": (1, [r"Metz \$([\d.]+)\\%\$\)"]),
    # F-D1, the retrained counterpart of the `both_mean` probe. One site, in
    # Sec. III-D: the abstract and conclusion state what the result does to
    # the reading without quoting the number, deliberately, so the paper
    # cannot end up citing two different values for one measurement. The
    # optional `\+` is because this delta is printed signed, unlike the
    # no-gate medians above -- a bare `[\d.]+` would capture from `1.0` and
    # silently drop the sign the sentence depends on.
    "dataset_gate": (1, [r"costs a median \$(\+?[\d.]+)\\%\$ against the probe"]),
}


def retrained_sites(tex_text: str) -> dict[str, list[float]]:
    """Extract every site's copy of the retrained figures from the manuscript.

    Returns the captured numbers per kind. A site whose wording changes enough
    to break its pattern yields a *short* list, which the caller reports --
    silently comparing fewer sites than intended is the one outcome this must
    not have.
    """
    import re
    body = re.sub(r"\s+", " ", tex_text)
    out: dict[str, list[float]] = {}
    for kind, (_, pats) in RETRAINED_SITES.items():
        vals: list[float] = []
        for pat in pats:
            for m in re.findall(pat, body):
                # `p` captures (mantissa, exponent); everything else is scalar.
                vals.append(float(m[0]) * 10 ** int(m[1]) if kind == "p"
                            else float(m))
        out[kind] = vals
    return out


def check(label: str, claimed, actual, tol: float = 5e-4, fmt: str = "{:.4f}") -> None:
    """Compare a paper claim against a re-derived value."""
    if isinstance(claimed, (int, float)) and not isinstance(claimed, bool):
        PINNED.append(round(float(claimed), 6))
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        good = abs(claimed - actual) <= tol
        shown = f"paper={fmt.format(claimed)}  derived={fmt.format(actual)}"
    else:
        good = claimed == actual
        shown = f"paper={claimed!r}  derived={actual!r}"
    (OK if good else BAD).append((label, shown))
    print(f"  [{'ok ' if good else 'MISMATCH'}] {label:46s} {shown}")


def ab(tag: str, seed: int, ds: str) -> float | None:
    p = OUT / f"rail_ab_{tag}_seed{seed}_200" / ds / "test_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["rmse"]


def vector(tag: str, ds: str) -> list[float]:
    """Per-seed RMSEs for one variant and dataset, seeds in paper order."""
    return [v for s in SEEDS if (v := ab(tag, s, ds)) is not None]


def mean_std(tag: str) -> tuple[float, float]:
    """Three-dataset average RMSE, and its std *over seeds*.

    The std must be taken across seeds, not across datasets: each seed
    contributes one three-dataset average, and those five numbers are the
    replicate sample. Averaging over seeds first (one number per dataset)
    then taking a std measures the Davis/KIBA/Metz spread instead, roughly
    ten times larger -- which is what an earlier version of this script did.
    """
    per_seed_avg = [statistics.mean([ab(tag, s, ds) for ds in DATASETS]) for s in SEEDS]
    return (statistics.mean(per_seed_avg), statistics.stdev(per_seed_avg))


def main() -> None:
    summary = json.loads((OUT / "summary_raildta_5seeds_200.json").read_text(encoding="utf-8"))
    fd = summary["fingerdta_dagger_per_seed"]

    print("=== Table I: matched protocol ===")
    # Every upper-group row is *checked*, not merely printed: the ensemble
    # member `rail_m2` has the same three-dataset average as the
    # FingerDTA-style row (.4356) by coincidence while differing on all
    # three datasets and on its std, so an eyeball comparison of averages
    # is not enough to tell which run backs which row.

    def from_summary(block: str, key: str) -> None:
        """Per-dataset means, avg +- std, and CI for a summary block."""
        per_ds = [statistics.mean(summary[block][s][ds][key] for s in map(str, SEEDS))
                  for ds in DATASETS]
        per_seed = [statistics.mean(summary[block][s][ds]["rmse"] for ds in DATASETS)
                    for s in map(str, SEEDS)]
        return (per_ds, statistics.mean(per_seed), statistics.stdev(per_seed),
                statistics.mean(statistics.mean(summary[block][s][ds]["ci"]
                                                for s in map(str, SEEDS))
                                for ds in DATASETS))

    pd_r, m_r, s_r, c_r = from_summary("single_swa_per_seed", "rmse")
    for ds, want in zip(DATASETS, (.4527, .3562, .5056)):
        check(f"Table I RAIL {ds}", want, round(pd_r[DATASETS.index(ds)], 4))
    check("Table I RAIL avg", .4381, round(m_r, 4))
    check("Table I RAIL std", .0090, round(s_r, 4))
    check("Table I RAIL CI", .8765, round(c_r, 4))

    pd_f = [statistics.mean(fd[s][ds]["rmse"] for s in map(str, SEEDS))
            for ds in DATASETS]
    per_f = [statistics.mean(fd[s][ds]["rmse"] for ds in DATASETS)
             for s in map(str, SEEDS)]
    for ds, want in zip(DATASETS, (.4469, .3534, .5064)):
        check(f"Table I FingerDTA-style {ds}", want,
              round(pd_f[DATASETS.index(ds)], 4))
    check("Table I FingerDTA-style avg", .4356, round(statistics.mean(per_f), 4))
    check("Table I FingerDTA-style std", .0113, round(statistics.stdev(per_f), 4))
    check("Table I FingerDTA-style CI", .8773,
          round(statistics.mean(statistics.mean(fd[s][ds]["ci"]
                                                for s in map(str, SEEDS))
                                for ds in DATASETS), 4))

    for tag, name, want_pd, want_avg, want_std, want_ci in (
            ("FM1", "M1", (.4462, .3490, .5029), .4327, .0082, .8782),
            ("FM2", "M2", (.4505, .3477, .4987), .4323, .0072, .8782)):
        for ds, want in zip(DATASETS, want_pd):
            check(f"Table I {name} {ds}", want,
                  round(statistics.mean(vector(tag, ds)), 4))
        m, s = mean_std(tag)
        check(f"Table I {name} avg", want_avg, round(m, 4))
        check(f"Table I {name} std", want_std, round(s, 4))
        cis = [json.loads((OUT / f"rail_ab_{tag}_seed{sd}_200/{ds}"
                           / "test_metrics.json").read_text(encoding="utf-8"))["ci"]
               for sd in SEEDS for ds in DATASETS]
        check(f"Table I {name} CI", want_ci,
              round(statistics.mean(cis), 4) if len(cis) == 15 else None)

    print("\n=== Sec. III-C: paired statistics vs the FingerDTA-style row ===")
    # Three conventions here are easy to get wrong and were each doubted
    # once during revision, so all three are pinned rather than the point
    # estimate alone:
    #   * the interval is t_{14} = 2.1448, not 1.96. With z it would be
    #     [-0.0062, +0.0004] -- plausible-looking and wrong;
    #   * "~14 seeds for 80% power" is *pairs* converted to seeds. Read as a
    #     plain (z_{.975}+z_{.8})^2/d^2 it gives 40.4, which looks like a
    #     ten-fold error but is the pair count: 43/3 = 14.3 seeds, since
    #     three datasets ride on each seed;
    #   * the comparison is over 15 (seed, dataset) pairs, not 5 seeds, so a
    #     std over seeds would understate the interval.
    m1 = [ab("FM1", s, ds) for s in SEEDS for ds in DATASETS]
    m2 = [ab("FM2", s, ds) for s in SEEDS for ds in DATASETS]
    f_ = [fd[str(s)][ds]["rmse"] for s in SEEDS for ds in DATASETS]
    diff = [a - b for a, b in zip(m1, f_)]
    md, sd = statistics.mean(diff), statistics.stdev(diff)
    se = sd / math.sqrt(len(diff))
    tcrit = tdist.ppf(0.975, len(diff) - 1)
    check("M1 pooled paired diff (paper)", -0.0029, round(md, 4))
    check("M1 CI low, t_{14} (paper)", -0.0065, round(md - tcrit * se, 4))
    check("M1 CI high, t_{14} (paper)", 0.0007, round(md + tcrit * se, 4))
    check("M1 wins n/15 (paper)", 10,
          sum(1 for a, b in zip(m1, f_) if a < b), fmt="{:.0f}")
    for ds, want in zip(DATASETS, (-0.0008, -0.0045, -0.0034)):
        check(f"M1 {ds} paired diff (paper)", want,
              round(statistics.mean([ab("FM1", s, ds) - fd[str(s)][ds]["rmse"]
                                     for s in SEEDS]), 4))
    check("M1 Cohen's d (paper 0.44)", 0.44, round(abs(md / sd), 2),
          tol=0.005, fmt="{:.2f}")
    n80 = next(n for n in range(5, 90) if TTestPower().power(
        effect_size=abs(md / sd), nobs=n, alpha=0.05) >= 0.80)
    check("M1 seeds for 80% power (paper ~14)", 14, n80 / len(DATASETS),
          tol=0.5, fmt="{:.1f}")
    check("M1 vs M2 p (paper 0.45)", 0.45,
          round(wilcoxon(m1, m2).pvalue, 2), tol=0.005, fmt="{:.2f}")

    print("\n=== Sec. III-E: ensembling ===")
    # The five members are single+m1..m4; the "*_avg" field in the summary
    # file is a Davis-only aggregate and is NOT what the paper quotes.
    ens = [statistics.mean(summary["ensemble_per_seed"][s][ds]["rmse"]
                           for ds in DATASETS) for s in map(str, SEEDS)]
    check("ensemble avg RMSE", .4089, round(statistics.mean(ens), 4))
    check("ensemble CI", .8897,
          round(statistics.mean(statistics.mean(summary["ensemble_per_seed"][s][ds]["ci"]
                                                for ds in DATASETS)
                                for s in map(str, SEEDS)), 4))
    member_avgs = []
    for mem in ("single", "m1", "m2", "m3", "m4"):
        rows = [[json.loads((OUT / f"rail_{mem}_seed{s}_200/{ds}"
                             / "test_metrics.json").read_text(encoding="utf-8"))["rmse"]
                 for ds in DATASETS] for s in SEEDS]
        member_avgs.append(statistics.mean(statistics.mean(r) for r in rows))
    # 0.4382 is the mean over the five members; the single `single` member
    # alone is 0.438150, which rounds to 0.4381 -- the value the paper used
    # to quote, describing the ensemble's gain against one model rather
    # than against the average member.
    check("members individually", .4382, round(statistics.mean(member_avgs), 4))

    # "within 0.2%" for the two alternative aggregations. The number the
    # paper quotes for this traces to `autoresearch/ensemble_main_results.json`,
    # which is the *earlier* `autoresearch/outputs/{single,m1..m4}` family --
    # a different set of runs from the 0.4089 the same sentence reports. The
    # check below recomputes the claim on the members that actually back
    # 0.4089, so the sentence cannot go stale if only one family changes.
    devs = []
    for scheme in ("val", "sample"):
        per_seed = []
        for s in SEEDS:
            per_ds = []
            for ds in DATASETS:
                cols, vrs = [], []
                for mem in MEMBERS_ENS:
                    tag = (f"rail_single_seed{s}_200" if mem == "single"
                           else f"rail_{mem}_seed{s}_200")
                    with (OUT / tag / ds / "test_predictions.csv").open(
                            encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    cols.append(np.asarray([float(r["y_pred"]) for r in rows]))
                    if scheme == "val":
                        with (OUT / tag / ds / "history.csv").open(
                                encoding="utf-8") as f:
                            vrs.append(min(float(r["val_rmse"])
                                           for r in csv.DictReader(f)))
                y = np.asarray([float(r["y_true"]) for r in rows])
                P = np.asarray(cols)
                equal = P.mean(axis=0)
                if scheme == "val":
                    w = 1.0 / np.maximum(np.asarray(vrs), 1e-8)
                    w = w / w.sum()
                    blended = (P * w.reshape(-1, 1)).sum(axis=0)
                else:
                    sw = 1.0 / np.maximum(np.abs(P - equal), 1e-8)
                    sw = sw / sw.sum(axis=0, keepdims=True)
                    blended = (P * sw).sum(axis=0)
                per_ds.append(float(np.sqrt(np.mean((y - blended) ** 2))))
            per_seed.append(statistics.mean(per_ds))
        devs.append(abs(100 * (statistics.mean(per_seed) - statistics.mean(ens))
                        / statistics.mean(ens)))
    check("alternative aggregations within (%)", 0.2, round(max(devs), 1),
          tol=0.1, fmt="{:.1f}")

    print("\n=== Sec. III-E: seed-42 cross-family ensemble ===")
    # The paper concedes that its 5-member ensemble is not evidence for the
    # interaction module, because the baseline's *own* ensemble matches it.
    # Both sides of that concession are rebuilt here from the raw test
    # predictions rather than from the summary file, so the check does not
    # depend on the summary having been regenerated.
    def ens_avg(fam: str, seed: int) -> float:
        single = (f"{fam}_single_seed{seed}_200" if fam == "rail"
                  else f"{fam}_seed{seed}_200")
        tags = [single] + [f"{fam}_m{i}_seed{seed}_200" for i in (1, 2, 3, 4)]
        per_ds = []
        for ds in DATASETS:
            cols = [np.loadtxt(OUT / t / ds / "test_predictions.csv",
                               delimiter=",", skiprows=1) for t in tags]
            y = cols[0][:, 0]
            for t, c in zip(tags, cols):
                assert np.allclose(c[:, 0], y), f"y_true mismatch {t} {ds}"
            per_ds.append(float(np.sqrt(np.mean(
                (y - np.mean([c[:, 1] for c in cols], axis=0)) ** 2))))
        return statistics.mean(per_ds)

    fens42 = ens_avg("fingerdta", 42)
    rail42 = ens_avg("rail", 42)
    check("F-Ens seed42 (baseline 5 members)", .3905, round(fens42, 4))
    check("RAIL ensemble seed42 (same 15 runs as avg)", .3935, round(ens[0], 4))
    check("RAIL ensemble seed42 (from predictions)", round(ens[0], 4), round(rail42, 4))
    check("baseline ensemble edge (%)", 0.8,
          round(100 * (rail42 - fens42) / fens42, 1), tol=0.05, fmt="{:.1f}")

    # The concession is scoped to one seed, so the scoping has to be true:
    # the sentence says seed 42 is the only seed where both ensembles exist.
    # If the seed-52 FingerDTA members are ever finished, this comparison
    # becomes a choice among seeds rather than a constraint, and the wording
    # (not just the check) needs to change.
    complete = [s for s in SEEDS
                if all((OUT / f"fingerdta_m{i}_seed{s}_200" / ds
                        / "test_predictions.csv").exists()
                       for i in (1, 2, 3, 4) for ds in DATASETS)]
    check("seeds with a complete baseline ensemble", "42",
          ",".join(map(str, complete)))

    print("\n=== Table II: F-A* ablation (5 seeds) ===")
    for tag in ("FA3", "FA7", "FA8", "FA10"):
        m, s = mean_std(tag)
        print(f"  {tag:5s} avg={m:.4f}+-{s:.4f}")

    print("\n=== Sec. III-C: M1/M2 vs RAIL, and vs FingerDTA ===")
    # The RAIL row's three-dataset average, re-derived above as m_r.
    railavg = m_r
    for tag, name, claimed in (("FM1", "M1", -1.2), ("FM2", "M2", -1.3)):
        m, _ = mean_std(tag)
        check(f"{name} vs RAIL improvement (%)", claimed,
              round(100 * (m - railavg) / railavg, 1), tol=0.05, fmt="{:+.1f}%")
    # Paired Wilcoxon, RAIL vs M1/M2, over the 15 (seed, dataset) pairs.
    for tag, name in (("FM1", "M1"), ("FM2", "M2")):
        a, b = [], []
        for s in map(str, SEEDS):
            for ds in DATASETS:
                if (v := ab(tag, int(s), ds)) is not None:
                    a.append(summary["single_swa_per_seed"][s][ds]["rmse"])
                    b.append(v)
        print(f"  {name} vs RAIL paired Wilcoxon p={wilcoxon(a, b).pvalue:.4f} "
              f"(paper: 0.0015/0.0043)  n={len(a)}")
    # Gated model vs the FingerDTA-style baseline, paired over 15 pairs.
    # Only M1's pooled figures are quoted in the paper; M2's differ
    # (-0.0033, CI [-0.0077,+0.0011], +0.0035/-0.0057/-0.0077 per dataset),
    # so the sentence now scopes them to M1.
    import numpy as _np
    from scipy.stats import t as _tdist
    base15 = [fd[s][ds]["rmse"] for s in map(str, SEEDS) for ds in DATASETS]
    for tag, name, want_p, want_wins in (("FM1", "M1", 0.15, 10),
                                         ("FM2", "M2", 0.14, 9)):
        got = [ab(tag, int(s), ds) for s in map(str, SEEDS) for ds in DATASETS]
        d = [a_ - b_ for a_, b_ in zip(got, base15)]
        check(f"{name} vs baseline p", want_p, round(wilcoxon(got, base15).pvalue, 2),
              tol=5e-3, fmt="{:.2f}")
        check(f"{name} wins vs baseline", want_wins, sum(1 for x in d if x < 0),
              fmt="{:.0f}")
        if name == "M1":
            hw = _tdist.ppf(0.975, len(d) - 1) * _np.std(d, ddof=1) / len(d) ** 0.5
            check("M1 pooled diff", -0.0029, round(statistics.mean(d), 4))
            check("M1 pooled CI half-width", 0.0036, round(hw, 4))
            for i, ds in enumerate(DATASETS):
                check(f"M1 per-dataset diff {ds}", (-0.0008, -0.0045, -0.0034)[i],
                      round(statistics.mean(d[i::3]), 4))

    # M1 vs M2: the paper used to call M1 "the best of three interaction
    # slots", which Table I contradicts (M2's average is lower, 0.4323 vs
    # 0.4327). They are in fact indistinguishable.
    m1 = [ab("FM1", s, ds) for s in SEEDS for ds in DATASETS]
    m2 = [ab("FM2", s, ds) for s in SEEDS for ds in DATASETS]
    check("M1 vs M2 p", 0.45, round(wilcoxon(m1, m2).pvalue, 2), tol=5e-3, fmt="{:.2f}")
    check("M1 better than M2 on n pairs", 4, sum(1 for x, y in zip(m1, m2) if x < y),
          fmt="{:.0f}")

    # F-A8 vs F-A3. Both the p-value and the interval half-width the paper
    # quotes next to it are claims, so neither stays a print.
    a, b = [], []
    for s in SEEDS:
        for ds in DATASETS:
            x, y = ab("FA3", s, ds), ab("FA8", s, ds)
            if x and y:
                a.append(x)
                b.append(y)
    if a:
        d8 = [y - x for x, y in zip(a, b)]
        # "its difference ... is bounded within +-1.3% (95% CI half-width
        # over the same 15 pairs)". The half-width is of the *absolute*
        # paired difference, quoted relative to the F-A3 mean: 0.00568 /
        # 0.4363 = 1.301%. Taken as the CI of the *relative* difference --
        # the more obvious reading -- it is 1.206%, which rounds to 1.2%.
        # Both are defensible; the paper's is the first, so the check uses
        # it and the other is recorded here rather than left to be
        # rediscovered as a suspected error.
        check("F-A8 vs F-A3 p", 0.28,
              round(wilcoxon(a, b).pvalue, 2), tol=5e-3, fmt="{:.2f}")
        check("F-A8 ahead on n/15", 10,
              sum(1 for x, y in zip(a, b) if x < y), fmt="{:.0f}")
        hw = tdist.ppf(0.975, len(d8) - 1) * statistics.stdev(d8) / math.sqrt(len(d8))
        check("F-A8 vs F-A3 bound (%)", 1.3,
              round(100 * hw / statistics.mean(a), 1), tol=0.05, fmt="{:.1f}%")

    print("\n=== Sec. III-D opening: gate weights (Fig. 2) ===")
    # Source: `gate_analysis.json`. Its *name* says nothing about which run
    # it describes -- analysis_gates.py writes that bare name only for its
    # DEFAULT_RUN, `rail_single_seed42_200` -- while the similarly named
    # `gate_analysis_rail_ab_FA3_seed42_200.json` holds a different model's
    # gates, and they differ qualitatively: davis drug is [0.000, 1.000,
    # 0.000] there against [0.390, 0.610, 0.000] here. An audit matching on
    # the family name would call the paper's Fig. 2 unsupported. The run
    # this asserts against is therefore named in the label.
    ga = json.loads((OUT / "gate_analysis.json").read_text(encoding="utf-8"))
    for ds, gate, idx, want, what in (
            ("davis", "drug", 0, 0.39, "Morgan"),
            ("davis", "drug", 1, 0.61, "feature-Morgan"),
            ("davis", "drug", 2, 0.00, "MACCS"),
            ("davis", "context", 1, 0.892, "target fingerprint"),
            ("kiba", "context", 2, 0.636, "sequence"),
            ("metz", "context", 2, 0.758, "sequence"),
            ("metz", "drug", 2, 0.002, "MACCS")):
        check(f"Fig2 rail_single_s42 {ds} {gate} {what}", want,
              round(ga[ds][gate]["mean_weights"][idx], 3), tol=5e-4, fmt="{:.3f}")
    # "KIBA uses all three views" is a claim about the floor, not one weight,
    # and it is now *only* a statement about what this figure plots (seed 42).
    # Across seeds KIBA's MACCS share falls as low as 0.3%, so the paper no
    # longer claims it of the dataset -- see the stability block below.
    check("Fig2 s42 only: kiba drug all three views", True,
          min(ga["kiba"]["drug"]["mean_weights"]) > 0.2)

    # --- what holds across seeds, which is what the text now claims ---
    # The version of this paragraph that rested on seed 42 alone was wrong in
    # three ways, each caught by running `analysis_gates.py` on the other four
    # seeds: Davis's drug gate puts 100% of its weight on Morgan at seed 72
    # (the exact opposite of seed 42's 39/61), KIBA's MACCS share runs 0.3% to
    # 21.7%, and Metz's context argmax leaves the sequence encoder at seed 82.
    # These checks read every gate_analysis file present, so adding a seed
    # extends the derived range and fails the check if the quoted interval no
    # longer covers it.
    runs = {42: ga}
    for p in sorted(OUT.glob("gate_analysis_rail_single_seed*_200.json")):
        runs[int(p.stem.split("_seed")[1].split("_")[0])] = json.loads(
            p.read_text(encoding="utf-8"))
    seeds = sorted(runs)
    print(f"       gate analyses available: {seeds}")

    def w(seed: int, ds: str, gate: str, idx: int) -> float:
        return runs[seed][ds][gate]["mean_weights"][idx]

    def argmax(seed: int, ds: str, gate: str) -> int:
        return runs[seed][ds][gate]["mean_weights"].index(
            max(runs[seed][ds][gate]["mean_weights"]))

    davis_ctx = [w(s, "davis", "context", 1) for s in seeds]
    kiba_ctx = [w(s, "kiba", "context", 2) for s in seeds]
    check("Fig2 davis ctx target-fp, low across seeds", 0.82,
          round(min(davis_ctx), 2), tol=5e-3, fmt="{:.2f}")
    check("Fig2 davis ctx target-fp, high across seeds", 0.94,
          round(max(davis_ctx), 2), tol=5e-3, fmt="{:.2f}")
    check("Fig2 davis ctx argmax = target-fp in all seeds", True,
          all(argmax(s, "davis", "context") == 1 for s in seeds))
    check("Fig2 kiba ctx sequence, low across seeds", 0.49,
          round(min(kiba_ctx), 2), tol=5e-3, fmt="{:.2f}")
    check("Fig2 kiba ctx sequence, high across seeds", 0.71,
          round(max(kiba_ctx), 2), tol=5e-3, fmt="{:.2f}")
    check("Fig2 kiba ctx argmax = sequence in all seeds", True,
          all(argmax(s, "kiba", "context") == 2 for s in seeds))
    # "on Metz in four of five" -- the one seed that leaves is why the paper
    # says four rather than five, so the count itself is the claim.
    check("Fig2 metz ctx argmax = sequence in four of five", True,
          sum(argmax(s, "metz", "context") == 2 for s in seeds) == 4)
    check("Fig2 davis drug MACCS stays low (<= 21%)", True,
          max(w(s, "davis", "drug", 2) for s in seeds) <= 0.21)
    # "the ratio ranges from 39/61 to 100/0": both ends are the claim.
    check("Fig2 davis drug, fcfp leads at some seed", True,
          any(w(s, "davis", "drug", 1) > w(s, "davis", "drug", 0)
              for s in seeds))
    check("Fig2 davis drug, morgan takes all at some seed", True,
          max(w(s, "davis", "drug", 0) for s in seeds) > 0.999)

    stds = [max(max(runs[s][ds][g]["std_across_samples"])
                for ds in DATASETS for g in ("drug", "context"))
            for s in seeds]
    check("Fig2 sample-level std, max across seeds", 0.35,
          round(max(stds), 2), tol=5e-3, fmt="{:.2f}")
    sw = [runs[s]["kiba"]["drug"]["argmax_switching_vs_global"]
          for s in seeds]
    check("Fig2 kiba drug switching, low across seeds", 0.28,
          round(min(sw), 2), tol=5e-3, fmt="{:.2f}")
    check("Fig2 kiba drug switching, high across seeds", 0.58,
          round(max(sw), 2), tol=5e-3, fmt="{:.2f}")
    rhos = [abs(r) for s in seeds for ds in DATASETS
            for g in ("drug", "context")
            for r in runs[s][ds][g]["rho_weight_error"]]
    check("Fig2 all |rho| <= 0.22 across seeds", True, max(rhos) <= 0.22)

    print("\n=== Sec. III-D: gate section claims ===")
    gi = json.loads((OUT / "gate_intervention_fa3.json").read_text(encoding="utf-8"))
    uni, pla, bth, base15, uni_rmse, bth_rmse = [], [], [], [], [], []
    for s in (42, 52, 62, 72, 82):
        for ds in DATASETS:
            c = gi["per_seed"][str(s)][ds]["conditions"]
            uni.append(c["uniform"]["delta_pct"])
            pla.append(c["placebo"]["delta_pct"])
            bth.append(c["both_mean"]["delta_pct"])
            base15.append(c["sample"]["rmse"])
            uni_rmse.append(c["uniform"]["rmse"])
            bth_rmse.append(c["both_mean"]["rmse"])
    sgn = lambda x, n: round(x, n) if x >= 0 else -round(-x, n)  # round-half-up
    check("uniform median (%)", 21.0, sgn(statistics.median(uni), 1),
          tol=0.05, fmt="{:+.1f}%")
    check("uniform mean (%)", 63, round(statistics.mean(uni)), tol=0.6, fmt="{:+.0f}%")
    # "(the mean, +63%, is inflated by two diverging Davis seeds)". The
    # sentence names a cause, not just a statistic, so the cause is checked:
    # the two most extreme pairs must both be Davis. It matters -- if the
    # tail were spread across datasets, dropping to the median because of
    # "two Davis seeds" would be a wrong reading of the same 15 numbers.
    worst = sorted(((gi["per_seed"][str(s)][ds]["conditions"]["uniform"]["delta_pct"], ds)
                    for s in (42, 52, 62, 72, 82) for ds in DATASETS),
                   reverse=True)[:2]
    check("two most extreme uniform pairs are Davis", 2,
          sum(1 for _, ds in worst if ds == "davis"), fmt="{:.0f}")
    print(f"       (they carry {100 * sum(v for v, _ in worst) / sum(uni):.0f}% "
          f"of the 15-pair sum: {', '.join(f'{v:+.0f}% {ds}' for v, ds in worst)})")
    # The paper quotes p to one significant figure ($6\times10^{-5}$), so the
    # tolerance is 5% of the claim, not a fixed absolute: 6.1e-5 is a correct
    # rounding of 6.1e-5, and an absolute 5e-7 tolerance would call it wrong.
    check("uniform p", 6e-5, round(wilcoxon(base15, uni_rmse).pvalue, 6),
          tol=3e-6, fmt="{:.1e}")
    check("uniform worse in n/15", 15, sum(1 for v in uni if v > 0), fmt="{:.0f}")
    # The paper's reading of the placebo rests on the median, because the
    # mean is dragged by the same diverging Davis seeds that inflate the
    # uniform mean -- so both statistics are pinned here, not just one.
    check("placebo median (%)", 3.4, round(statistics.median(pla), 1),
          tol=0.05, fmt="{:+.1f}%")
    check("placebo mean (%)", 6.4, round(statistics.mean(pla), 1),
          tol=0.05, fmt="{:+.1f}%")
    check("both_mean median (%)", 1.5, round(statistics.median(bth), 1),
          tol=0.05, fmt="{:+.1f}%")
    check("both_mean p", 6e-4, round(wilcoxon(base15, bth_rmse).pvalue, 6),
          tol=3e-5, fmt="{:.1e}")
    # Per-dataset medians: the sentence that carries the argument contrasts
    # the placebo against uniform separately on each dataset, so each of the
    # six numbers it quotes is checked rather than the pooled pair alone.
    for ds, want_uni, want_pla in (("davis", 65.0, 1.9), ("kiba", 12.7, 13.0),
                                   ("metz", 31.6, 2.8)):
        u = [gi["per_seed"][str(s)][ds]["conditions"]["uniform"]["delta_pct"]
             for s in (42, 52, 62, 72, 82)]
        p = [gi["per_seed"][str(s)][ds]["conditions"]["placebo"]["delta_pct"]
             for s in (42, 52, 62, 72, 82)]
        check(f"{ds} uniform median (%)", want_uni,
              round(statistics.median(u), 1), tol=0.05, fmt="{:+.1f}%")
        check(f"{ds} placebo median (%)", want_pla,
              round(statistics.median(p), 1), tol=0.05, fmt="{:+.1f}%")
    check("post-hoc baseline (paper 0.4469)", 0.4469,
          round(statistics.mean(base15), 4))

    print("\n=== Sec. III-D: retrained no-gate arm at matched state ===")
    # The "post-hoc overstates gating's value by Nx" clause is a *ratio of two
    # arms*, and review W1's condition was that both arms sit on the same
    # checkpoint state. The post-hoc arm runs on best_state.pt; Table II's
    # A1/A3 pair was measured at the final-fit SWA state, where the same
    # variant differs by 2.4% -- larger than the effect being reported.
    # `analysis_retrained_matched_state.py` moves the retrained arm to
    # best-validation so the ratio is like-for-like; this block checks the
    # paper quotes *those* numbers.
    ms_path = OUT / "retrained_matched_state.json"
    # "The arms differ only by whether gates were present during training" is
    # the sentence that licenses reading the two RMSEs as the architectural
    # value of gating. It is checkable at the config level and is checked
    # here, for every seed where both configs exist: the gated and no-gate
    # runs must be identical field-for-field apart from the two gate flags.
    # Any other difference (interaction mode, seed, epochs, SWA schedule)
    # would make the comparison an ablation of two things at once.
    gate_diff, gate_n = [], 0
    for s in SEEDS:
        a = OUT / f"rail_ab_FA3_seed{s}_200" / "config.json"
        b = OUT / f"rail_ab_FA11_seed{s}_200" / "config.json"
        if not (a.exists() and b.exists()):
            continue
        ca = json.loads(a.read_text(encoding="utf-8"))
        cb = json.loads(b.read_text(encoding="utf-8"))
        gate_n += 1
        diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
        if diff != {"use_drug_gate", "use_context_gate"}:
            gate_diff.append(f"seed{s}:{sorted(diff)}")
    # `gate_n > 0` keeps this from passing vacuously before any F-A11 run
    # writes its config: "no pairs compared" must not read as "no differences".
    check("gated/no-gate configs differ only in the two gate flags", True,
          gate_n > 0 and not gate_diff, fmt="{}")
    print(f"       compared {gate_n}/5 seed config pairs")
    if gate_diff:
        print(f"       unexpected config differences: {', '.join(gate_diff)}")

    # F-D1's counterpart to the block above: it must be F-A3 with only
    # `gate_mode` changed, or the retrained dataset-level figure measures a
    # different architecture. Compared on *effective* configs -- framework
    # defaults filled in -- because the two were written across a config
    # schema revision: F-D1 seeds 42/52/62 predate `split_method`/`cold_key`
    # being recorded while 72/82 carry them at exactly the values
    # `build_dataloaders` defaults to, so a raw key-set comparison reports a
    # difference that does not exist. That false failure is why the defaults
    # are resolved here rather than assumed absent.
    try:
        from framework.data import build_dataloaders as _build_dataloaders
        _defaults = {n: p.default for n, p in
                     inspect.signature(_build_dataloaders).parameters.items()
                     if p.default is not inspect.Parameter.empty}
    except Exception as exc:               # pragma: no cover - env-dependent
        print(f"  [warn] could not read framework defaults ({exc}); "
              f"comparing raw configs")
        _defaults = {}
    dg_diff, dg_n = [], 0
    for s in SEEDS:
        a = OUT / f"rail_ab_FA3_seed{s}_200" / "config.json"
        b = OUT / f"rail_ab_FD1_seed{s}_200" / "config.json"
        if not (a.exists() and b.exists()):
            continue
        ca = json.loads(a.read_text(encoding="utf-8"))
        cb = json.loads(b.read_text(encoding="utf-8"))
        for cfg in (ca, cb):
            for key, value in _defaults.items():
                cfg.setdefault(key, value)
        dg_n += 1
        diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
        if diff != {"gate_mode"}:
            dg_diff.append(f"seed{s}:{sorted(diff)}")
    check("F-D1/F-A3 configs differ only in gate_mode", True,
          dg_n > 0 and not dg_diff, fmt="{}")
    print(f"       compared {dg_n}/5 seed config pairs")
    if dg_diff:
        print(f"       unexpected config differences: {', '.join(dg_diff)}")
    if not ms_path.exists():
        print("  [skip] retrained_matched_state.json absent "
              "(F-A11 sweep incomplete)")
    else:
        ms = json.loads(ms_path.read_text(encoding="utf-8"))
        rows_m = ms["rows"]
        gi_ref = json.loads((OUT / "gate_intervention_fa3.json").read_text(
            encoding="utf-8"))
        # State-matching is the entire basis of the ratio and is invisible in
        # the paper, so it is checked rather than assumed: the gated arm must
        # reproduce the post-hoc run's own unmodified-baseline RMSE, which
        # pins the checkpoint, the loader and the denormalisation to be the
        # same ones. A silent mismatch here would make the ratio meaningless
        # while every downstream number still looked plausible.
        worst_m = max(abs(r["gated"] - gi_ref["per_seed"][str(r["seed"])]
                          [r["dataset"]]["conditions"]["sample"]["rmse"])
                      for r in rows_m)
        check("retrained gated arm == post-hoc baseline", True, worst_m < 1e-4,
              fmt="{}")
        print(f"       n={len(rows_m)} pairs, max |diff| = {worst_m:.2e}")
        # Staleness: the JSON is a derived artefact, so a re-run dataset would
        # leave it quoting old numbers. Every input must exist and predate it.
        # A *missing* input counts as bad, not as "fresh by default": the JSON
        # asserts a pair was measured, and a pair whose run directory is gone
        # can no longer be re-derived -- so the claim is unbacked either way.
        stale = []
        for r in rows_m:
            src = (OUT / f"rail_ab_FA11_seed{r['seed']}_200" / r["dataset"]
                   / "test_metrics.json")
            if not src.exists():
                stale.append(f"{r['seed']}/{r['dataset']} (missing)")
            elif src.stat().st_mtime > ms_path.stat().st_mtime + 1:
                stale.append(f"{r['seed']}/{r['dataset']} (newer)")
        check("retrained JSON backed by all 15 F-A11 runs", True, not stale,
              fmt="{}")
        if stale:
            print(f"       unbacked: {', '.join(stale)}")
        ds_med = {ds: statistics.median([r["delta_pct"] for r in rows_m
                                         if r["dataset"] == ds])
                  for ds in DATASETS}
        derived_m = {"median": statistics.median([r["delta_pct"] for r in rows_m]),
                     "mean": statistics.mean([r["delta_pct"] for r in rows_m]),
                     "p": float(ms["p_value"]),
                     "davis": ds_med["davis"], "kiba": ds_med["kiba"],
                     "metz": ds_med["metz"],
                     "ratio": 21.0 / statistics.median([r["delta_pct"]
                                                        for r in rows_m])}
        # F-D1 lives in its own artefact (it needs `best_state.pt`, so unlike
        # the other JSONs it is not re-derivable from the released bundle),
        # but it is checked through the same claim table as the no-gate arm so
        # that both retrained figures are pinned by one mechanism.
        dg_path = OUT / "dataset_gate_matched_state.json"
        if dg_path.exists():
            dg = json.loads(dg_path.read_text(encoding="utf-8"))
            derived_m["dataset_gate"] = statistics.median(
                [r["delta_pct"] for r in dg["rows"]])
            # The artefact is allowed to be short: seed 72's F-D1 sweep was
            # stopped, so the arm covers 13 of 15 pairs. What is *not*
            # allowed is being short and silent, because the paper quotes a
            # median from this file. The invariant is therefore "complete, or
            # self-describing" -- a short file must carry its own gap list, a
            # gap note, and the `finalized` flag that says the arm was closed
            # deliberately rather than left mid-run.
            gap = ([tuple(p) for p in dg.get("unevaluated_checkpoints", [])]
                   + [tuple(p) for p in dg.get("untrained_pairs", [])])
            self_describing = (bool(gap) and dg.get("finalized")
                               and dg.get("gap_note"))
            if not dg.get("complete", True) and not self_describing:
                BAD.append(("retrained dataset_gate",
                            f"artefact is partial and does not name its gap "
                            f"({dg.get('n_pairs')} pairs, "
                            f"finalized={dg.get('finalized')}, gap={gap})"))
                print(f"  [bad] dataset_gate artefact is partial and does not "
                      f"name its gap: {dg.get('n_pairs')} pairs, "
                      f"finalized={dg.get('finalized')}, gap={gap}")
            elif gap:
                n_seeds = len({r["seed"] for r in dg["rows"]})
                print(f"  [gap] dataset_gate artefact is short but "
                      f"self-describing: {dg.get('n_pairs')} pairs over "
                      f"{n_seeds} seeds, {len(gap)} pair(s) named")
        else:
            # Not `[skip]`-and-forget: the manuscript quotes this number, so a
            # missing artefact means the claim is unbacked. Only reachable
            # before the F-D1 top-up has been run.
            print("  [skip] dataset_gate_matched_state.json absent "
                  "(F-D1 top-up pending)")
        for name, derived in derived_m.items():
            want = RETRAINED_CLAIM[name]
            if want is None:
                # The artefact exists but the manuscript still carries the
                # placeholder. Reported as a failure rather than skipped: the
                # paper and this file are then known to disagree, and the only
                # safe reading of a green audit is "these agree".
                BAD.append((f"retrained {name}",
                            "artefact exists but paper claim not transcribed"))
                print(f"  [todo] retrained {name}: paper claim not transcribed"
                      f"  derived={derived:.4g}")
                continue
            tol = 5e-2 if name not in ("p", "ratio") else abs(want) * 0.05
            fmt = "{:+.1f}%" if name != "p" else "{:.1e}"
            check(f"retrained {name}", want, derived, tol=tol, fmt=fmt)

        # Claim -> manuscript. The check above ties the recorded claim to the
        # artefact; this ties the manuscript to the recorded claim, so the
        # chain has no unchecked link. It also requires every copy to be
        # *identical*: before this revision three of the four retrained sites
        # quoted the final-fit delta (1.7%/12x) while Sec. III-D quoted the
        # matched-state one (2.1%/10x) -- two different measurements of one
        # claim, unsignposted. A partial update is the failure this catches,
        # and it is worse than either value alone because the paper then
        # contradicts itself.
        # Overridable so the substitution can be audited on the dry-run copy
        # *before* it is applied. Pointing this at a scratch file is the only
        # way to run the site checks against prospective text: the patterns
        # are anchored on post-substitution wording and miss on the old
        # manuscript, so "audit passes" and "the edit is correct" would
        # otherwise have to be checked in the same step, when a failure no
        # longer tells you which of the two went wrong.
        tex_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
            OUT.parent.parent / "paper" / "ICASSP2027_armvdta.tex")
        if not tex_path.exists():
            print(f"  [skip] manuscript not found at {tex_path}")
        else:
            sites = retrained_sites(tex_path.read_text(encoding="utf-8"))
            for kind, (want_n, _) in RETRAINED_SITES.items():
                vals = sites[kind]
                agree = len(vals) == want_n and len(set(vals)) == 1
                check(f"retrained {kind}: {want_n} site(s), all identical",
                      True, agree, fmt="{}")
                if not agree:
                    print(f"       {kind}: expected {want_n} site(s), found "
                          f"{len(vals)}: {vals}")
                    continue
                want = RETRAINED_CLAIM[kind]
                if want is None:
                    print(f"  [todo] retrained {kind} site = {vals[0]:g}, "
                          f"claim not recorded")
                    continue
                check(f"retrained {kind} site matches claim", want, vals[0],
                      tol=1e-9, fmt="{:g}")

    print("\n=== Table II upper group: seed-42 fixed-interaction ablation ===")
    # These live in the *autoresearch* tree, not rail_dta/outputs. A second,
    # similarly named family (`autoresearch/outputs/icasp_a*_seed42`) holds
    # different numbers from a different run generation; matching on the
    # directory name alone silently compares the wrong family, which is how
    # an earlier pass wrongly concluded this table was unsupported.
    AR = OUT.parent.parent / "autoresearch" / "outputs"
    claimed = {"a1": (.4311, .8787), "a2": (.4250, .8785), "a3": (.4237, .8791),
               "a4": (.4257, .8779), "a5": (.4605, .8786), "a6": (.4218, .8793)}
    for tag, (pr, pc) in claimed.items():
        f = AR / tag / "summary.json"
        if not f.exists():
            check(f"Table II {tag}", "present", "missing")
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        check(f"Table II {tag} RMSE", pr, round(s["avg_rmse"], 4))
        check(f"Table II {tag} CI", pc, round(s["avg_ci"], 4))

    # The Delta column, which the argument rests on ("only the *product
    # interaction* exceeds seed noise") and which nothing pinned until
    # audit_coverage.py asked for it. Computed from the unrounded RMSEs:
    # taking the table's own four-decimal values instead moves a1 from
    # +1.746% to +1.747% and, more to the point, an earlier draft of this
    # table had +0.2% where +0.24% rounds to +0.2% only because the mean
    # difference is small -- reading it off the displayed means is how a
    # correct rounding starts looking like an off-by-one-decimal error.
    def autotag(tag: str) -> float | None:
        f = AR / tag / "summary.json"
        return (json.loads(f.read_text(encoding="utf-8"))["avg_rmse"]
                if f.exists() else None)

    a3 = autotag("a3")
    if a3:
        for tag, want in (("a1", 1.7), ("a2", 0.3), ("a4", 0.5),
                          ("a5", 8.7), ("a6", -0.4)):
            r = autotag(tag)
            if r is not None:
                check(f"Table II {tag} delta (%)", want,
                      round(100 * (r - a3) / a3, 1), tol=0.05, fmt="{:+.1f}%")

    print("\n=== Sec. III-D: seed-noise floor for the seed-42 ablation ===")
    # The upper group of Table II is seed-42 only; the paper argues only the
    # product term clears seed noise. That argument needs the noise floor,
    # which is the per-seed std of the *five-seed* variants of this same
    # protocol. An earlier draft said 1.6--2.2%, which no variant reaches at
    # either end: the family spans 1.67% (F-M2) to 2.14% (F-A7).
    pcts = {}
    for tag in ("FM1", "FM2", "FA3", "FA7", "FA8", "FA10"):
        per = [statistics.mean([ab(tag, s, ds) for ds in DATASETS]) for s in SEEDS]
        pcts[tag] = 100 * statistics.stdev(per) / statistics.mean(per)
    per_single = [statistics.mean([summary["single_swa_per_seed"][str(s)][ds]["rmse"]
                                   for ds in DATASETS]) for s in SEEDS]
    pcts["single"] = 100 * statistics.stdev(per_single) / statistics.mean(per_single)
    lo, hi = min(pcts.values()), max(pcts.values())
    print("  per-variant per-seed std: "
          + "  ".join(f"{k}={v:.2f}%" for k, v in sorted(pcts.items())))
    check("seed-noise floor low", 1.7, round(lo, 1), tol=5e-3, fmt="{:.1f}%")
    check("seed-noise floor high", 2.1, round(hi, 1), tol=5e-3, fmt="{:.1f}%")

    # The lower group's Delta column, relative to F-A3, and the "span only
    # 0.4%" that the convergence-neutrality sentence turns on.
    fa3_mean = mean_std("FA3")[0]
    fam = {}
    for tag, want in (("FA7", 0.4), ("FA8", 0.4), ("FA10", 0.2)):
        fam[tag] = mean_std(tag)[0]
        check(f"Table II {tag} delta (%)", want,
              round(100 * (fam[tag] - fa3_mean) / fa3_mean, 1), tol=0.05,
              fmt="{:+.1f}%")
    fam["FA3"] = fa3_mean
    check("Table II F-A* convergence span (%)", 0.4,
          round(100 * (max(fam.values()) - min(fam.values())) / min(fam.values()), 1),
          tol=0.05, fmt="{:.1f}%")

    print("\n=== effect of gating, matched architectures ===")
    fa3, fa11 = mean_std("FA3")[0], mean_std("FA11")[0] if all(
        ab("FA11", s, ds) for s in SEEDS for ds in DATASETS) else None
    print(f"  F-A3={fa3:.4f}  F-A11={'pending' if fa11 is None else f'{fa11:.4f}'}")
    if fa11:
        d = [100 * (ab("FA11", s, ds) - ab("FA3", s, ds)) / ab("FA3", s, ds)
             for s in SEEDS for ds in DATASETS]
        print(f"  retrained no-gate delta: median={statistics.median(d):+.2f}%  "
              f"mean={statistics.mean(d):+.2f}%  worse in {sum(1 for v in d if v > 0)}/{len(d)}")
        # Sec. III-D quotes this pair as the *final-fit* counterpart of the
        # matched-state arm above ("1.4% behind F-A3 (1.1% median, worse in 11
        # of 15)"). It used to be printed but never checked, which left it the
        # only paper literal with a dedicated script and no audit entry -- so
        # the script's removal would have made the sentence unbacked. Checked
        # here instead, the number survives the script.
        check("F-A11 final-fit delta, median (%)", 1.1,
              round(statistics.median(d), 1), tol=0.05, fmt="{:+.1f}%")
        check("F-A11 final-fit pairs worse than F-A3", 11,
              sum(1 for v in d if v > 0), tol=0, fmt="{:.0f}")

    print("\n=== Sec. III-A: dataset sizes and label scales ===")
    # The sd claim ("differ in mean but not in spread") is the justification
    # for comparing per-dataset RMSE at all, so it is worth re-deriving.
    import csv as _csv
    DATA = OUT.parent.parent / "data"
    claimed_ds = {  # dataset: (drugs, targets, pairs, sd as printed)
        "davis": (68, 442, 30056, 0.89),
        "kiba": (2111, 229, 118254, 0.84),
        "metz": (1423, 170, 35259, 0.96),
    }
    for ds, (nd, nt, npairs, sd) in claimed_ds.items():
        rows = list(_csv.DictReader((DATA / ds / "affinity.csv")
                                    .open(encoding="utf-8")))
        vals = [float(list(r.values())[-1]) for r in rows]
        check(f"{ds} drugs", nd, len({list(r.values())[0] for r in rows}),
              fmt="{:.0f}")
        check(f"{ds} targets", nt, len({list(r.values())[1] for r in rows}),
              fmt="{:.0f}")
        check(f"{ds} pairs", npairs, len(rows), fmt="{:.0f}")
        check(f"{ds} label sd", sd, round(statistics.pstdev(vals), 2),
              tol=5e-3, fmt="{:.2f}")

    print("\n=== Sec. III-F: ensemble spread ranks error but is uncalibrated ===")
    # Re-derived from the raw prediction CSVs rather than from
    # review_analysis.json, so the check survives that file being
    # regenerated. The paper's Q4/Q1 ratios were 8.9/11.3/2.0, which no
    # artifact or recomputation reproduces: the same sentence's r and
    # coverage match this computation exactly, so the ratios were the odd
    # ones out. Correct values are 4.4/6.3/1.6.
    # `np` is imported at module scope and used by a nested helper earlier
    # in main(); a second local import here would make every `np` in main()
    # a local cell and break that helper with a NameError.
    import csv as _csv2
    MEMBERS = MEMBERS_ENS

    def _preds(path):
        yt, yp = [], []
        with path.open(encoding="utf-8") as f:
            for row in _csv2.DictReader(f):
                yt.append(float(row["y_true"]))
                yp.append(float(row["y_pred"]))
        return np.asarray(yt), np.asarray(yp)

    # The paper quotes five-seed ranges, not seed-42 points. It used to quote
    # the points, and a reviewer's obvious next question -- "is the coverage
    # 0.69 at other seeds, or was that read off one lucky run?" -- is what the
    # ranges answer. They are worth having here because two of the three
    # quantities are *not* stable: Davis's Q4/Q1 runs 2.4 to 4.4, so the
    # single-seed figure was its maximum. The r and coverage columns barely
    # move (0.43-0.47, 0.66-0.72), which is the honest picture: the spread
    # ranks error consistently and is consistently uncalibrated.
    # The values here were the paper's earlier seed-42 points: 0.46, 4.4,
    # 0.69 for Davis, and 8.9/11.3/2.0 for the ratios before those were found
    # to be reproduced by nothing.
    claimed_spread = {          # ds: ((r_lo, r_hi), (q_lo, q_hi), (c_lo, c_hi))
        "davis": ((0.43, 0.47), (2.4, 4.4), (0.66, 0.72)),
        "kiba": ((0.53, 0.56), (4.9, 6.7), (0.67, 0.70)),
        "metz": ((0.24, 0.28), (1.5, 1.6), (0.46, 0.47)),
    }
    for ds, ((r_lo, r_hi), (q_lo, q_hi), (c_lo, c_hi)) in claimed_spread.items():
        rs, qs, cs = [], [], []
        for seed in SEEDS:
            single = OUT / f"rail_single_seed{seed}_200/{ds}/test_predictions.csv"
            members = [OUT / f"rail_{m}_seed{seed}_200/{ds}/test_predictions.csv"
                       for m in MEMBERS]
            if not single.exists() or not all(p.exists() for p in members):
                continue
            y, _ = _preds(single)
            stack = np.stack([_preds(p)[1] for p in members])
            mu, sig = stack.mean(0), stack.std(0)
            q1, q4 = sig <= np.quantile(sig, 0.25), sig >= np.quantile(sig, 0.75)
            r1 = float(np.sqrt(np.mean((y[q1] - mu[q1]) ** 2)))
            r4 = float(np.sqrt(np.mean((y[q4] - mu[q4]) ** 2)))
            rs.append(round(float(np.corrcoef(sig, np.abs(y - mu))[0, 1]), 2))
            qs.append(round(r4 / r1, 1))
            cs.append(round(float(np.mean(np.abs(y - mu) <= 1.645 * sig)), 2))
        print(f"       {ds}: n_seeds={len(rs)} r={sorted(rs)} "
              f"Q4/Q1={sorted(qs)} cov={sorted(cs)}")
        for label, lo, hi, vals in (("r", r_lo, r_hi, rs),
                                    ("Q4/Q1", q_lo, q_hi, qs),
                                    ("coverage", c_lo, c_hi, cs)):
            if not vals:
                check(f"{ds} {label} low", lo, None)
                continue
            check(f"{ds} {label} low", lo, min(vals), tol=5e-3, fmt="{:.2f}")
            check(f"{ds} {label} high", hi, max(vals), tol=5e-3, fmt="{:.2f}")

    print("\n=== ratio sensitivity (the ~24x headline) ===")

    # The headline is a ratio of two medians, so a trim applied only to the
    # numerator is not the ratio a reader recomputes after applying the
    # paper's own exclusion rule. analysis_ratio_sensitivity.py applies every
    # trim to BOTH arms; these claims are pinned against its output rather
    # than against the numerator alone.
    rs = json.loads((OUT / "ratio_sensitivity.json").read_text(encoding="utf-8"))
    pooled = {k: v["ratio_of_medians"] for k, v in rs["trims"].items()}
    mor = {k: v["median_of_ratios"] for k, v in rs["trims"].items()}
    check("headline 24x (untrimmed, median ratio)", 24,
          round(pooled["none"], 0), tol=0.5, fmt="{:.0f}")
    check("trimmed on both arms (paper 19.7x)", 19.7,
          round(pooled["gt50"], 1), tol=0.05, fmt="{:.1f}")
    check("least favourable estimator low (12.5x)", 12.5,
          round(mor["gt50"], 1), tol=0.05, fmt="{:.1f}")
    check("least favourable estimator high (16.5x)", 16.5,
          round(mor["none"], 1), tol=0.05, fmt="{:.1f}")
    # "never below 9.7x" is a claim about the floor over EVERY estimator,
    # trim and dataset -- its weakest cell is a per-dataset median-of-ratios,
    # well below the pooled one. Pinning it against the pooled estimator
    # alone would pass while the sentence overstated.
    gmin = rs["global_min"]
    check("never below 9.7x (floor, all variants)", 9.7,
          round(gmin["value"], 1), tol=0.05, fmt="{:.1f}")
    print(f"        weakest cell: {gmin['key']}/{gmin['estimator']} "
          f"over {gmin['n_variants']} variants")

    print("\n=== cold-start paragraph ===")

    # The sentence quotes a range from *this* tree's cold runs, and it has
    # three ways to go silently wrong, so all three are checked rather than
    # the range alone:
    #   * the partition key -- an id-keyed run is not a cold split at all,
    #     which is exactly how the superseded `autoresearch/outputs/icasp_*
    #     cold*` numbers came to be quoted as if they were one;
    #   * the arm -- the sentence names F-A3, so the run behind it must be
    #     fixed interactions with both gates on;
    #   * the range, and the ~0.44 it rises from, which must come from that
    #     same arm so the before/after pair is one architecture.
    cold_means: dict[str, float] = {}
    for split in ("cold_drug", "cold_target"):
        base = OUT / f"rail_cold_FA3_{split}_seed42_200"
        cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
        check(f"cold {split}: keyed on entity, not id", "entity",
              cfg.get("cold_key"), fmt="{}")
        check(f"cold {split}: the F-A3 arm the sentence names",
              ("fixed", True, True),
              (cfg.get("interaction"), cfg.get("use_drug_gate"),
               cfg.get("use_context_gate")), fmt="{}")
        cold_means[split] = statistics.mean(
            [json.loads((base / ds / "test_metrics.json").read_text(
                encoding="utf-8"))["rmse"] for ds in DATASETS])

    check("cold-start range low (paper 0.62)", 0.62,
          round(min(cold_means.values()), 2), tol=5e-3, fmt="{:.2f}")
    check("cold-start range high (paper 0.66)", 0.66,
          round(max(cold_means.values()), 2), tol=5e-3, fmt="{:.2f}")
    check("cold-start baseline (~0.44), same arm, random split", 0.44,
          round(mean_std("FA3")[0], 2), tol=5e-3, fmt="{:.2f}")

    print("\n=== Table II lower group + architecture claims ===")

    # These live in the *root* outputs/ tree (rail_dta/../outputs), not
    # rail_dta/outputs. Seed 42 uses two naming generations depending on the
    # model; see audit_baseline_rows.py, which resolves both and prints which
    # directory served each seed.
    import subprocess
    for script in ("audit_baseline_rows.py", "audit_architecture.py"):
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent
                                            / script)], check=False)

    # For audit_coverage.py: which values this audit pinned. Written even
    # when some checks failed -- a coverage report read alongside a failing
    # audit is still the right pair of facts.
    (OUT / "audit_pinned_paper_numbers.json").write_text(
        json.dumps(sorted(set(PINNED))), encoding="utf-8")

    print(f"\n=== {len(OK)} checks ok, {len(BAD)} mismatched ===")
    for label, shown in BAD:
        print(f"  MISMATCH {label}: {shown}")


if __name__ == "__main__":
    main()
