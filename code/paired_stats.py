"""Paired statistics for the FLEX results, computed from data/per_problem.csv.

Problems are paired by index (the eval shuffles MATH-500 with a fixed seed, so
row i is the same problem in every run; gold answers are cross-checked).

Reports, for each run against the baseline:
  - problems solved within a token budget (correct and completion <= budget)
  - problems only the run solves / only the baseline solves, exact McNemar p
  - runs that hit the generation limit (finish_reason != stop)
  - mean completion tokens with a paired bootstrap 95% CI

Usage:
    python paired_stats.py                      # FLEX vs baseline, 8000 tokens
    python paired_stats.py --budget 10000 --all
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "per_problem.csv"


def load(path):
    runs = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            runs[r["run"]][int(r["idx"])] = {
                "gold": r["gold"], "stop": r["finish_reason"] == "stop",
                "tokens": int(r["completion_tokens"]),
                "pass": bool(int(r["pass_corrected"])),
                "pass_original": bool(int(r["pass_original"])),
            }
    return {k: [v[i] for i in range(500)] for k, v in runs.items()}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: 2 * P(Binomial(b + c, 0.5) <= min(b, c))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def bootstrap_ci(diffs, n_boot=20000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def solved(row, budget):
    return row["pass"] and (budget is None or row["tokens"] <= budget)


def compare(base, cand, budget):
    assert all(a["gold"].strip() == b["gold"].strip() for a, b in zip(base, cand)), "gold mismatch"
    only_cand = sum(1 for a, b in zip(base, cand) if solved(b, budget) and not solved(a, budget))
    only_base = sum(1 for a, b in zip(base, cand) if solved(a, budget) and not solved(b, budget))
    return {
        "solved_base": sum(solved(a, budget) for a in base),
        "solved_cand": sum(solved(b, budget) for b in cand),
        "only_cand": only_cand, "only_base": only_base,
        "p": mcnemar_exact(only_base, only_cand),
        "capped_base": sum(not a["stop"] for a in base),
        "capped_cand": sum(not b["stop"] for b in cand),
        "mean_tokens_base": sum(a["tokens"] for a in base) / 500,
        "mean_tokens_cand": sum(b["tokens"] for b in cand) / 500,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--baseline", default="qwen36")
    ap.add_argument("--budget", type=int, default=8000,
                    help="completion-token budget; 0 means no limit")
    ap.add_argument("--all", action="store_true", help="compare every run, not just FLEX")
    ap.add_argument("--bootstrap", action="store_true", help="also print token-mean CI")
    args = ap.parse_args()
    budget = args.budget or None
    runs = load(args.csv)
    base = runs[args.baseline]
    names = [n for n in runs if n != args.baseline and (args.all or n == "flex_ffn_1p25")]
    label = "no limit" if budget is None else f"<= {budget} tokens"
    print(f"solved ({label}); baseline = {args.baseline}\n")
    print(f"{'run':<22}{'base':>6}{'run':>6}{'only run':>10}{'only base':>11}{'p':>10}"
          f"{'capped':>14}")
    for name in names:
        c = compare(base, runs[name], budget)
        print(f"{name:<22}{c['solved_base']:>6}{c['solved_cand']:>6}{c['only_cand']:>10}"
              f"{c['only_base']:>11}{c['p']:>10.2g}"
              f"{str(c['capped_base']) + '->' + str(c['capped_cand']):>14}")
        if args.bootstrap:
            diffs = [b["tokens"] - a["tokens"] for a, b in zip(base, runs[name])]
            lo, hi = bootstrap_ci(diffs)
            print(f"{'':<22}mean tokens {c['mean_tokens_base']:.0f} -> "
                  f"{c['mean_tokens_cand']:.0f}, paired bootstrap 95% CI [{lo:.0f}, {hi:.0f}]")


if __name__ == "__main__":
    main()
