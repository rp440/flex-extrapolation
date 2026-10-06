"""Flatten raw eval_math500.py result files into one per-problem CSV.

Each row is one problem in one run. Generations are dropped (they are large);
everything the README's statistics need is kept: finish reason, token counts, the
original grader's verdict, and the corrected verdict from grade.py.

Usage:
    python export_per_problem.py --out ../data/per_problem.csv \
        qwen36=results/math500_full_greedy_qwen36_alpha1_p9.json \
        flex_ffn_1p25=results/math500_full_greedy_qwen36_mlp_1p25_p9.json ...
"""
from __future__ import annotations

import argparse
import csv
import json

from grade import regrade_row

FIELDS = ["run", "idx", "level", "subject", "gold", "finish_reason",
          "prompt_tokens", "completion_tokens", "pass_original", "pass_corrected"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+", help="NAME=path/to/results.json")
    args = ap.parse_args()
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for spec in args.runs:
            name, path = spec.split("=", 1)
            rows = json.load(open(path))["results"]
            assert len(rows) == 500 and all(r is not None for r in rows), path
            for i, r in enumerate(rows):
                usage = r.get("usage") or {}
                writer.writerow({
                    "run": name, "idx": i, "level": r["level"], "subject": r["subject"],
                    "gold": r["gold"], "finish_reason": r["finish_reason"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "pass_original": int(bool(r["pass"])),
                    "pass_corrected": int(regrade_row(r)["pass2"]),
                })
            print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
