"""Corrected grader for MATH-500 runs produced by eval_math500.py.

The original grader in eval_math500.py had false negatives: its boxed-answer
regex fails on doubly nested braces (\\boxed{\\frac{\\sqrt{3}}{3}}), and equivalent
forms such as \\frac65 vs \\frac{6}{5}, 30 vs 30^\\circ, or \\text{Evelyn} vs Evelyn
were marked wrong when math_verify was not importable on the eval host.

This module re-grades every row with
  1. the original numeric_eq (with math_verify importable),
  2. a Hendrycks-style string normalization (degrees, \\text, \\dfrac, \\frac65,
     base subscripts, ordinals),
  3. a balanced-brace \\boxed{...} extractor.
Only equivalence false negatives are fixed. Extraction of the final answer from
truncated traces is unchanged.

Usage:
    python grade.py results.json            # prints original vs corrected pass
"""
from __future__ import annotations

import json
import re
import sys

import eval_math500 as ev


def boxed_balanced(text: str) -> str | None:
    """Last \\boxed{...} with balanced braces (any nesting depth)."""
    text = ev.THINK_RE.sub("", text or "")
    if "<think>" in text:
        text = text.split("<think>")[0]
    last = None
    for m in re.finditer(r"\\boxed\s*\{", text):
        i = m.end()
        depth, j = 1, i
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[i:j - 1].strip()
    return last


def strip_string(s: str) -> str:
    """Hendrycks-style MATH normalization (the subset that mattered here)."""
    s = str(s or "").strip()
    s = s.replace("\n", "").replace("\\!", "").replace("\\,", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)
    for _ in range(4):
        s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)        # \frac65 -> \frac{6}{5}
    s = re.sub(r"\\frac\s*(\d)\s*\{", r"\\frac{\1}{", s)             # \frac9{19}
    s = re.sub(r"\\frac\{([^{}]+)\}\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"_\{(\d+)\}", r"_\1", s)                             # 4343_{6} -> 4343_6
    m = re.fullmatch(r"(\d+)\^\{?th\}?\s*(grade)?", s.replace(" ", ""))
    if m:
        s = m.group(1)
    s = s.replace(" ", "")
    m = re.match(r"^[A-Za-z]\\?=(.+)$", s)                           # x=5 -> 5
    if m:
        s = m.group(1)
    return s


def equiv(pred, gold) -> bool:
    if pred is None or str(pred).strip() == "":
        return False
    if ev.numeric_eq(pred, gold):
        return True
    a, b = strip_string(pred), strip_string(gold)
    if a and a == b:
        return True
    if ev.numeric_eq(a, b):
        return True
    try:
        from math_verify import parse, verify
        wa = f"\\boxed{{{a}}}" if "\\boxed" not in a else a
        wb = f"\\boxed{{{b}}}" if "\\boxed" not in b else b
        ga, gb = parse(wa), parse(wb)
        if ga and gb and verify(ga, gb):
            return True
    except Exception:
        pass
    return False


def regrade_row(row: dict) -> dict:
    gold = str(row["gold"]).strip()
    candidates = [row.get("pred")]
    balanced = boxed_balanced(row.get("gen") or row.get("content") or "")
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    ok = any(equiv(c, gold) for c in candidates if c is not None)
    return {"pass2": bool(ok), "pred2": next((c for c in candidates if c), None)}


def main() -> None:
    rows = json.load(open(sys.argv[1]))["results"]
    graded = [regrade_row(r) for r in rows]
    orig = sum(bool(r["pass"]) for r in rows)
    fixed = sum(g["pass2"] for g in graded)
    down = sum(1 for r, g in zip(rows, graded) if r["pass"] and not g["pass2"])
    print(f"original {orig}/{len(rows)}  corrected {fixed}/{len(rows)}  "
          f"(rows that went from pass to fail: {down})")


if __name__ == "__main__":
    main()
