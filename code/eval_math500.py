"""MATH-500 full runner against an existing llama-server (OpenAI-compatible).

Does NOT start its own server - pass --port to hit one already running.
Scores with math_verify (LaTeX/symbolic/numeric equivalence).

Usage:
    python3 eval_math500_full.py --port 8090 --n 500 --parallel 4 \
        --out /workspace/losslessTTT/results/mathfull_TAG.json

--n 500 runs the full split. --n 200 for a quick subset.
"""
from __future__ import annotations
import argparse, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import requests


BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
NUM_RE = re.compile(r"-?\d+(?:[\.,]\d+)?")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

MATH_INSTR = ("Solve the following math problem step by step. "
              "Put your final answer in \\boxed{} on the last line.")


def extract_math(text: str) -> str | None:
    text = THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    ms = BOXED_RE.findall(text)
    if ms:
        return ms[-1].strip()
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def numeric_eq(a, b) -> bool:
    if a is None or b is None: return False
    a0, b0 = str(a).strip(), str(b).strip()
    if a0 == "" or b0 == "": return False
    if a0 == b0: return True
    def _latex_norm(s):
        return (s.replace(" ", "").replace("\n", "").replace("\t", "")
                 .replace("\\left", "").replace("\\right", "")
                 .replace("\\,", "").replace("\\!", ""))
    if _latex_norm(a0) == _latex_norm(b0): return True
    def _dedollar(s):
        return s[1:-1].strip() if s.startswith("$") and s.endswith("$") else s
    a1, b1 = _dedollar(a0), _dedollar(b0)
    m = re.match(r"^\s*[A-Za-z]\s*=\s*(.+)$", a1); a2 = m.group(1).strip() if m else a1
    m = re.match(r"^\s*[A-Za-z]\s*=\s*(.+)$", b1); b2 = m.group(1).strip() if m else b1
    if a2 == b2: return True
    a_interval, b_interval = _interval_norm(a2), _interval_norm(b2)
    if a_interval and b_interval and a_interval == b_interval:
        return True
    if _symbolic_eq(a2, b2):
        return True
    try:
        def _f(s):
            return float(s.replace(",", "").replace("\\!", "").replace("\\,", "")
                          .replace("\\$", "").replace("$", "").strip())
        if abs(_f(a2) - _f(b2)) < 1e-6: return True
    except Exception:
        pass
    try:
        from math_verify import parse, verify
        wa = a0 if "\\boxed" in a0 else f"\\boxed{{{a0}}}"
        wb = b0 if "\\boxed" in b0 else f"\\boxed{{{b0}}}"
        ga, gb = parse(wa), parse(wb)
        if ga and gb and verify(ga, gb): return True
    except Exception:
        pass
    return False


def _interval_norm(s: str) -> str | None:
    s = (s or "").strip()
    compact = (s.replace(" ", "").replace("\\left", "").replace("\\right", "")
                 .replace("\\lambda", "x").replace("lambda", "x"))
    m = re.fullmatch(r"([\[\(])([^,]+),([^\]\)]+)([\]\)])", compact)
    if m:
        left, lo, hi, right = m.groups()
        return f"{left}{_scalar_norm(lo)},{_scalar_norm(hi)}{right}"
    m = re.fullmatch(r"([^<>=]+)(<=|<|\\leq?|\\lt)x(<=|<|\\leq?|\\lt)([^<>=]+)", compact)
    if m:
        lo, op1, op2, hi = m.groups()
        left = "[" if op1 in ("<=", "\\le", "\\leq") else "("
        right = "]" if op2 in ("<=", "\\le", "\\leq") else ")"
        return f"{left}{_scalar_norm(lo)},{_scalar_norm(hi)}{right}"
    m = re.fullmatch(r"([^<>=]+)(>=|>|\\geq?|\\gt)x(>=|>|\\geq?|\\gt)([^<>=]+)", compact)
    if m:
        hi, op1, op2, lo = m.groups()
        left = "[" if op2 in (">=", "\\ge", "\\geq") else "("
        right = "]" if op1 in (">=", "\\ge", "\\geq") else ")"
        return f"{left}{_scalar_norm(lo)},{_scalar_norm(hi)}{right}"
    return None


def _scalar_norm(s: str) -> str:
    s = (s or "").strip()
    try:
        return str(Fraction(s))
    except Exception:
        return s


def _symbolic_eq(a: str, b: str) -> bool:
    try:
        import sympy as sp
    except Exception:
        return False

    def convert(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
        s = s.replace("\\,", "").replace("\\!", "").replace(" ", "")
        for _ in range(8):
            ns = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
            ns = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", ns)
            if ns == s:
                break
            s = ns
        s = re.sub(r"\\sqrt([A-Za-z0-9]+)", r"sqrt(\1)", s)
        s = s.replace("\\pi", "pi")
        s = s.replace("\\cdot", "*").replace("\\times", "*")
        s = re.sub(r"(?<=\d)(?=sqrt|pi|[A-Za-z(])", "*", s)
        s = re.sub(r"(?<=\))(?=\d|sqrt|pi|[A-Za-z(])", "*", s)
        return s

    try:
        aa = sp.sympify(convert(a), locals={"sqrt": sp.sqrt, "pi": sp.pi})
        bb = sp.sympify(convert(b), locals={"sqrt": sp.sqrt, "pi": sp.pi})
        return bool(sp.simplify(aa - bb) == 0)
    except Exception:
        return False


def load_math500(n: int, seed: int = 0, level_filter: str | None = None):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = list(ds)
    if level_filter:
        allowed = {
            x.strip() for x in str(level_filter).split(",") if x.strip()
        }
        allowed |= {f"Level {x}" for x in allowed}
        rows = [r for r in rows if str(r.get("level")) in allowed]
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    items = []
    for r in rows[:n]:
        gold = r.get("answer")
        if gold is None:
            m = BOXED_RE.search(r.get("solution", ""))
            gold = m.group(1).strip() if m else ""
        items.append({"q": r["problem"], "gold": str(gold).strip(),
                      "level": r.get("level"), "subject": r.get("subject")})
    return items


def chat_complete(port: int, model: str, user: str, max_tokens: int,
                  temperature: float = 0.0, top_p: float | None = None,
                  seed: int | None = None, timeout: int = 1800,
                  enable_thinking: bool = True) -> dict:
    body = {"model": model,
            "messages": [{"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens,
            "stream": False}
    if top_p is not None:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                      json=body, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    text = "\n".join(x for x in (reasoning, content) if x).strip()
    return {
        "text": text,
        "content": content,
        "reasoning_content": reasoning,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
        "model": data.get("model"),
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def summarize(items: list[dict], results: list[dict | None], elapsed_s: float) -> dict:
    completed = sum(1 for r in results if r is not None)
    n_pass = sum(1 for r in results if r and r.get("pass"))
    by_level = {}
    for r in results:
        if not r:
            continue
        lv = str(r.get("level", "?"))
        by_level.setdefault(lv, {"n": 0, "p": 0})
        by_level[lv]["n"] += 1
        by_level[lv]["p"] += int(bool(r.get("pass")))
    for x in by_level.values():
        x["pass_at_1"] = x["p"] / x["n"] if x["n"] else 0
    return {
        "n": len(items),
        "completed": completed,
        "n_pass": n_pass,
        "pass_at_1": n_pass / len(items) if items else 0,
        "by_level": by_level,
        "elapsed_s": elapsed_s,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--model", default="local")
    ap.add_argument("--level", default=None,
                    help="filter to a single difficulty (e.g. '5' for L5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = load_math500(args.n, args.seed, args.level)
    print(f"[m500] {len(items)} items  level={args.level or 'all'}", flush=True)

    out_path = Path(args.out)
    partial_path = out_path.with_name(out_path.name + ".partial")
    results = [None] * len(items)
    if partial_path.exists():
        try:
            prior = json.loads(partial_path.read_text())
            prior_results = prior.get("results") or []
            if len(prior_results) == len(items):
                results = prior_results
                completed = sum(1 for r in results if r is not None)
                print(f"[m500] resumed partial rows: {completed}/{len(items)} "
                      f"from {partial_path}", flush=True)
        except Exception as e:
            print(f"[m500] ignoring unreadable partial {partial_path}: {e}", flush=True)
    t0 = time.time()

    def worker(i, it):
        try:
            resp = chat_complete(args.port, args.model,
                                 f"{MATH_INSTR}\n\n{it['q']}",
                                 max_tokens=args.max_tokens)
            gen = resp["text"]
            pred = extract_math(gen)
            ok = numeric_eq(pred, it["gold"])
            return i, {"gold": it["gold"], "pred": pred, "pass": ok,
                       "level": it["level"], "subject": it["subject"],
                       "gen_chars": len(gen),
                       "content_chars": resp["content_chars"],
                       "reasoning_chars": resp["reasoning_chars"],
                       "finish_reason": resp["finish_reason"],
                       "usage": resp["usage"],
                       "served_model": resp["model"],
                       "content": resp["content"],
                       "reasoning_content": resp["reasoning_content"],
                       "gen": gen,
                       "err": ""}
        except Exception as e:
            return i, {"gold": it["gold"], "pred": None, "pass": False,
                       "level": it["level"], "subject": it["subject"],
                       "gen_chars": 0, "err": f"{type(e).__name__}: {e}"[:300]}

    pending = [(i, it) for i, it in enumerate(items) if results[i] is None]
    done = len(items) - len(pending)
    if done:
        write_json_atomic(partial_path, summarize(items, results, time.time() - t0))

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(worker, i, it) for i, it in pending]
        for fut in as_completed(futs):
            i, row = fut.result()
            results[i] = row
            done += 1
            write_json_atomic(partial_path, summarize(items, results, time.time() - t0))
            if done % max(1, len(items) // 20) == 0 or done == len(items):
                print(f"  [{done}/{len(items)}] "
                      f"pass so far: {sum(1 for r in results if r and r.get('pass'))} "
                      f"elapsed: {time.time()-t0:.0f}s", flush=True)

    dt = time.time() - t0
    summary = summarize(items, results, dt)
    n_pass = summary["n_pass"]
    write_json_atomic(out_path, summary)
    print(f"\n[m500] pass@1 = {summary['pass_at_1']:.3f}  "
          f"({n_pass}/{len(items)})  elapsed={dt:.0f}s", flush=True)
    for lv, x in sorted(summary["by_level"].items()):
        print(f"  L{lv}: {x['p']}/{x['n']} = {x['pass_at_1']:.3f}")
    print(f"[m500] saved -> {out_path}")


if __name__ == "__main__":
    main()
