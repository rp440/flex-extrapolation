# Reproducing FLEX

Three levels, from cheapest to most expensive.

## 1. Recompute every number in the README (no GPU, about 30 seconds)

`data/per_problem.csv` has one row per problem per run (5,000 rows): finish reason,
prompt and completion tokens, the original grader's verdict, and the corrected verdict.

```bash
pip install numpy gguf pytest requests sympy math-verify
python code/paired_stats.py --bootstrap          # FLEX vs Qwen3.6, 8,000-token budget
python code/paired_stats.py --all                # every control
python code/paired_stats.py --budget 10000       # budget sweep
python code/paired_stats.py --budget 0 --all     # no token limit
python -m pytest tests -q                        # build, mask, and published-number tests
```

Expected first line of the first command: `flex_ffn_1p25  294  369  98  23  3.2e-12  52->22`.

## 2. Rebuild the FLEX checkpoint (needs the two source GGUFs, about 37 GB each)

Inputs: Q8_0 GGUF files of Qwen3.5-35B-A3B (older) and Qwen3.6-35B-A3B (newer).

```bash
python code/build_flex.py --old Qwen3.5-35B-A3B-Q8_0.gguf --new Qwen3.6-35B-A3B-Q8_0.gguf \
    --out qwen36_mlp_1p25.Q8_0.gguf --alpha 1.0 --alpha-mlp 1.25 --workers 12
```

The other runs in `data/per_problem.csv`:

| Run name in CSV | Build flags | Server slots |
|---|---|---|
| `qwen36` | none (the original Qwen3.6 GGUF) | 9 |
| `flex_ffn_1p25` | `--alpha 1.0 --alpha-mlp 1.25` | 9 |
| `random_same_norm` | `--alpha 1.0 --alpha-mlp 1.25 --control-mode random --control-seed 125` | 9 |
| `shuffled_same_norm` | `--alpha 1.0 --alpha-mlp 1.25 --control-mode shuffle --control-seed 125` | 9 |
| `ffn_0p75` | `--alpha 1.0 --alpha-mlp 0.75` | 12 |
| `full_1p25` | `--alpha 1.25` | 12 |
| `embed_1p25_run1`, `embed_1p25_run2` | `--alpha 1.0 --alpha-embed 1.25` (two runs) | 12, 24 |
| `b7_noearly_mlp` | a Terminal-Bench oriented layer-specific recipe (flags not included here) | 12 |

What was checked about `build_flex.py`: on small test GGUF files containing routed
experts, shared experts, a router, attention, norm and embedding tensors, its output is
byte for byte identical to the research script that built the original checkpoints, for the
FLEX, random-control, shuffled-control, whole-model, alpha 0.75, and embedding builds.
`tests/test_flex.py` also checks the mask, the formula, the norm of the controls, and that
the attention option changes the attention tensors.

## 3. Regenerate the MATH-500 answers (needs a GPU box with llama.cpp)

Serve each checkpoint with `llama-server` and run the evaluator against it:

```bash
llama-server -m MODEL.gguf --port 8090 -c $((32768 * SLOTS)) -ngl 99 \
    --parallel SLOTS --cont-batching --no-webui
python code/eval_math500.py --port 8090 --n 500 --seed 0 --parallel SLOTS \
    --max-tokens 32768 --out results.json
```

Settings used for every run: greedy decoding (temperature 0), thinking enabled (the model's
default), the HuggingFaceH4/MATH-500 test split shuffled with seed 0 (so row i is the same
problem in every run), and a limit of 32,768 tokens per sample counting the prompt. The
prompt is "Solve the following math problem step by step. Put your final answer in
\boxed{} on the last line." followed by the problem.

Then grade and flatten:

```bash
python code/grade.py results.json                       # original vs corrected pass count
python code/export_per_problem.py --out data/per_problem.csv \
    qwen36=results_base.json flex_ffn_1p25=results_flex.json ...
```

With greedy decoding, the number of server slots can change individual generations. The two
embedding runs (12 and 24 slots) differ by 19 problems at the 8,000-token budget (314 and
295 solved), which is a useful size for run-to-run spread when reading the controls table.
