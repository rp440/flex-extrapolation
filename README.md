# FLEX: Family-Localized Weight-Space Extrapolation

Qwen3.6 was post-trained from Qwen3.5. This repo asks whether the difference between those two checkpoints can be pushed a little further, with no training, and where in the weights that difference matters.

FLEX takes Qwen3.6 as is and moves only its FFN tensors 25% further along the Qwen3.5 to Qwen3.6 direction. On MATH-500 under an 8,000-token budget, the model solves 369/500 problems instead of 294/500 (58.8% to 73.8%).

Independent research by Ruchit Patel.

## Result

Full MATH-500, 500 problems, greedy decoding, Qwen3.6-35B-A3B (Q8_0 GGUF), 32,768-token limit per sample (prompt included).

| Measure | Qwen3.6 | FLEX (FFN, alpha = 1.25) | Change | Paired test |
|---|---|---|---|---|
| Solved within 8,000 tokens | 294 (58.8%) | 369 (73.8%) | +75 | exact McNemar p = 3.2e-12 |
| Runs that hit the token cap | 52 | 22 | -30 | p = 1.8e-4 |
| Mean completion tokens | 10,411 | 7,402 | -29% | bootstrap 95% CI [-3,875, -2,173] |
| Solved with no token limit | 475 (95.0%) | 483 (96.6%) | +8 | p = 0.22 |

How to read it: FLEX's gain comes from long derivations that now finish. Of the 20 problems FLEX solves that Qwen3.6 does not, all 20 are cases where Qwen3.6 ran into the 32,768-token limit and FLEX finished the same problem correctly. There is no problem where both models finish and only FLEX gets it right. The advantage grows as the budget shrinks:

| Budget | Problems only FLEX solves | Problems only Qwen3.6 solves | FLEX share of the differences |
|---|---|---|---|
| no limit | 20 | 12 | 62.5% |
| 12,000 tokens | 68 | 23 | 74.7% |
| 10,000 tokens | 80 | 25 | 76.2% |
| 8,000 tokens | 98 | 23 | 81.0% |

## Controls

Every control below goes through the same build pipeline and is compared against the same Qwen3.6 baseline, on the same 8,000-token metric.

| Variant | What it tests | Solved within 8,000 tokens | Paired p vs Qwen3.6 |
|---|---|---|---|
| Qwen3.6 | baseline | 294 | n/a |
| FLEX, real FFN direction, alpha = 1.25 | the method | 369 | 3.2e-12 |
| Random direction, same norm per tensor | is it just perturbation size? | 282 | 0.23 |
| Shuffled delta, same norm per tensor | is it just the delta's value distribution? | 302 | 0.48 |
| Real FFN direction, alpha = 0.75 | same size, opposite sign | 289 | 0.70 |
| Whole model, alpha = 1.25 | extrapolate every tensor | 397 | below 1e-15 |

The random, shuffled, and sign-flipped versions of the FFN update leave Qwen3.6 where it was, so the effect is tied to the actual direction of the update and not to how big the nudge is. Extrapolating the whole model gives a larger 8,000-token gain (397) than FFN-only (369). FFN-only keeps uncapped accuracy at 483, against 477 for the whole model.

## What "FFN family" means here

The mask is every GGUF tensor whose name contains `ffn_`. In this MoE model that is:

- the routed expert weights (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`)
- the shared expert weights (`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`)
- the router (`ffn_gate_inp`) and the shared-expert gate (`ffn_gate_inp_shexp`)

Attention, SSM, norm, and embedding tensors stay at the Qwen3.6 values. Splitting the mask into experts, shared experts, and router is open future work.

## Build recipe

For each tensor in the mask, with `W35` the Qwen3.5 weight and `W36` the Qwen3.6 weight:

```
W_flex = W36 + (alpha - 1) * (W36 - W35)      alpha = 1.25
```

Tensors are dequantized to fp32, updated, and written back as Q8_0. The tensors outside the mask pass through the same dequantize and requantize step with alpha = 1.0. The random and shuffled controls use the same step, so the baseline-level scores of those controls show the pipeline itself does not move the result.

## Grading

Absolute scores use a corrected grader. The original eval host lacked a working `math_verify` and used a boxed-answer regex that fails on nested braces, which marked about 8% of correct answers wrong in every run. All nine runs were regraded locally with balanced-brace extraction, Hendrycks-style normalization, and `math_verify`. The regrade only turns false negatives into passes, and it added 35 to 43 passes to every run, so comparisons between runs are unaffected. Uncorrected counts are 435 for Qwen3.6 and 441 for FLEX.

The "8,000-token budget" is a filter on the 32,768-cap runs: a problem counts as solved if it is correct and its completion is at most 8,000 tokens. With greedy decoding this matches running with an 8,000-token limit. At 8,192 tokens the counts are 300 and 370.

## Scope and future work

- **Models.** One same-lineage pair, Qwen3.5 to Qwen3.6 (35B-A3B), built from Q8_0 GGUF files. Other model families and native BF16 weights are future work.
- **Choice of alpha and mask.** Both were chosen by hand. A per-family or per-layer search for alpha is future work.
- **Router.** The router is included in the mask. Extrapolating experts alone, and routing alone, are future work.
- **Beyond MATH-500.** These runs used the whole-model alpha = 1.25 build, not the FFN-only build: MBPP 78 to 82 of 100, HumanEval 152 to 148 of 164, GSM8K 47 to 46 of 50. FFN-only runs on these benchmarks are future work. On Terminal-Bench (89 tasks) the score goes from 42 to 37: across all 89, FLEX solves 10 that Qwen3.6 missed and loses 15 it had solved (two-sided sign test p ≈ 0.42, so no real difference). A task-by-task look at those 25 is future work.
- **Why it works.** The result shows longer derivations finishing. Whether the update teaches shorter reasoning or changes termination behavior, and what the 12 problems FLEX loses have in common (10 of them hit the token cap under FLEX), are open questions.

## Repo contents

- `figures/`: concept diagram, budget sweep, controls panel, per-problem rescue scatter
- `data/per_problem.csv`: one row per problem per run (9 runs x 500 problems): finish reason, token counts, original and corrected grading
- `code/build_flex.py`: builds a FLEX checkpoint from two GGUF files
- `code/eval_math500.py`: the MATH-500 runner used against `llama-server`
- `code/grade.py`: the corrected grader
- `code/export_per_problem.py`: flattens raw result files into the CSV
- `code/paired_stats.py`: every p-value and confidence interval above, computed from the CSV
- `tests/test_flex.py`: tests for the mask, the formula, the controls, a small end-to-end build, and the published numbers
- `REPRODUCE.md`: commands for each level of reproduction

Every number in this README can be recomputed with `python code/paired_stats.py --all`.

## Citation

Ruchit Patel, "FLEX: Family-Localized Weight-Space Extrapolation," independent research, 2026.
