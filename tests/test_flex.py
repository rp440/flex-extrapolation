"""Tests for the FLEX build, selection mask, and published statistics.

Run from the repo root:  python -m pytest tests -q
Needs: numpy, gguf, pytest, math-verify (grade tests), sympy, requests.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

import build_flex as bf  # noqa: E402
import paired_stats as ps  # noqa: E402


# ---------------------------------------------------------------- selection mask
FFN = ["blk.3.ffn_up_exps.weight", "blk.3.ffn_gate_exps.weight", "blk.3.ffn_down_exps.weight",
       "blk.3.ffn_up_shexp.weight", "blk.3.ffn_gate_shexp.weight", "blk.3.ffn_down_shexp.weight",
       "blk.3.ffn_gate_inp.weight", "blk.3.ffn_gate_inp_shexp.weight"]
NOT_FFN = ["blk.3.attn_q.weight", "blk.3.attn_output.weight", "blk.3.attn_norm.weight",
           "blk.3.ssm_a", "token_embd.weight", "output.weight", "output_norm.weight"]


@pytest.mark.parametrize("name", FFN)
def test_ffn_mask_selects_every_ffn_tensor_including_router(name):
    assert bf.select_alpha(name, 1.0, 1.25) == (1.25, True)


@pytest.mark.parametrize("name", NOT_FFN)
def test_ffn_mask_leaves_everything_else_alone(name):
    assert bf.select_alpha(name, 1.0, 1.25) == (1.0, False)


def test_attention_and_embedding_options_take_priority_over_ffn_mask():
    assert bf.select_alpha("blk.0.attn_q.weight", 1.0, 1.25, alpha_attn=1.1) == (1.1, True)
    assert bf.select_alpha("blk.0.attn_norm.weight", 1.0, None, alpha_attn=1.1) == (1.0, False)
    assert bf.select_alpha("token_embd.weight", 1.0, 1.25, alpha_embed=0.9) == (0.9, True)
    assert bf.select_alpha("output_norm.weight", 1.0, None, alpha_embed=0.9) == (1.0, False)


def test_global_alpha_applies_to_all_tensors():
    for name in FFN + NOT_FFN:
        assert bf.select_alpha(name, 1.25, None)[0] == 1.25


# ---------------------------------------------------------------- the formula
def test_extrapolate_formula_and_endpoints():
    rng = np.random.default_rng(0)
    w_old = rng.standard_normal((8, 16)).astype(np.float32)
    w_new = rng.standard_normal((8, 16)).astype(np.float32)
    assert np.allclose(bf.extrapolate(w_old, w_new, 1.0), w_new)
    assert np.allclose(bf.extrapolate(w_old, w_new, 0.0), w_old)
    assert np.allclose(bf.extrapolate(w_old, w_new, 1.25), w_new + 0.25 * (w_new - w_old), atol=1e-6)


def test_controls_keep_norm_and_differ_from_real_delta():
    rng = np.random.default_rng(1)
    delta = rng.standard_normal((16, 32)).astype(np.float32)
    for mode in ("random", "shuffle"):
        fake = bf.control_delta(delta, "blk.0.ffn_up_exps.weight", mode, 125)
        assert fake.shape == delta.shape
        assert np.isclose(np.linalg.norm(fake), np.linalg.norm(delta), rtol=1e-4)
        assert not np.allclose(fake, delta)
        again = bf.control_delta(delta, "blk.0.ffn_up_exps.weight", mode, 125)
        assert np.array_equal(fake, again)  # deterministic per (seed, tensor name)
    shuf = bf.control_delta(delta, "x", "shuffle", 125)
    assert np.array_equal(np.sort(shuf.ravel()), np.sort(delta.ravel()))  # same entries


# ---------------------------------------------------------------- end to end on tiny GGUFs
TENSORS = {
    "blk.0.ffn_up_exps.weight": ("q8", (4, 32, 64)),
    "blk.0.ffn_gate_shexp.weight": ("q8", (32, 64)),
    "blk.0.ffn_gate_inp.weight": ("f32", (4, 64)),
    "blk.0.attn_q.weight": ("q8", (32, 64)),
    "blk.0.attn_norm.weight": ("f32", (64,)),
    "token_embd.weight": ("q8", (32, 64)),
}


def _write_gguf(path, seed):
    rng = np.random.default_rng(seed)
    w = GGUFWriter(str(path), "qwen3moe")
    w.add_uint32("qwen3moe.block_count", 1)
    w.add_string("general.name", "tiny")
    for name, (kind, shape) in TENSORS.items():
        arr = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        if kind == "q8":
            packed = quantize(arr, Q.Q8_0)
            w.add_tensor(name, packed, raw_shape=packed.shape, raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(name, arr, raw_shape=arr.shape, raw_dtype=Q.F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def _read(path):
    out = {}
    for t in GGUFReader(str(path)).tensors:
        arr = dequantize(t.data, t.tensor_type).astype(np.float32)
        out[t.name] = arr.reshape(tuple(int(s) for s in reversed(t.shape)))
    return out


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("gguf")
    _write_gguf(d / "old.gguf", 1)
    _write_gguf(d / "new.gguf", 2)

    def build(name, *flags):
        out = d / f"{name}.gguf"
        sys.argv = ["build_flex.py", "--old", str(d / "old.gguf"), "--new", str(d / "new.gguf"),
                       "--out", str(out), "--workers", "1", *flags]
        bf.main()
        return _read(out)

    return {"old": _read(d / "old.gguf"), "new": _read(d / "new.gguf"),
            "flex": build("flex", "--alpha", "1.0", "--alpha-mlp", "1.25"),
            "attn": build("attn", "--alpha-attn", "1.25"),
            "rand": build("rand", "--alpha-mlp", "1.25", "--control-mode", "random",
                          "--control-seed", "125")}


Q8_TOL = 0.01  # Q8_0 roundtrip error on weights of scale 0.1


def test_flex_build_moves_only_ffn_tensors(built):
    old, new, flex = built["old"], built["new"], built["flex"]
    for name in TENSORS:
        want = new[name] + 0.25 * (new[name] - old[name]) if "ffn_" in name else new[name]
        assert np.abs(flex[name] - want).max() < Q8_TOL, name
    # the FFN tensors really moved, the others did not
    assert np.abs(flex["blk.0.ffn_up_exps.weight"] - new["blk.0.ffn_up_exps.weight"]).max() > 0.02
    assert np.abs(flex["blk.0.attn_q.weight"] - new["blk.0.attn_q.weight"]).max() < Q8_TOL
    assert np.abs(flex["token_embd.weight"] - new["token_embd.weight"]).max() < Q8_TOL


def test_router_is_part_of_the_mask(built):
    old, new, flex = built["old"], built["new"], built["flex"]
    name = "blk.0.ffn_gate_inp.weight"
    assert np.allclose(flex[name], new[name] + 0.25 * (new[name] - old[name]), atol=1e-6)


def test_attention_option_actually_changes_attention(built):
    old, new, attn = built["old"], built["new"], built["attn"]
    name = "blk.0.attn_q.weight"
    assert np.abs(attn[name] - new[name]).max() > 0.02
    assert np.abs(attn[name] - (new[name] + 0.25 * (new[name] - old[name]))).max() < Q8_TOL
    assert np.abs(attn["blk.0.ffn_up_exps.weight"] - new["blk.0.ffn_up_exps.weight"]).max() < Q8_TOL


def test_random_control_has_flex_sized_change_but_wrong_direction(built):
    old, new, flex, rand = built["old"], built["new"], built["flex"], built["rand"]
    name = "blk.0.ffn_up_exps.weight"
    step_real = np.linalg.norm(flex[name] - new[name])
    step_fake = np.linalg.norm(rand[name] - new[name])
    assert np.isclose(step_real, step_fake, rtol=0.05)
    cos = np.sum((flex[name] - new[name]) * (rand[name] - new[name])) / (step_real * step_fake)
    assert abs(cos) < 0.1


# ---------------------------------------------------------------- published numbers
def test_headline_numbers_reproduce_from_the_published_csv():
    runs = ps.load(ROOT / "data" / "per_problem.csv")
    c = ps.compare(runs["qwen36"], runs["flex_ffn_1p25"], 8000)
    assert (c["solved_base"], c["solved_cand"]) == (294, 369)
    assert (c["only_cand"], c["only_base"]) == (98, 23)
    assert c["p"] < 1e-11
    assert (c["capped_base"], c["capped_cand"]) == (52, 22)
    u = ps.compare(runs["qwen36"], runs["flex_ffn_1p25"], None)
    assert (u["solved_base"], u["solved_cand"], u["only_cand"], u["only_base"]) == (475, 483, 20, 12)
