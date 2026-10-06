"""Build a FLEX checkpoint: extrapolate selected tensors along a version delta.

For every tensor that exists in both GGUF files with the same shape:

    W_out = W_new + (alpha - 1) * (W_new - W_old)

where W_old is the older checkpoint (Qwen3.5) and W_new the newer endpoint
(Qwen3.6). alpha = 1.0 returns the newer endpoint unchanged.

Tensor selection:
  --alpha-mlp A   tensors whose name contains "ffn_" use A (the FLEX mask):
                  routed experts (ffn_*_exps), shared experts (ffn_*_shexp),
                  the router (ffn_gate_inp) and the shared-expert gate.
  --alpha-attn A  attention weights (blk.N.attn_{q,k,v,qkv,output,gate}.weight)
  --alpha-embed A token_embd.weight and output.weight
                  (embed has priority over attn, attn over the FFN mask)
  --alpha A       every other tensor uses A (default 1.0, i.e. untouched).

Controls (applied to the tensors that receive a non-trivial alpha):
  --control-mode random    replace the delta with Gaussian noise of the same
                           Frobenius norm, seeded per tensor name
  --control-mode shuffle   randomly permute the delta's entries within the tensor

All tensors are dequantized to fp32, updated, and written back in --out-qtype
(Q8_0 by default). F32 source tensors stay F32.

Example (the FLEX artifact):
    python build_flex.py --old Qwen3.5-35B-A3B-Q8_0.gguf \
        --new Qwen3.6-35B-A3B-Q8_0.gguf --out qwen36_mlp_1p25.Q8_0.gguf \
        --alpha 1.0 --alpha-mlp 1.25
"""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import dequantize, quantize

FFN_NAME = re.compile(r"\bffn_")
ATTN_NAME = re.compile(r"^blk\.\d+\.attn_(q|k|v|qkv|output|gate)\.weight$")
EMBED_NAME = re.compile(r"^(token_embd|output)\.weight$")

QTYPES = {
    "Q8_0": GGMLQuantizationType.Q8_0,
    "F16": GGMLQuantizationType.F16,
    "BF16": GGMLQuantizationType.BF16,
    "F32": GGMLQuantizationType.F32,
}

_CFG: dict = {}
_OLD: dict = {}
_NEW: dict = {}


def select_alpha(name: str, alpha: float, alpha_mlp: float | None,
                 control_mode: str = "none", alpha_attn: float | None = None,
                 alpha_embed: float | None = None) -> tuple[float, bool]:
    """Return (alpha for this tensor, whether the control delta applies)."""
    if alpha_embed is not None and EMBED_NAME.match(name):
        return alpha_embed, alpha_embed != 0.0
    if alpha_attn is not None and ATTN_NAME.match(name):
        return alpha_attn, alpha_attn != 0.0
    if alpha_mlp is not None and FFN_NAME.search(name):
        return alpha_mlp, alpha_mlp != 0.0
    has_override = any(a is not None for a in (alpha_mlp, alpha_attn, alpha_embed))
    controlled = control_mode != "none" and alpha != 0.0 and not has_override
    return alpha, controlled


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def control_delta(delta: np.ndarray, name: str, mode: str, seed: int) -> np.ndarray:
    """Fake delta with the same Frobenius norm as the real one."""
    rng = np.random.default_rng(stable_seed(seed, name))
    if mode == "shuffle":
        flat = delta.reshape(-1).copy()
        rng.shuffle(flat)
        return flat.reshape(delta.shape).astype(np.float32, copy=False)
    if mode == "random":
        fake = rng.standard_normal(delta.shape, dtype=np.float32)
        delta_norm = float(np.linalg.norm(delta))
        fake_norm = float(np.linalg.norm(fake))
        if delta_norm == 0.0 or fake_norm == 0.0:
            return np.zeros_like(delta, dtype=np.float32)
        fake *= delta_norm / fake_norm
        return fake.astype(np.float32, copy=False)
    return delta


def extrapolate(w_old: np.ndarray, w_new: np.ndarray, alpha: float,
                fake_delta: np.ndarray | None = None) -> np.ndarray:
    delta = w_new - w_old if fake_delta is None else fake_delta
    return w_new + (alpha - 1.0) * delta


def to_fp32(tensor) -> np.ndarray:
    arr = dequantize(tensor.data, tensor.tensor_type)
    arr = arr.reshape(tuple(int(s) for s in reversed(tensor.shape)))
    return arr.astype(np.float32)


def worker_init(old_path, new_path, cfg):
    _CFG.update(cfg)
    _OLD.update({t.name: t for t in GGUFReader(old_path).tensors})
    _NEW.update({t.name: t for t in GGUFReader(new_path).tensors})


def work(name: str):
    t_old, t_new = _OLD[name], _NEW[name]
    w_old, w_new = to_fp32(t_old), to_fp32(t_new)
    if w_old.shape != w_new.shape:
        w_out = w_new
    else:
        alpha, controlled = select_alpha(
            name, _CFG["alpha"], _CFG["alpha_mlp"], _CFG["control_mode"],
            _CFG["alpha_attn"], _CFG["alpha_embed"])
        fake = None
        if _CFG["control_mode"] != "none" and controlled:
            fake = control_delta(w_new - w_old, name, _CFG["control_mode"],
                                 _CFG["control_seed"])
        w_out = extrapolate(w_old, w_new, alpha, fake)
    if t_old.tensor_type == GGMLQuantizationType.F32:
        out = w_out.astype(np.float32)
        return name, out, tuple(int(s) for s in out.shape), int(GGMLQuantizationType.F32)
    qtype = _CFG["out_qtype"]
    packed = quantize(w_out.astype(np.float32), qtype)
    if not isinstance(packed, np.ndarray):
        packed = np.frombuffer(packed, dtype=np.uint8)
    return name, packed, tuple(int(s) for s in packed.shape), int(qtype)


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    scalar_writers = {
        GGUFValueType.UINT8: writer.add_uint8, GGUFValueType.INT8: writer.add_int8,
        GGUFValueType.UINT16: writer.add_uint16, GGUFValueType.INT16: writer.add_int16,
        GGUFValueType.UINT32: writer.add_uint32, GGUFValueType.INT32: writer.add_int32,
        GGUFValueType.UINT64: writer.add_uint64, GGUFValueType.INT64: writer.add_int64,
        GGUFValueType.FLOAT32: writer.add_float32, GGUFValueType.FLOAT64: writer.add_float64,
        GGUFValueType.BOOL: lambda n, v: writer.add_bool(n, bool(v)),
    }
    for key, field in reader.fields.items():
        if key in skip or not field.types:
            continue
        kind = field.types[0]
        try:
            if kind == GGUFValueType.ARRAY:
                sub = field.types[1]
                vals = []
                for idx in field.data:
                    part = field.parts[idx]
                    if sub == GGUFValueType.STRING:
                        vals.append(bytes(part).decode("utf-8"))
                    else:
                        v = part.tolist() if hasattr(part, "tolist") else part
                        vals.append(v[0] if isinstance(v, list) and len(v) == 1 else v)
                writer.add_array(key, vals)
            elif kind == GGUFValueType.STRING:
                writer.add_string(key, bytes(field.parts[field.data[0]]).decode("utf-8"))
            else:
                v = field.parts[field.data[0]]
                v = v.item() if hasattr(v, "item") else v
                if kind in scalar_writers:
                    scalar_writers[kind](key, v)
        except Exception as exc:  # keep going; report which key was skipped
            print(f"  [meta] skipped {key}: {exc}")


def build(args) -> None:
    t0 = time.time()
    out_qtype = QTYPES[args.out_qtype]
    r_old, r_new = GGUFReader(args.old), GGUFReader(args.new)
    old_names = {t.name for t in r_old.tensors}
    new_names = {t.name for t in r_new.tensors}
    common = sorted(old_names & new_names)
    only_new = sorted(new_names - old_names)
    print(f"[flex] old={len(old_names)} new={len(new_names)} common={len(common)} "
          f"only_new={len(only_new)}")

    arch_field = r_old.fields.get("general.architecture")
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(args.out, arch, use_temp_file=True)
    copy_metadata(r_old, writer)

    # tensors only present in the newer file (e.g. rope_freqs) are copied as is
    new_map = {t.name: t for t in r_new.tensors}
    for name in only_new:
        t = new_map[name]
        arr = to_fp32(t)
        if t.tensor_type == GGMLQuantizationType.F32:
            writer.add_tensor(name, arr, raw_shape=tuple(int(s) for s in arr.shape),
                              raw_dtype=GGMLQuantizationType.F32)
        else:
            packed = quantize(arr, out_qtype)
            if not isinstance(packed, np.ndarray):
                packed = np.frombuffer(packed, dtype=np.uint8)
            writer.add_tensor(name, packed, raw_shape=tuple(int(s) for s in packed.shape),
                              raw_dtype=out_qtype)
    del r_old, r_new

    cfg = {"alpha": args.alpha, "alpha_mlp": args.alpha_mlp,
           "alpha_attn": args.alpha_attn, "alpha_embed": args.alpha_embed,
           "control_mode": args.control_mode, "control_seed": args.control_seed,
           "out_qtype": out_qtype}
    with Pool(args.workers, worker_init, (args.old, args.new, cfg)) as pool:
        for done, (name, arr, raw_shape, dtype) in enumerate(
                pool.imap(work, common, chunksize=1), 1):
            writer.add_tensor(name, arr, raw_shape=raw_shape,
                              raw_dtype=GGMLQuantizationType(dtype))
            if done % 50 == 0 or done == len(common):
                print(f"  [{done}/{len(common)}] {(time.time() - t0) / 60:.1f} min", flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[flex] wrote {args.out} ({Path(args.out).stat().st_size / 1e9:.1f} GB) "
          f"in {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="older checkpoint GGUF (Qwen3.5)")
    ap.add_argument("--new", required=True, help="newer endpoint GGUF (Qwen3.6)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="alpha for tensors outside the FFN mask (1.0 = unchanged)")
    ap.add_argument("--alpha-mlp", type=float, default=None,
                    help="alpha for tensors whose name contains ffn_")
    ap.add_argument("--alpha-attn", type=float, default=None,
                    help="alpha for attention weight tensors")
    ap.add_argument("--alpha-embed", type=float, default=None,
                    help="alpha for token_embd.weight and output.weight")
    ap.add_argument("--control-mode", choices=["none", "random", "shuffle"], default="none")
    ap.add_argument("--control-seed", type=int, default=0)
    ap.add_argument("--out-qtype", choices=list(QTYPES), default="Q8_0")
    ap.add_argument("--workers", type=int, default=8)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
