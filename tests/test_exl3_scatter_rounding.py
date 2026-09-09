# SPDX-License-Identifier: Apache-2.0
"""Discover the installed E2 scatter epilogue's rounding; run with serving stopped.

    python tests/test_exl3_scatter_rounding.py [--device cuda:0] [--expected fma|separate]

No extension loader/build, E3 kernel, reconstructed GEMM, or timing is involved.
The ABI comes from .auto/current/installed-exl3-fat-gemm.cu (M64), not the
repository's M128 implementation. K4 MCG packed int16 generation follows
_check_fat_kernel in test_exl3_overlay.py. `a` is already the native GEMM input;
no input Hadamard or FP16 output conversion is inserted.

Weight-one scatter into zero obtains the native post-SVH FP32 value v. With
p = RN32(v*w) and initial output -p, separate RN32 multiply/add gives +0;
fused RN32(v*w-p) gives the product-rounding residual. FP32 x FP16 has at
most 24+11=35 significant bits. FP64 represents that product exactly, and
subtracting its nearby FP32 rounding is exact (Sterbenz); the final FP32
cast therefore supplies the round-to-nearest-even FMA oracle, not an
ordinary, potentially contracted Python/GPU expression.

All oracle arithmetic runs eagerly on CPU. Identifying lanes require normal,
finite operands/products and nonzero normal residuals. Exact-zero lanes are
also compared, but cannot identify contraction. This does NOT establish FTZ,
NaN payload, exceptional-value, other rounding-mode, or multi-expert ordering
contracts. Results describe only the loaded binary/device and sampled shapes.
Importing this module does not import torch or execute GPU work.
"""

import argparse
import json


# M tails exercise the installed 16-row subtiles and 64-row CTAs. K is a
# multiple of 16 (including odd double-buffer iteration counts); N must be
# divisible by 128, so an arbitrary N tail would be an invalid native call.
SHAPES = (
    (1, 16, 128),
    (15, 32, 256),
    (16, 48, 128),
    (17, 128, 384),
    (63, 256, 128),
    (64, 48, 256),
    (65, 128, 384),
    (129, 256, 256),
)
# Exactly representable, normal FP16 values; no powers of two or +/-1.
WEIGHTS = (0.333251953125, -0.333251953125, 0.75048828125,
           -0.75048828125, 1.0009765625, -1.0009765625)


def run_probe(device="cuda:0", expected=None):
    """Return JSON-safe evidence; `ok` gates discovery and optional expectation.

    Missing CUDA/native ABI or invalid/nondiscriminatory fixtures raise
    RuntimeError. A neither/mixed match or expectation mismatch returns ok=False.
    This function deliberately performs GPU work only when explicitly called.
    """
    if expected not in (None, "fma", "separate"):
        raise ValueError("expected must be None, 'fma', or 'separate'")

    import torch
    import exllamav3_ext as ext  # Import the installed binary directly; never build.

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the probe requires an explicitly available CUDA device")
    scatter = getattr(ext, "exl3_fat_gemm_scatter", None)
    if scatter is None:
        raise RuntimeError("installed extension lacks exl3_fat_gemm_scatter")

    def require(condition, message):
        if not bool(condition):
            raise RuntimeError(message)

    def normal_or_zero(tensor, label):
        tiny = torch.finfo(torch.float32).tiny
        require(torch.all(torch.isfinite(tensor)), f"{label}: nonfinite fixture")
        require(torch.all((tensor == 0) | (tensor.abs() >= tiny)),
                f"{label}: subnormal fixture would confound contraction with FTZ")

    def bits(tensor):
        return tensor.contiguous().view(torch.int32)

    def hex32(tensor, row, col):
        return f"0x{int(bits(tensor)[row, col]) & 0xffffffff:08x}"

    records = []
    for shape_index, (m, k, n) in enumerate(SHAPES):
        seed = 41270 + shape_index
        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Bounded nonzero normal FP16 inputs, with both signs and mantissa bits.
        a_cpu = torch.randint(1, 1024, (m, k), generator=gen).float() / 1024
        signs = torch.randint(0, 2, (m, k), generator=gen) * 2 - 1
        a_cpu = (a_cpu * signs).half()
        trellis_cpu = torch.randint(
            -30000, 30000, (k // 16, n // 16, 64),
            dtype=torch.int16, generator=gen,
        )
        scales = torch.tensor([0.625, -0.875, 1.125, -1.375], dtype=torch.float16)
        svh_cpu = scales.repeat(n // len(scales))
        a, trellis, svh = (t.to(device) for t in (a_cpu, trellis_cpu, svh_cpu))
        token_idx = torch.arange(m, dtype=torch.int64, device=device)
        ones = torch.ones(m, dtype=torch.float16, device=device)

        def native(out, weight):
            # Identity indices: exactly one writer per destination, no atomics
            # or route collisions. Reuse the identical original a/trellis/svh.
            scatter(a, trellis, out, svh, token_idx, weight, 4, True, False)

        unweighted = torch.zeros((m, n), dtype=torch.float32, device=device)
        native(unweighted, ones)
        value = unweighted.cpu()  # Synchronous copy also surfaces launch errors.
        normal_or_zero(value, f"{(m, k, n)} post-SVH value")
        repeat = torch.zeros_like(unweighted)
        native(repeat, ones)
        require(torch.equal(bits(value), bits(repeat.cpu())),
                f"{(m, k, n)}: weight-one native scatter is not bitwise repeatable")

        for weight_number in WEIGHTS:
            weight_cpu = torch.full((m,), weight_number, dtype=torch.float16)
            require(float(weight_cpu[0]) == weight_number, "weight must be exactly FP16")
            weight = weight_cpu.to(device)
            # Materialize CPU tensors at each step: no compiler can contract
            # the separate oracle. Double starts from FP32 v and actual FP16 w.
            product64 = value.double() * weight_cpu.double().unsqueeze(1)
            rounded_product = product64.float()
            initial = -rounded_product
            # The weight-one/+0 probe cannot recover a hidden -0. Use +0
            # for zero lanes so that lost sign cannot confound contraction.
            initial.masked_fill_(value == 0, 0.0)
            separate = rounded_product + initial
            residual64 = product64 + initial.double()
            fma = residual64.float()
            normal_or_zero(product64, "exact product")
            normal_or_zero(rounded_product, "rounded product")
            normal_or_zero(initial, "initial output")
            normal_or_zero(fma, "FMA residual")
            require(torch.all(bits(separate) == 0), "separate oracle must be positive zero")
            discriminatory = fma != 0
            count = int(discriminatory.sum())
            require(count > 0, f"{(m, k, n)}, weight={weight_number}: no nonzero residual")
            require(torch.all(value[discriminatory] != 0), "identifying values must be normal")

            out = initial.to(device)
            native(out, weight)
            actual = out.cpu()
            # Compare every FP32 bit, not a tolerance or only identifying lanes.
            fma_mismatches = int((bits(actual) != bits(fma)).sum())
            separate_mismatches = int((bits(actual) != bits(separate)).sum())
            contract = ("fma" if fma_mismatches == 0 else
                        "separate" if separate_mismatches == 0 else "neither")
            row, col = discriminatory.nonzero()[0].tolist()
            records.append({
                "shape_mkn": [m, k, n], "seed": seed,
                "weight": weight_number,
                "weight_fp16_bits": f"0x{int(weight_cpu.view(torch.int16)[0]) & 0xffff:04x}",
                "elements": m * n, "discriminatory_elements": count,
                "zero_value_elements": int((value == 0).sum()),
                "contract": contract,
                "fma_bit_mismatches": fma_mismatches,
                "separate_bit_mismatches": separate_mismatches,
                "witness": {
                    "row": row, "column": col,
                    "value": float(value[row, col]),
                    "exact_product_residual": float(residual64[row, col]),
                    "value_bits": hex32(value, row, col),
                    "initial_bits": hex32(initial, row, col),
                    "native_bits": hex32(actual, row, col),
                    "fma_bits": hex32(fma, row, col),
                    "separate_bits": hex32(separate, row, col),
                },
            })
            require(torch.equal(weight.cpu(), weight_cpu), "native mutated route weights")

        for name, gpu, cpu in (("a", a, a_cpu), ("trellis", trellis, trellis_cpu),
                               ("svh", svh, svh_cpu)):
            require(torch.equal(gpu.cpu().view(torch.int16), cpu.view(torch.int16)),
                    f"native mutated {name}")
        require(torch.equal(token_idx.cpu(), torch.arange(m)), "native mutated indices")

    contracts = {record["contract"] for record in records}
    contract = next(iter(contracts)) if len(contracts) == 1 else "mixed"
    return {
        "contract": contract,
        "expected": expected,
        "ok": contract in ("fma", "separate") and expected in (None, contract),
        "extension_path": str(getattr(ext, "__file__", "unknown")),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "case_count": len(records),
        "discriminatory_elements": sum(r["discriminatory_elements"] for r in records),
        "limitations": "Sampled finite-normal RN32 cancellation only; does not establish "
                       "FTZ, NaN payload, exceptional-value, or multi-expert ordering contracts.",
        "cases": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected", choices=("fma", "separate"))
    args = parser.parse_args()
    try:
        result = run_probe(device=args.device, expected=args.expected)
    except (ImportError, RuntimeError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
