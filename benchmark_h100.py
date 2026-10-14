"""Run exact operator microbenchmarks on an available CUDA GPU.

Requires numpy, torch, triton. Compares packed Boolean rows with FP16/FP32
tensor-core reductions. Input legal relations are synthetic and resident.
Reported rates exclude chess indexing, relation construction, graph scheduling,
packing, disk I/O and distance computation. They are not generator throughput.
FP queries=16 is only applicable if 16 useful state vectors share one relation.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def bit_kernel(A, W, L, O, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < ROWS
        mask = tl.load(A + i, ok, other=0)
        winbits = tl.load(W + i // 64, ok, other=0)
        lossbits = tl.load(L + i // 64, ok, other=0)
        win = (mask & lossbits) != 0
        loss = (mask != 0) & ((mask & ~winbits) == 0)
        tl.store(O + i, tl.where(win, 1, tl.where(loss, 2, 0)).to(tl.uint8), ok)

    @triton.jit
    def fp_kernel(A, E, O, QUERIES: tl.constexpr):
        g = tl.program_id(0)
        r = tl.arange(0, 64)
        k = tl.arange(0, 64)
        q = tl.arange(0, 16)
        a = tl.load(A + g * 4096 + r[:, None] * 64 + k[None, :])
        e = tl.load(E + g * 64 * QUERIES + k[:, None] * QUERIES + q[None, :], q[None, :] < QUERIES, other=0)
        z = tl.dot(a, e, out_dtype=tl.float32).to(tl.int32)
        deg = tl.sum(a.to(tl.int32), 1)
        win = z % 65 != 0
        loss = (deg[:, None] > 0) & (z == 0)
        result = tl.where(win, 1, tl.where(loss, 2, 0)).to(tl.uint8)
        tl.store(O + g * 64 * QUERIES + r[:, None] * QUERIES + q[None, :], result, q[None, :] < QUERIES)

    p = argparse.ArgumentParser()
    p.add_argument('--groups', type=int, default=8192)
    p.add_argument('--output', default='h100_benchmark.json')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required; no benchmark was performed.')
    rng = np.random.default_rng(1292)
    groups = args.groups
    a = rng.random((groups, 64, 64)) < .18
    # For some rows only a winning child is available; include degree-zero rows.
    a[:, 0] = False
    a[:, 1] = False; a[:, 1, 0] = True
    bits = np.packbits(a, axis=-1, bitorder='little').copy().view('<u8').reshape(groups, 64)
    state = rng.integers(0, 3, size=(groups, 64), dtype=np.uint8)
    state[:, 0] = 1
    wb = np.packbits(state == 1, axis=-1, bitorder='little').copy().view('<u8').reshape(groups)
    lb = np.packbits(state == 2, axis=-1, bitorder='little').copy().view('<u8').reshape(groups)
    device = lambda x: torch.from_numpy(np.ascontiguousarray(x)).cuda()
    ab, w, l = device(bits), device(wb), device(lb)
    af = device(a.astype(np.float16))
    out = torch.empty(groups * 64, device='cuda', dtype=torch.uint8)
    run_bit = lambda: bit_kernel[(triton.cdiv(groups * 64, 256),)](ab, w, l, out, groups * 64, 256)
    run_bit()
    expected = np.where((a & (state == 2)[:, None, :]).any(-1), 1,
                        np.where(a.any(-1) & ~(a & (state != 1)[:, None, :]).any(-1), 2, 0)).astype(np.uint8)
    assert np.array_equal(out.cpu().numpy().reshape(groups, 64), expected)
    bit_ms = triton.testing.do_bench(run_bit)
    results = [dict(kernel='packed_bitset', queries=1, ms=bit_ms,
                    useful_row_queries_per_s=groups * 64 / (bit_ms / 1000))]
    for queries in (1, 16):
        states = np.repeat(state[:, :, None], queries, axis=2)
        if queries > 1:
            states[:, :, 1:] = rng.integers(0, 3, size=(groups, 64, queries - 1), dtype=np.uint8)
        encoded = (65 * (states != 1) + (states == 2)).astype(np.float16)
        e = device(encoded)
        fpout = torch.empty((groups, 64, queries), device='cuda', dtype=torch.uint8)
        run_fp = lambda: fp_kernel[(groups,)](af, e, fpout, queries)
        run_fp()
        want = np.empty((groups, 64, queries), np.uint8)
        for q in range(queries):
            s = states[:, :, q]
            want[:, :, q] = np.where((a & (s == 2)[:, None, :]).any(-1), 1,
                          np.where(a.any(-1) & ~(a & (s != 1)[:, None, :]).any(-1), 2, 0))
        assert np.array_equal(fpout.cpu().numpy(), want)
        ms = triton.testing.do_bench(run_fp)
        results.append(dict(kernel='fp16_fp32_encoded', queries=queries, ms=ms,
                       useful_row_queries_per_s=groups * 64 * queries / (ms / 1000),
                       padded_tensor_columns=16))
    data = dict(gpu=torch.cuda.get_device_name(), total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                torch_version=torch.__version__, triton_version=triton.__version__, groups=groups,
                microbenchmark_only=True, includes_move_generation=False, results=results)
    Path(args.output).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__': main()
