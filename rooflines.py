"""Explicit, reproducible capacity/roofline models. No empirical time measurements."""
import json
import math
from pathlib import Path


def model():
    rows = []
    for n in range(5, 10):
        states = 2 * 462 * math.prod(range(62 - (n - 2) + 1, 63))
        rows.append(dict(pieces=n, states=states, wdl_two_buffers_gb=states*.5/1e9,
                         one_16bit_distance_gb=states*2/1e9,
                         king_slice_two_buffers_gb=states*.5/462/1e9,
                         hours_200_sweeps={str(rate): states*200/rate/3600 for rate in [.25e9, 1e9, 5e9]}))
    fp = []
    for reuse in [1, 4, 16, 64]:
        # FP16 adjacency 64x64, one FP16 query vector, FP32 temporary counts.
        # This favorable model omits the padding for narrow N and relation creation.
        bytes_per_row = 128/reuse + 2 + 4
        fp.append(dict(relation_reuse=reuse, bytes_per_row=bytes_per_row,
                       hbm_rows_per_s=3.35e12/bytes_per_row,
                       useful_fp16_compute_rows_per_s=989.5e12/128,
                       compute_rows_per_s_if_one_query_padded_to_16=989.5e12/(128*16)))
    return dict(h100_sxm_memory_gb=80, hbm_tb_per_s=3.35,
                dense_fp16_tflops=989.5, capacity=rows, fp_relation_model=fp,
                warning='Bounds only. Actual kernels must count padding, packing, occupancy and relation construction.')


if __name__ == '__main__':
    data=model(); Path('rooflines.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data, indent=2))
