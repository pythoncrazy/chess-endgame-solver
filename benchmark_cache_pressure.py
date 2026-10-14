"""Measured page eviction and origin reads. No assumed production hit rates."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from lookup_storage import BatchReader
from singleflight_store import SingleFlightStore
from aligned_ranges import AlignedRangeStore
from r2_blocks import LocalRangeStore


def run(directory,trace):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    rows=[]
    for strategy in ('page-cache-only','compressed-range-cache','aligned-compressed-cache'):
        origin=LocalRangeStore(directory);store=origin;compressed=None
        if strategy!='page-cache-only':
            compressed=SingleFlightStore(origin,cache_bytes=1<<20);store=compressed
        if strategy=='aligned-compressed-cache':
            store=AlignedRangeStore(compressed,{o['key']:o['bytes'] for o in manifest['objects']},tile_bytes=16384)
        reader=BatchReader(manifest,store,data_cache_bytes=2*65536,index_cache_bytes=4096,
                           directory_cache_bytes=4096,max_batch_decoded_bytes=8<<20)
        start=time.perf_counter();answers=[];timings=[]
        for ids in trace:
            t=time.perf_counter();answers.append(reader.probe_many(ids));timings.append(time.perf_counter()-t)
        row=dict(strategy=strategy,logical_positions=sum(map(len,trace)),batches=len(trace),
            origin_reads=origin.requests,origin_bytes=origin.bytes_read,decoded_cache_bytes=reader.data.size,
            decoded_pages=reader.stats['decoded_pages'],packed_pages=reader.stats['packed_pages'],data_cache_hits=reader.stats['data_cache_hits'],
            local_batch_median_ms=float(np.median(timings)*1000),local_batch_p95_ms=float(np.percentile(timings,95)*1000),
            local_elapsed_seconds=time.perf_counter()-start)
        if compressed:row.update(compressed_cache_bytes=compressed.cache.size,compressed_cache_hits=compressed.cache_hits)
        rows.append((row,np.concatenate(answers)));print(json.dumps(row),flush=True)
    for _,values in rows[1:]:assert np.array_equal(values,rows[0][1])
    return [row for row,_ in rows]


def main(directory,output='cache_pressure_benchmark.json'):
    manifest=json.loads((Path(directory)/'manifest.json').read_text());rng=np.random.default_rng(7015)
    ids=rng.integers(0,manifest['positions'],size=2048,dtype=np.int64)
    uniform=[ids[b:b+16] for b in range(0,len(ids),16)]
    clustered=[]
    for _ in range(128):
        begin=int(rng.integers(0,manifest['positions']-512));clustered.append(begin+rng.integers(0,512,16))
    result=dict(status='pass',dataset='complete native KPvKP WDL stream',archive_format=manifest['format'],
        workload_uniform=run(directory,uniform),workload_clustered=run(directory,clustered),
        warning='Synthetic index traces over one real four-piece material, not observed game or fleet traffic. Local CPU timings are not network latency. Working sets/encoded data fit the configured compressed cache; no eight-piece cache-hit extrapolation.',
        live_r2_tested=False,h100_benchmarked=False)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'cache_pressure_benchmark.json')
